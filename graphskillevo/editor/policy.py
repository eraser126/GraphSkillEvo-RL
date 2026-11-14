"""Local causal LLM policy. Optional dependencies are imported only on explicit use."""
from pathlib import Path
import json
import inspect
from contextlib import contextmanager
from .episode import EditorTurn


class ScriptedEditor:
    """Offline contract fixture, never advertised as a learned policy."""
    def __init__(self, responses): self.responses=iter(responses)

    def generate(self,messages,deterministic=False):
        return EditorTurn(next(self.responses),prompt=messages)


class LocalLLMEditor:
    def __init__(self, model, tokenizer, max_new_tokens=512, max_context=8192,
                 gradient_checkpointing=True, logprob_chunk_size=128):
        self.model,self.tokenizer=model,tokenizer
        self.max_new_tokens,self.max_context=max_new_tokens,max_context
        # Keep dropout disabled during sampling AND gradient evaluation of likelihood ratios.
        self.model.eval()
        self.logprob_chunk_size=logprob_chunk_size
        if type(logprob_chunk_size) is not int or logprob_chunk_size<1:
            raise ValueError('Positive log-probability chunk size required')
        self.checkpointing=False
        if gradient_checkpointing and getattr(model,'supports_gradient_checkpointing',False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.checkpointing=True
        base=model.get_base_model() if hasattr(model,'get_base_model') else model
        params=inspect.signature(base.forward).parameters
        self.logits_selector=next((k for k in ('logits_to_keep','num_logits_to_keep') if k in params),None)
        # Transformers 4.50+ can replace values equal to its global defaults
        # with model-specific sampling settings, even with a GenerationConfig.
        self._generation_options = {}
        generate = getattr(base, 'generate', None)
        if generate is not None and 'use_model_defaults' in inspect.signature(generate).parameters:
            self._generation_options['use_model_defaults'] = False

    @contextmanager
    def _forward_mode(self, gradients):
        """Enable checkpoint stack gates without enabling dropout in child blocks.

        Qwen3/GPT2 gate activation checkpointing on the decoder stack's training
        flag. Decoder blocks and attention/dropout remain in eval mode in BOTH
        forward and recomputation, preserving the original sampling distribution.
        """
        self.model.eval()
        owners=[]
        if gradients and self.checkpointing:
            for module in self.model.modules():
                if getattr(module,'gradient_checkpointing',False):
                    owners.append((module,module.training));module.training=True
        try:yield
        finally:
            for module,training in owners:module.training=training

    @classmethod
    def load(cls,path,device='cpu',max_new_tokens=512,max_context=8192,lora_rank=0):
        path=Path(path).resolve()
        if not path.is_dir(): raise ValueError('A local model directory is required; automatic downloads are disabled')
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        adapter=path/'adapter_config.json'
        base=Path(json.loads(adapter.read_text())['base_model_name_or_path']).resolve() if adapter.is_file() else path
        if not base.is_dir(): raise ValueError('LoRA checkpoint requires its original LOCAL base model')
        tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
                                                  torch_dtype=torch.float32 if device=='cpu' else torch.bfloat16)
        model.to(device)
        if adapter.is_file():
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,str(path),is_trainable=True,local_files_only=True)
        elif lora_rank:
            from peft import LoraConfig,get_peft_model
            model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=lora_rank,lora_alpha=2*lora_rank,
                                                 target_modules='all-linear',lora_dropout=0.0))
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.eos_token_id
        if tokenizer.pad_token_id is None: raise ValueError('Tokenizer requires a pad or EOS token')
        return cls(model,tokenizer,max_new_tokens,max_context)

    def generate(self,messages,deterministic=False):
        import torch
        from transformers import GenerationConfig
        if getattr(self.tokenizer,'chat_template',None):
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        else:
            ids=self.tokenizer.encode('\n'.join(x['role']+': '+x['content'] for x in messages)+'\nassistant:',
                                      add_special_tokens=True)
        if not ids or len(ids)+self.max_new_tokens>self.max_context:
            raise ValueError('Editor context budget exceeded; evidence is not silently truncated')
        device=next(self.model.parameters()).device
        inputs=torch.tensor([ids],device=device)
        config=GenerationConfig(max_new_tokens=self.max_new_tokens,do_sample=not deterministic,
                                 temperature=1.0,top_k=0,top_p=1.0,repetition_penalty=1.0,
                                 pad_token_id=self.tokenizer.pad_token_id,
                                 eos_token_id=self.tokenizer.eos_token_id,bos_token_id=self.tokenizer.bos_token_id)
        self.model.eval()
        with torch.no_grad():
            generated=self.model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                                          generation_config=config,**self._generation_options)
            completion=generated[0,len(ids):].tolist()
            if not completion: raise RuntimeError('Empty model completion')
            # Same raw-softmax distribution as sampling: temperature=1, no top-k/p filtering.
            old=self.logps(ids,completion).detach().cpu().tolist()
        return EditorTurn(self.tokenizer.decode(completion,skip_special_tokens=True),ids,completion,old,messages)

    def logps(self,prompt_ids,completion_ids):
        import torch
        if not prompt_ids or not completion_ids: raise ValueError('Cannot score empty token sequence')
        ids=torch.tensor([prompt_ids+completion_ids],device=next(self.model.parameters()).device)
        kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        if self.logits_selector:
            # Include the final (unused) position so causal alignment and full
            # sequence/position semantics are identical to the original forward.
            kwargs[self.logits_selector]=len(completion_ids)+1
        gradients=torch.is_grad_enabled()
        with self._forward_mode(gradients):
            output=self.model(**kwargs)
        logits=output.logits[0]
        logits=logits[:-1] if self.logits_selector else logits[len(prompt_ids)-1:-1]
        labels=ids[0,len(prompt_ids):]
        def selected_logps(scores,targets):
            return torch.log_softmax(scores.float(),dim=-1).gather(-1,targets[:,None]).squeeze(-1)
        chunks=[]
        for start in range(0,len(completion_ids),self.logprob_chunk_size):
            scores=logits[start:start+self.logprob_chunk_size]
            targets=labels[start:start+self.logprob_chunk_size]
            if gradients and scores.requires_grad:
                from torch.utils.checkpoint import checkpoint
                values=checkpoint(selected_logps,scores,targets,use_reentrant=False)
            else:values=selected_logps(scores,targets)
            chunks.append(values)
        return torch.cat(chunks)

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path);self.tokenizer.save_pretrained(path)


class ChatEditor:
    """Untrained API baseline/frozen adaptation only; cannot be passed to GRPO."""
    def __init__(self,max_tokens=2048,timeout=120): self.max_tokens,self.timeout=max_tokens,timeout

    def generate(self,messages,deterministic=False):
        from graphskillevo.model import chat_optimizer
        response,_=chat_optimizer(system=messages[0]['content'],user=messages[1]['content'],
                                  max_completion_tokens=self.max_tokens,timeout=self.timeout,
                                  retries=1,stage='skill_editor_baseline')
        return EditorTurn(response,prompt=messages)
