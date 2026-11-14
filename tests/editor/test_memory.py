"""Numerical equivalence and backward contracts, no optimizer steps or API calls."""
from copy import deepcopy
import pytest
import torch
from graphskillevo.editor.policy import LocalLLMEditor


def dense_logps(model,prompt,completion):
    model.eval();ids=torch.tensor([prompt+completion])
    scores=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,len(prompt)-1:-1].float()
    labels=ids[0,len(prompt):]
    return torch.log_softmax(scores,-1).gather(-1,labels[:,None]).squeeze(-1)


def assert_equivalent(base,prompt,completion):
    reference=deepcopy(base);optimized=deepcopy(base)
    before={n:p.detach().clone() for n,p in optimized.named_parameters()}
    editor=LocalLLMEditor(optimized,None,gradient_checkpointing=True,logprob_chunk_size=2)
    expected=dense_logps(reference,prompt,completion)
    actual=editor.logps(prompt,completion)
    assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
    (-expected.mean()).backward();(-actual.mean()).backward()
    for (name,a),(other,b) in zip(reference.named_parameters(),optimized.named_parameters()):
        assert name==other
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:assert torch.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5),name
        assert torch.equal(b,before[name])
    assert not optimized.training
    assert all(not m.training for m in optimized.modules())
    with torch.no_grad():old=editor.logps(prompt,completion)
    assert torch.allclose(actual.detach(),old,atol=2e-6,rtol=2e-6)
    return editor


def test_checkpointed_gpt2_matches_original_values_and_gradients():
    from transformers import GPT2Config,GPT2LMHeadModel
    torch.manual_seed(14)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=64,n_embd=16,n_layer=2,n_head=2,
                                    resid_pdrop=.7,embd_pdrop=.7,attn_pdrop=.7))
    editor=assert_equivalent(model,[1,4,5,6],[7,8,9,10,11])
    assert editor.checkpointing


def test_qwen3_completion_only_logits_and_gradients():
    transformers=pytest.importorskip('transformers')
    if not hasattr(transformers,'Qwen3Config'):pytest.skip('Qwen3 training environment required')
    torch.manual_seed(19)
    config=transformers.Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                                    num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                                    max_position_embeddings=64,attention_dropout=.6)
    model=transformers.Qwen3ForCausalLM(config)
    editor=assert_equivalent(model,[1,4,5,6,7,8],[9,10,11])
    assert editor.logits_selector=='logits_to_keep'


def test_single_token_alignment_with_empty_checkpoint_branch():
    from transformers import GPT2Config,GPT2LMHeadModel
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=16,n_embd=8,n_layer=1,n_head=1))
    editor=LocalLLMEditor(model,None,gradient_checkpointing=False,logprob_chunk_size=1)
    with torch.no_grad():
        assert torch.allclose(editor.logps([1],[2]),dense_logps(model,[1],[2]))


def test_qwen3_lora_values_and_adapter_gradients_match_dense_reference():
    transformers=pytest.importorskip('transformers')
    peft=pytest.importorskip('peft')
    if not hasattr(transformers,'Qwen3Config'):pytest.skip('Qwen3 training environment required')
    torch.manual_seed(23)
    config=transformers.Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                                    num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                                    max_position_embeddings=64,attention_dropout=.6)
    model=peft.get_peft_model(transformers.Qwen3ForCausalLM(config),
                             peft.LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,
                                             target_modules='all-linear',lora_dropout=0.0))
    # Nonzero B matrices exercise gradients through both halves of each adapter.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.normal_(std=.03)
    assert_equivalent(model,[1,4,5,6,7],[8,9,10,11,12])
