"""Library integration with an untrained miniature model, no download or weight update.

These are unit checks of token likelihoods and checkpoint I/O, not a learning or
benchmark experiment. Pytest removes the temporary model files.
"""
from pathlib import Path
import pytest
import torch

transformers=pytest.importorskip('transformers')
from graphskillevo.editor.policy import LocalLLMEditor


def local_fixture(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,GPT2Config,GPT2LMHeadModel
    vocab={'[PAD]':0,'[UNK]':1,'[BOS]':2,'[EOS]':3,'system':4,'user':5,'hello':6,'assistant':7}
    tokenizer=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',
                                 bos_token='[BOS]',eos_token='[EOS]')
    torch.manual_seed(13)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=8,n_positions=64,n_ctx=64,n_embd=8,n_layer=1,n_head=1,
                                    bos_token_id=2,eos_token_id=3,pad_token_id=0,
                                    resid_pdrop=.8,embd_pdrop=.8,attn_pdrop=.8))
    model.save_pretrained(path);fast.save_pretrained(path)


def test_offline_local_load_generate_and_exact_likelihoods(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    local_fixture(tmp_path/'base')
    editor=LocalLLMEditor.load(tmp_path/'base',max_new_tokens=2,max_context=64)
    editor.model.generation_config.temperature = 0.7
    editor.model.generation_config.top_p = 0.8
    original_processors = editor.model._get_logits_processor

    def check_sampling(generation_config, *args, **kwargs):
        assert generation_config.temperature == 1.0
        assert generation_config.top_p == 1.0
        assert generation_config.top_k == 0
        return original_processors(generation_config, *args, **kwargs)

    monkeypatch.setattr(editor.model, '_get_logits_processor', check_sampling)
    before={k:v.clone() for k,v in editor.model.state_dict().items()}
    turn=editor.generate([dict(role='system',content='hello'),dict(role='user',content='hello')])
    assert len(turn.completion_ids)==len(turn.old_logps)>0
    recomputed=editor.logps(turn.prompt_ids,turn.completion_ids).detach()
    assert torch.allclose(recomputed,torch.tensor(turn.old_logps),atol=1e-6)
    assert not editor.model.training
    assert all(torch.equal(v,before[k]) for k,v in editor.model.state_dict().items())
    with pytest.raises(ValueError): LocalLLMEditor.load(tmp_path/'missing_model')


def test_optional_lora_checkpoint_roundtrip_without_training(tmp_path,monkeypatch):
    pytest.importorskip('peft')
    monkeypatch.setenv('HF_HUB_OFFLINE','1');monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    local_fixture(tmp_path/'base')
    editor=LocalLLMEditor.load(tmp_path/'base',lora_rank=2,max_context=64)
    assert any('lora_' in n and p.requires_grad for n,p in editor.model.named_parameters())
    assert not any('base_layer' in n and p.requires_grad for n,p in editor.model.named_parameters())
    before=editor.logps([2,4],[6,3]).detach()
    editor.save(tmp_path/'adapter')
    reloaded=LocalLLMEditor.load(tmp_path/'adapter',lora_rank=2,max_context=64)
    assert torch.allclose(reloaded.logps([2,4],[6,3]).detach(),before,atol=1e-6)
