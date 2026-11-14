"""No real keys or network calls."""
import json
from types import SimpleNamespace
import pytest
from graphskillevo.model.gateway import load_gateway,normalize_base_url


def test_gateway_private_config(monkeypatch,tmp_path):
    p=tmp_path/'gateway.json';p.write_text(json.dumps(dict(base_url='https://example.invalid',api_key='unit-test-dummy',model='gpt-6.1-sol')))
    monkeypatch.setenv('GRAPHSKILLEVO_GATEWAY_FILE',str(p))
    for k in ['GATEWAY_BASE_URL','GATEWAY_API_KEY','GATEWAY_MODEL_NAME','MODEL_NAME']:monkeypatch.delenv(k,raising=False)
    assert load_gateway()==dict(base_url='https://example.invalid/v1',api_key='unit-test-dummy',model='gpt-6.1-sol')


def test_gateway_secret_not_put_in_logged_config(monkeypatch):
    import graphskillevo.model_setup as setup
    import graphskillevo.model.gateway as module
    captured={}
    monkeypatch.setattr(module,'load_gateway',lambda:dict(base_url='https://example.invalid/v1',api_key='unit-test-dummy',model='gpt-6.1-sol'))
    monkeypatch.setattr(setup,'configure_azure_openai',lambda **kwargs:captured.update(kwargs))
    for name in ['set_target_backend','set_optimizer_backend','set_target_deployment','set_optimizer_deployment','configure_codex_exec','set_reasoning_effort']:
        monkeypatch.setattr(setup,name,lambda *a,**k:None)
    cfg={};setup.configure_models(cfg)
    assert 'unit-test-dummy' not in json.dumps(cfg)
    assert cfg['target_model']=='gpt-6.1-sol' and captured['target_api_key']=='unit-test-dummy'


def test_gpt61_routes_to_responses_and_replays_tool_context():
    from graphskillevo.model.azure_openai import _needs_responses_api,_messages_to_responses_input
    assert _needs_responses_api('gpt-6.1-sol')
    output=[dict(type='reasoning',id='r1',summary=[]),dict(type='function_call',call_id='c1',name='bash',arguments='{}')]
    data,instructions=_messages_to_responses_input([dict(role='system',content='fixed'),
        dict(role='assistant',content='',_responses_output=output),dict(role='tool',tool_call_id='c1',content='observed')])
    assert data[:2]==output and data[2]['type']=='function_call_output'
    assert instructions=='fixed'


def test_base_url_validation():
    assert normalize_base_url('https://example.invalid/v1/')=='https://example.invalid/v1'
    with pytest.raises(ValueError):normalize_base_url('https://key@example.invalid')


def test_responses_preserves_document_images():
    from graphskillevo.model.azure_openai import _messages_to_responses_input
    items,_=_messages_to_responses_input([dict(role='user',content=[dict(type='text',text='Read document'),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,fixture',detail='high'))])])
    assert items[0]['content'][0]['type']=='input_text'
    assert items[0]['content'][1]==dict(type='input_image',image_url='data:image/png;base64,fixture',detail='high')
