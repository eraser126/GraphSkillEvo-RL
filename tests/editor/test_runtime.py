"""No task environment, API, LLM loading or optimizer parameter update is performed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from graphskillevo.editor.accounting import Ledger,BudgetExceeded
from graphskillevo.editor.backends import validate_rows,CommandBackend
from graphskillevo.editor.episode import Candidate,EditorTurn
from graphskillevo.editor.grpo import advantages,clipped_loss,update_group
from graphskillevo.editor.policy import LocalLLMEditor,ScriptedEditor
from graphskillevo.editor.recorder import EvidenceRecorder
from graphskillevo.editor.schema import ContractError,fingerprint
from graphskillevo.editor.runner import adapt,evaluate_query
from graphskillevo.editor.config import EditorConfig
from graphskillevo.editor.verification import CheckEngine

ROOT=Path(__file__).resolve().parents[2]


def graph(): return json.loads((ROOT/'examples/editor/graph.json').read_text())


def test_grpo_group_and_clipping_gradients_only():
    assert torch.equal(advantages([1,1]),torch.zeros(2))
    a=advantages([0,1,2,3]);assert abs(float(a.mean()))<1e-6
    for reward in (1.,-1.):
        logits=torch.tensor([-1.,-2.],requires_grad=True)
        loss=clipped_loss(logits,torch.tensor([-1.1,-1.9]),reward)
        loss.backward();assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError): advantages([float('nan'),1])


def test_completion_mask_and_full_group_update_without_parameter_changes():
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(8,4);self.head=torch.nn.Linear(4,8)
        def forward(self,input_ids,**kw): return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
    torch.manual_seed(7);model=LM();editor=LocalLLMEditor(model,None)
    cs=[]
    for i,r in enumerate([0.,1.]):
        turns=[]
        for prompt,completion in [([1,2],[3,4]),([1,5,6],[7])]:
            old=editor.logps(prompt,completion).detach().tolist()
            turns.append(EditorTurn('{}',prompt,completion,old))
        cs.append(Candidate({},turns,True,{},r))
    class Spy:
        def __init__(self): self.steps=0
        def zero_grad(self,**kw): model.zero_grad(set_to_none=True)
        def step(self): self.steps+=1  # Deliberately never changes parameters.
    before={k:v.clone() for k,v in model.state_dict().items()}
    spy=Spy();result=update_group(editor,cs,spy)
    assert result['updated'] and spy.steps==1
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not model.training


def test_api_action_cannot_be_trained():
    class Editor: model=SimpleNamespace(eval=lambda:None)
    cs=[Candidate({},[EditorTurn('{}')],True,{},r) for r in (0.,1.)]
    with pytest.raises(ValueError): update_group(Editor(),cs,None)


def test_ledger_retains_failed_usage_and_restart(tmp_path):
    ledger=Ledger(tmp_path,max_calls=1)
    def broken(): raise ConnectionError('no request actually sent')
    with pytest.raises(ConnectionError): ledger.call('offline_failure',broken)
    again=Ledger(tmp_path,max_calls=1);assert again.calls==1
    with pytest.raises(BudgetExceeded): again.check()


def test_ledger_counts_internal_provider_calls(tmp_path,monkeypatch):
    import graphskillevo.model as router
    totals=dict(total_tokens=0,calls=0)
    monkeypatch.setattr(router,'get_token_summary',lambda:{'_total':dict(totals)})
    ledger=Ledger(tmp_path)
    def panel(): totals.update(total_tokens=90,calls=3)
    ledger.call('panel',panel)
    assert ledger.tokens==90 and ledger.calls==3


def test_worker_incomplete_or_invalid_score_is_not_zero():
    tasks=[dict(id='a',payload={})]
    with pytest.raises(RuntimeError): validate_rows(tasks,[],'reward')
    with pytest.raises(RuntimeError): validate_rows(tasks,[dict(id='a',score=float('nan'))],'reward')
    with pytest.raises(RuntimeError): validate_rows(tasks,[dict(id='a',score=0,error='timeout')],'reward')


def test_external_runner_is_argv_and_new_job_paths(tmp_path):
    ledger=Ledger(tmp_path)
    with pytest.raises(ContractError): CommandBackend('echo foo',tmp_path,ledger)
    with pytest.raises(ContractError): CommandBackend(['runner'],tmp_path,ledger)


def test_recorder_captures_at_end_not_final_state(tmp_path):
    source=tmp_path/'data.json';source.write_text('{"unit":"m"}')
    recorder=EvidenceRecorder('fixture',tmp_path);recorder.start('inspect')
    recorder.end('inspect',visible_files={'output':'data.json'})
    source.write_text('{"unit":"kg"}')
    trace=recorder.finish(True)
    assert trace.public()['snapshots'][0]['values']['output']['unit']=='m'
    with pytest.raises(ContractError): recorder.end('inspect',visible_files={'secret':'../outside'})
    protected=tmp_path/'oracle';protected.mkdir();(protected/'answer').write_text('hidden')
    with pytest.raises(ContractError): recorder.end('inspect',visible_files={'bad':'oracle/answer'})


def test_adaptation_freezes_parameters_and_does_not_access_reward_or_query(tmp_path):
    g=graph();recorder=EvidenceRecorder('support_fixture',tmp_path);recorder.start('inspect')
    recorder.end('inspect',values={'output':{'value':10,'unit':''}});trace=recorder.finish(False)
    calls=[]
    class Backend:
        def run(self,graph,tasks,phase):
            calls.append(phase);assert phase=='support'
            return [dict(id='support_fixture',score=0,trace=trace.public())]
    responses=json.loads((ROOT/'examples/editor/responses.json').read_text())
    editor=ScriptedEditor(map(json.dumps,responses));editor.model=torch.nn.Linear(2,2)
    before={k:v.clone() for k,v in editor.model.state_dict().items()}
    unit=dict(unit_id='fixture',graph=g,support=[dict(id='support_fixture',payload={})],
              reward='FORBIDDEN_REWARD',query='FORBIDDEN_QUERY')
    artifact=adapt(editor,Backend(),CheckEngine(),Ledger(tmp_path),EditorConfig(),unit,tmp_path)
    assert calls==['support'] and artifact['query_used'] is False
    assert not any(p.requires_grad for p in editor.model.parameters())
    assert all(torch.equal(v,before[k]) for k,v in editor.model.state_dict().items())


def test_final_evaluation_checks_frozen_artifact(tmp_path):
    g=graph();unit=dict(unit_id='u',graph=g,support=[dict(id='s',payload={})],query=[dict(id='q',payload={})])
    artifact=dict(unit_id='u',graph=g,graph_hash=fingerprint(g),source_graph_hash=fingerprint(g),
                  unit_fingerprint=fingerprint(unit),
                  support_ids=['s'],query_used=False)
    class Backend:
        def run(self,g,tasks,phase):
            assert phase=='query';return [dict(id='q',score=1)]
    result=evaluate_query(Backend(),unit,artifact,tmp_path);assert result['score']==1
    changed=deepcopy(artifact);changed['graph']['version']=99
    with pytest.raises(ValueError): evaluate_query(Backend(),unit,changed,tmp_path)
    changed_unit=deepcopy(unit);changed_unit['query'][0]['payload']['changed']=True
    with pytest.raises(ValueError): evaluate_query(Backend(),changed_unit,artifact,tmp_path)
