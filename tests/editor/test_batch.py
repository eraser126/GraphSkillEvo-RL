"""Batch contracts with fake workers/editors only. No model/API/optimizer training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from graphskillevo.editor.batch import load_batch_manifest,batches,run_group,train_batches
from graphskillevo.editor.evidence import Trace
from graphskillevo.editor.config import EditorConfig
from graphskillevo.editor.accounting import Ledger
from graphskillevo.editor.episode import EditorTurn

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def graph():return json.loads((ROOT/'examples/editor/graph.json').read_text())


def tasks(count=4):return [dict(id=f't{i}',family='same',payload={'instruction':f'case {i}'}) for i in range(count)]


class FakeEditor:
    def __init__(self):self.prompts=[];self.model=SimpleNamespace(parameters=lambda:[])
    def generate(self,messages,deterministic=False):
        data=json.loads(messages[1]['content']);self.prompts.append(data)
        ops=[] if data['stage']=='checks' else [dict(op='set_guidance',payload='Reusable changed guidance')]
        return EditorTurn(json.dumps(dict(base_version=data['graph']['version'],operations=ops,
                                          evidence_ids=[],rationale='Synthetic contract edit')))
    def save(self,p):Path(p).mkdir(parents=True,exist_ok=True);(Path(p)/'FAKE').write_text('test fixture')


class FakeBackend:
    def __init__(self):self.calls=[]
    def run(self,graph,tasks,phase):
        self.calls.append((phase,[x['id'] for x in tasks]))
        rows=[]
        for t in tasks:
            score=int(graph['global_guidance']=='Reusable changed guidance')
            row=dict(id=t['id'],score=score)
            if phase=='support':row['trace']=Trace.create(dict(task_id=t['id'],events=[],snapshots=[],success=bool(score))).public()
            rows.append(row)
        return rows


def test_single_baseline_four_candidates_then_update(graph,tmp_path):
    from graphskillevo.editor.verification import CheckEngine
    cfg=EditorConfig(cost_weight=0);backend=FakeBackend();editor=FakeEditor();seen=[]
    def update(*a):seen.append(len(backend.calls));return {'updated':False,'reason':'spy_only'}
    candidates,metrics=run_group(editor,backend,CheckEngine(),Ledger(tmp_path/'ledger'),cfg,graph,tasks(),tmp_path/'group',update)
    assert [x[0] for x in backend.calls]==['support']+['reward']*4
    assert all(x[1]==['t0','t1','t2','t3'] for x in backend.calls)
    assert seen==[5] and len(editor.prompts)==8
    assert metrics['expected_worker_executions']==20
    assert all(c.reward==1 for c in candidates)
    assert all(len(p['support'])==4 for p in editor.prompts)


def test_batch_shuffle_complete_and_family_local():
    rows=tasks(9)
    for i,t in enumerate(rows):t['family']='odd' if i%2 else 'even'
    panels=batches(rows,4,42)
    assert sorted(t['id'] for p in panels for t in p)==sorted(t['id'] for t in rows)
    assert all(len({t['family'] for t in p})==1 for p in panels)
    assert panels==batches(rows,4,42)


def test_manifest_allows_same_family_across_task_splits_but_not_same_task(graph,tmp_path):
    (tmp_path/'g.json').write_text(json.dumps(graph))
    rows=tasks(3)
    data=dict(schema_version=2,benchmark='synthetic',training_allowed=False,provenance='test',
              graph='g.json',splits=dict(train=[rows[0]],val=[rows[1]],test=[rows[2]]))
    p=tmp_path/'m.json';p.write_text(json.dumps(data));assert load_batch_manifest(p)['graph']==graph
    data['splits']['test']=[rows[0]];p.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_batch_manifest(p)


def test_dataset_answers_rejected(graph,tmp_path):
    (tmp_path/'g.json').write_text(json.dumps(graph));rows=tasks(3);rows[0]['payload']['gold_answer']='private'
    data=dict(schema_version=2,benchmark='synthetic',training_allowed=False,provenance='test',graph='g.json',
              splits=dict(train=[rows[0]],val=[rows[1]],test=[rows[2]]))
    p=tmp_path/'m.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_batch_manifest(p)


def test_task_context_and_source_are_in_evidence(graph,tmp_path):
    from graphskillevo.editor.spreadsheet import SpreadsheetEvidenceSession
    from graphskillevo.editor.episode import prompt
    s=SpreadsheetEvidenceSession(graph,'synthetic',tmp_path)
    s.observe('TASK_CONTEXT','Preserve formulas while sorting')
    (tmp_path/'solution.py').write_text('print("script evidence")')
    nid=graph['nodes'][0]['node_id'];s.checkpoint(nid,'start');s.checkpoint(nid,'end')
    trace=s.finish(False)
    data=json.loads(prompt(graph,[trace],[],'checks')[1]['content'])
    assert 'Preserve formulas' in str(data)
    assert 'script evidence' in str(data)
    assert trace.public()['snapshots'][0]['values']['script']['source_truncated'] is False


def test_prompt_compression_preserves_goal_and_recent_failure(graph,tmp_path):
    from graphskillevo.editor.spreadsheet import SpreadsheetEvidenceSession
    from graphskillevo.editor.episode import prompt
    s=SpreadsheetEvidenceSession(graph,'synthetic',tmp_path)
    s.observe('TASK_CONTEXT','Actual task goal')
    for i in range(20):s.observe('tool '+str(i),'Long observation '+('x'*1500))
    s.observe('last tool','ERROR_FINAL')
    trace=s.finish(False);before=trace.id
    view=prompt(graph,[trace],[],'checks')[1]['content']
    assert 'Actual task goal' in view and 'ERROR_FINAL' in view
    assert 'Partial display' in view and trace.id==before


def test_training_loop_never_uses_test_and_updates_only_after_group(graph,tmp_path,monkeypatch):
    import torch
    import graphskillevo.editor.grpo as grpo
    from graphskillevo.editor.verification import CheckEngine
    backend=FakeBackend();editor=FakeEditor();updates=[]
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:SimpleNamespace(state_dict=lambda:{}))
    monkeypatch.setattr(grpo,'update_group',lambda *a,**k:updates.append(len(backend.calls)) or {'updated':False})
    data=dict(graph=graph,training_allowed=True,splits=dict(train=tasks(4),val=[dict(id='v',family='same',payload={'instruction':'val'})],
                                                          test=[dict(id='SECRET_TEST',family='same',payload={'instruction':'test'})]))
    cfg=EditorConfig(updates=1,batch_size=4,cost_weight=0,dev_frequency=1)
    train_batches(editor,backend,CheckEngine(),Ledger(tmp_path/'ledger'),cfg,data,tmp_path)
    assert updates==[5]
    assert not any('SECRET_TEST' in ids for _,ids in backend.calls)
    assert json.loads((tmp_path/'summary.json').read_text())['test_used'] is False
    assert (tmp_path/'best_dev_model/FAKE').is_file()
