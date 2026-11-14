"""Offline schema and feedback contracts. No model download/API/optimizer update."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest

from graphskillevo.editor.schema import (ContractError,validate_graph,render_worker,import_markdown,
                                          apply_patch,parse_patch,fingerprint)
from graphskillevo.editor.evidence import Trace,snapshot,align,SnapshotStore
from graphskillevo.editor.verification import CheckEngine,JudgeProtocolError
from graphskillevo.editor.diagnosis import diagnose
from graphskillevo.editor.episode import propose
from graphskillevo.editor.policy import ScriptedEditor
from graphskillevo.editor.data import load_manifest

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def graph(): return json.loads((ROOT/'examples/editor/graph.json').read_text())


def event(i,kind,node,observation='',reads=(),writes=()):
    return dict(index=i,kind=kind,node_ids=[node] if node else [],action='',observation=observation,
                reads=list(reads),writes=list(writes))


def trace_record(success=False):
    events=[event(0,'node_start','inspect'),event(1,'node_end','inspect','values extracted',writes=['data']),
            event(2,'node_start','compute'),event(3,'step','compute','read data',reads=['data']),
            event(4,'node_end','compute','result ready')]
    return dict(task_id='support_fixture',events=events,
                snapshots=[snapshot('inspect',1,{'output':{'value':10,'unit':''}}),
                           snapshot('compute',4,{'output':''})],success=success)


def patch(g,ops): return dict(base_version=g['version'],operations=ops,evidence_ids=[],rationale='Evidence-led edit')


def unit_check(cid='unit_present',path='/output/unit'):
    return dict(check_id=cid,applies_when='always',target=path,requirement='Preserve unit',
                evidence_needed=['output'],method='rule',rule=dict(kind='nonempty',path=path))


def test_roundtrip_original_markdown_and_ids(graph):
    worker=render_worker(graph)
    loaded=import_markdown(worker,'imported')
    assert len(loaded['nodes'])==2
    assert loaded['workflows'][0]['nodes']==[n['node_id'] for n in loaded['nodes']]


def test_checks_do_not_leak_to_worker(graph):
    before=render_worker(graph)
    graph['nodes'][0]['checks'].append(unit_check())
    assert render_worker(graph)==before
    assert 'unit_present' not in before


def test_cycles_allowed_and_references_required(graph):
    graph['workflows'][0]['nodes'].append('inspect');validate_graph(graph)
    graph['workflows'][0]['nodes'].append('missing')
    with pytest.raises(ContractError): validate_graph(graph)


def test_transactional_delete_and_rewire(graph):
    original=deepcopy(graph)
    with pytest.raises(ContractError):
        apply_patch(graph,patch(graph,[dict(op='delete_node',node_id='inspect')]),'graph')
    assert graph==original
    w=deepcopy(graph['workflows'][0]);w['nodes']=['compute']
    new=apply_patch(graph,patch(graph,[dict(op='delete_node',node_id='inspect'),dict(op='set_workflow',payload=w)]),'graph')
    assert len(new['nodes'])==1 and graph==original and new['version']==1


def test_stage_and_stale_patch_rejected(graph):
    with pytest.raises(ContractError):
        apply_patch(graph,patch(graph,[dict(op='update_node',node_id='inspect',payload={'instruction':'new'})]),'checks')
    p=patch(graph,[]);p['base_version']=5
    with pytest.raises(ContractError): apply_patch(graph,p,'checks')
    assert apply_patch(graph,patch(graph,[]),'checks')==graph


def test_no_generated_code_or_json_recovery(graph):
    c=unit_check();c['method']='python';graph['nodes'][0]['checks']=[c]
    with pytest.raises(ContractError): validate_graph(graph)
    with pytest.raises(ContractError): parse_patch('```json\n{}\n```')


def test_snapshot_integrity_and_immutability(tmp_path):
    r=trace_record();t=Trace.create(r);old=t.id
    r['snapshots'][0]['values']['output']['value']=99
    assert t.id==old and t.public()['snapshots'][0]['values']['output']['value']==10
    with pytest.raises(ContractError): Trace.create(r)
    s=SnapshotStore(tmp_path);path=s.put(t);assert s.put(t)==path


def test_final_state_cannot_masquerade_as_node_snapshot():
    r=trace_record();r['snapshots'][0]['at_event']=3
    with pytest.raises(ContractError): Trace.create(r)


def test_no_marker_is_not_a_skip(graph):
    r=trace_record();r['events']=[];r['snapshots']=[]
    a=align(graph,Trace.create(r));assert set(a['statuses'].values())=={'alignment_unknown'}
    r['events']=[event(0,'skip','inspect')]
    a=align(graph,Trace.create(r));assert a['statuses']['inspect']=='skipped'


def test_off_graph_and_revisits(graph):
    r=trace_record();r['events'] += [event(5,'step','external_step','exploration'),event(6,'node_start','inspect'),
                                   event(7,'node_end','inspect','revisited')]
    a=align(graph,Trace.create(r));assert a['off_graph']==[5]
    d=diagnose(graph,CheckEngine().run(graph,[Trace.create(r)]))
    assert d['deviations'][0]['revisits']==['inspect']


def test_unlabeled_action_is_unknown_not_proven_off_graph(graph):
    r=trace_record();r['events'].append(event(5,'step',None,'unlabeled'))
    a=align(graph,Trace.create(r));assert a['unmapped_events']==[5] and a['off_graph']==[]


def test_missing_snapshot_unknown_not_pass(graph):
    r=trace_record();r['snapshots']=[]
    report=CheckEngine().run(graph,[Trace.create(r)])[0]
    assert report['checks'][0]['status']=='unknown'


def test_false_success_and_false_failure_are_signals(graph):
    reports=CheckEngine().run(graph,[Trace.create(trace_record())])
    d=diagnose(graph,reports)
    assert d['contradictions'][0]['kind']=='failure_without_detected_check_failure'
    graph['nodes'][0]['checks'].append(unit_check())
    report=CheckEngine().run(graph,[Trace.create(trace_record(True))])
    assert diagnose(graph,report)['contradictions'][0]['kind']=='success_despite_check_failure'


def test_upstream_candidate_requires_artifact_evidence(graph):
    graph['nodes'][1]['checks']=[unit_check('result_nonempty','/output')]
    r=trace_record();d=diagnose(graph,CheckEngine().run(graph,[Trace.create(r)]))
    assert any(c['node_id']=='inspect' and c['reason']=='upstream_artifact_candidate' for c in d['candidates'])
    r['events'][3]['reads']=[]
    d=diagnose(graph,CheckEngine().run(graph,[Trace.create(r)]))
    assert not any(c['reason']=='upstream_artifact_candidate' for c in d['candidates'])


def test_invalid_fixed_judge_is_not_an_edit_failure(graph):
    c=unit_check();c['method']='llm';c.pop('rule');graph['nodes'][0]['checks']=[c]
    engine=CheckEngine(lambda request:{'status':'pass','reason':'unsupported','evidence_keys':[]})
    with pytest.raises(JudgeProtocolError): engine.run(graph,[Trace.create(trace_record())])


def test_two_turn_feedback_is_rechecked_and_used(graph):
    t=Trace.create(trace_record());seen=[]
    responses=json.loads((ROOT/'examples/editor/responses.json').read_text())
    class Editor:
        def generate(self,messages,deterministic=False):
            from graphskillevo.editor.episode import EditorTurn
            context=json.loads(messages[1]['content']);seen.append(context)
            return EditorTurn(json.dumps(responses[len(seen)-1]),prompt=messages)
    c=propose(graph,[t],CheckEngine(),Editor())
    assert c.valid and len(c.turns)==2 and c.graph['version']==2
    assert any(x['check_id']=='unit_present' and x['status']=='fail' for x in seen[1]['reports'][0]['checks'])
    assert t.id==c.audit['trace_ids'][0]
    assert 'Normalize incompatible units' in c.graph['nodes'][0]['instruction']


def test_invalid_second_turn_rolls_back_first_turn(graph):
    responses=json.loads((ROOT/'examples/editor/responses.json').read_text());responses[1]['base_version']=99
    c=propose(graph,[Trace.create(trace_record())],CheckEngine(),ScriptedEditor(map(json.dumps,responses)))
    assert not c.valid and c.graph==graph and len(c.turns)==2


def test_checker_outage_after_condition_edit_propagates(graph):
    c=unit_check();c['method']='llm';c.pop('rule')
    first=patch(graph,[dict(op='add_check',node_id='inspect',payload=c)])
    def outage(request): raise ConnectionError('offline fixed judge unavailable')
    with pytest.raises(ConnectionError):
        propose(graph,[Trace.create(trace_record())],CheckEngine(outage),ScriptedEditor([json.dumps(first)]))


def test_heldout_panels_never_enter_editor_prompt(graph):
    # propose accepts no reward/query parameter; public traces reject top-level hidden fields.
    r=trace_record();r['gold_answer']='DO_NOT_EXPOSE'
    with pytest.raises(ContractError): Trace.create(r)
    from graphskillevo.editor.episode import prompt
    messages=prompt(graph,[Trace.create(trace_record())],[],'graph')
    assert 'DO_NOT_EXPOSE' not in json.dumps(messages)


def write_manifest(tmp_path, graph, split='train', benchmark='permitted_fixture'):
    (tmp_path/'graph.json').write_text(json.dumps(graph))
    u=json.loads((ROOT/'examples/editor/manifest.json').read_text())['items'][0]
    u.update(split=split,benchmark=benchmark,training_allowed=True,graph='graph.json')
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(schema_version=1,items=[u])))
    return p,u


def test_split_overlap_and_skillsbench_training_rejected(tmp_path,graph):
    p,u=write_manifest(tmp_path,graph,benchmark='SkillsBench')
    with pytest.raises(ContractError): load_manifest(p)
    p,u=write_manifest(tmp_path,graph)
    u['query'][0]=deepcopy(u['support'][0]);p.write_text(json.dumps(dict(schema_version=1,items=[u])))
    with pytest.raises(ContractError): load_manifest(p)


def test_skill_family_leakage_rejected(tmp_path,graph):
    p,u=write_manifest(tmp_path,graph);v=deepcopy(u);v.update(unit_id='heldout',split='test')
    p.write_text(json.dumps(dict(schema_version=1,items=[u,v])))
    with pytest.raises(ContractError): load_manifest(p)


def test_config_and_cli_default_are_offline(tmp_path):
    from graphskillevo.editor.config import EditorConfig
    assert EditorConfig.load(ROOT/'configs/editor/minimal.yaml').variant=='full'
    cmd=[sys.executable,'-m','graphskillevo.run_editor','--config',str(ROOT/'configs/editor/minimal.yaml'),
         '--manifest',str(ROOT/'examples/editor/manifest.json')]
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)['model_loaded'] is False
    r=subprocess.run(cmd+['--mode','adapt','--unit-id','contract_fixture','--out-root',str(tmp_path/'blocked')],
                     capture_output=True,text=True)
    assert r.returncode==2 and not (tmp_path/'blocked').exists()


def test_training_requires_second_opt_in(tmp_path,graph):
    p,_=write_manifest(tmp_path,graph)
    r=subprocess.run([sys.executable,'-m','graphskillevo.run_editor','--config',str(ROOT/'configs/editor/minimal.yaml'),
                      '--manifest',str(p),'--mode','train','--allow-api','--out-root',str(tmp_path/'blocked')],
                     capture_output=True,text=True)
    assert r.returncode==2 and 'allow-training' in r.stderr and not (tmp_path/'blocked').exists()
