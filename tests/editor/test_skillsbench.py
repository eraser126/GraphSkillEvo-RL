"""SkillsBench overlay/normalization contracts, never invoking BenchFlow or containers."""
import json
from pathlib import Path
import pytest
from graphskillevo.editor.skillsbench import stage_skills,plan_task,normalize_result
from graphskillevo.editor.schema import ContractError


def task(tmp_path):
    checkout=tmp_path/'checkout';t=checkout/'tasks'/'example';s=t/'environment'/'skills'
    (s/'target').mkdir(parents=True);(s/'other').mkdir()
    (t/'task.md').write_text('public prompt')
    (s/'target'/'SKILL.md').write_text('---\nname: target\ndescription: reusable\n---\nold')
    (s/'target'/'reference.txt').write_text('shared reference')
    (s/'other'/'SKILL.md').write_text('untouched')
    (t/'verifier').mkdir();(t/'verifier'/'test.sh').write_text('hidden evaluator')
    return checkout,t


def test_only_target_skill_body_changes(tmp_path):
    checkout,t=task(tmp_path)
    out=stage_skills(t,'target','NEW GRAPH',tmp_path/'overlay')
    assert (out/'target'/'SKILL.md').read_text().endswith('NEW GRAPH')
    assert (out/'target'/'SKILL.md').read_text().startswith('---\nname: target')
    assert (out/'other'/'SKILL.md').read_text()=='untouched'
    assert (out/'target'/'reference.txt').read_text()=='shared reference'
    assert not (out/'verifier').exists()
    assert (t/'environment/skills/target/SKILL.md').read_text().endswith('old')


def test_task_plan_is_pinned_location_and_no_execution(tmp_path):
    checkout,t=task(tmp_path)
    plan=plan_task(dict(id='id',payload=dict(task_dir='tasks/example',target_skill='target')),
                   'NEW',checkout,tmp_path/'plan','bench','fixed-agent','fixed-model')
    assert plan['benchmark_executed'] is False and '--skills-dir' in plan['argv']
    with pytest.raises(ContractError): stage_skills(t,'../../verifier','bad',tmp_path/'bad')
    with pytest.raises(ContractError): stage_skills(t,str(t/'environment/skills/target'),'bad',tmp_path/'abs')
    with pytest.raises(ContractError): plan_task(dict(id='bad',payload=dict(task_dir='../outside',target_skill='x')),
                                                'bad',checkout,tmp_path/'invalid','bench','a','m')


def test_score_and_usage_normalization_fail_closed(tmp_path):
    p=tmp_path/'result.json';p.write_text(json.dumps(dict(rewards=dict(reward=1),token_usage=dict(total_tokens=20,calls=2))))
    r=normalize_result(p,'id','support','token_usage.total_tokens','token_usage.calls')
    assert r['score']==1 and r['trace']['snapshots']==[] and r['tokens']==20
    p.write_text(json.dumps(dict(rewards=dict(reward=1))))
    with pytest.raises(RuntimeError): normalize_result(p,'id','reward','token_usage.total_tokens','token_usage.calls')
    p.write_text(json.dumps(dict(error='timeout',rewards=dict(reward=0))))
    with pytest.raises(RuntimeError): normalize_result(p,'id','reward','token_usage.total_tokens','token_usage.calls')
