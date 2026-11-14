"""SkillsBench skill-overlay planner and fail-closed BenchFlow CLI wrapper.

No BenchFlow import is needed in the editor's Python 3.10 process. BenchFlow can
live in a separate Python 3.12 environment. Real execution still needs integration
verification against the chosen pinned release; default planning is offline.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from .accounting import write_json
from .evidence import Trace,events_from_conversation
from .schema import ContractError,fields


def stage_skills(task_dir,target_skill,worker_skill,out_dir):
    task_dir=Path(task_dir).resolve();source=task_dir/'environment'/'skills'
    if not isinstance(target_skill,str) or Path(target_skill).is_absolute() or '..' in Path(target_skill).parts:
        raise ContractError('Target skill must be a non-traversing relative package path')
    target=source/target_skill/'SKILL.md'
    if not source.is_dir() or not (task_dir/'task.md').is_file() or not target.resolve().is_relative_to(source.resolve()):
        raise ContractError('Expected a native task and an in-tree target skill')
    if not target.is_file(): raise ContractError('Target skill does not exist')
    if any(p.is_symlink() for p in source.rglob('*')) or source.is_symlink():
        raise ContractError('Skill symlinks require an explicit input audit')
    out_dir=Path(out_dir).resolve()
    if out_dir.is_relative_to(task_dir): raise ContractError('Overlay must be outside the original task package')
    if out_dir.exists(): raise ContractError('Skill overlay destination must be fresh')
    shutil.copytree(source,out_dir)
    original=target.read_text(encoding='utf-8')
    frontmatter=re.match(r'\A---\r?\n.*?\r?\n---\r?\n',original,re.S)
    prefix=frontmatter[0] if frontmatter else ''
    # Preserve package discovery metadata and every other skill/reference/script.
    (out_dir/target_skill/'SKILL.md').write_text(prefix+worker_skill,encoding='utf-8')
    return out_dir


def plan_task(task,worker_skill,checkout,output,bench_bin,agent,model,sandbox='docker'):
    fields(task,{'id','payload'},{'id','payload'})
    fields(task['payload'],{'task_dir','target_skill'},{'task_dir','target_skill'})
    checkout=Path(checkout).resolve();taskdir=(checkout/task['payload']['task_dir']).resolve()
    if not taskdir.is_relative_to(checkout/'tasks') or not taskdir.is_dir():
        raise ContractError('Task must be under the pinned checkout tasks directory')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    overlay=stage_skills(taskdir,task['payload']['target_skill'],worker_skill,output/'skills')
    jobs=output/'jobs'
    argv=[bench_bin,'eval','run','--tasks-dir',str(taskdir),'--agent',agent,'--model',model,
          '--sandbox',sandbox,'--skill-mode','with-skill','--skills-dir',str(overlay),
          '--jobs-dir',str(jobs),'--concurrency','1']
    result=dict(task_id=task['id'],argv=argv,jobs_dir=str(jobs),
                original_task=str(taskdir),skill_overlay=str(overlay),
                grader_modified=False,training_started=False,benchmark_executed=False)
    write_json(output/'plan.json',result);return result


def nested(row,path):
    value=row
    for key in path.split('.'):
        if not isinstance(value,dict) or key not in value: return None
        value=value[key]
    return value


def normalize_result(path,task_id,phase,tokens_field,calls_field):
    path=Path(path);r=json.loads(path.read_text(encoding='utf-8'))
    if r.get('error') or r.get('verifier_error'): raise RuntimeError('BenchFlow rollout/verifier error')
    rewards=r.get('rewards')
    value=rewards.get('reward') if isinstance(rewards,dict) else None
    if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise RuntimeError('Unknown BenchFlow reward schema; refusing to infer a score')
    tokens,calls=nested(r,tokens_field),nested(r,calls_field)
    if type(tokens) is not int or tokens<0 or type(calls) is not int or calls<0:
        raise RuntimeError('Missing trusted internal usage; select fields for the installed BenchFlow version')
    result=dict(id=task_id,score=float(value),tokens=tokens,calls=calls)
    if phase=='support':
        capture=path.parent/'editor_trace.json'
        if capture.is_file(): trace=Trace.create(json.loads(capture.read_text(encoding='utf-8')))
        else:
            # Capture only the agent conversation. Never parse grader/oracle artifacts as evidence.
            conversation=path.parent/'trajectory'/'conversation.json'
            rows=json.loads(conversation.read_text(encoding='utf-8')) if conversation.is_file() else []
            trace=Trace.create(dict(task_id=task_id,events=events_from_conversation(rows),snapshots=[],success=value==1))
        result['trace']=trace.public()
    return result


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--response',required=True)
    p.add_argument('--checkout',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--bench-bin',default='bench');p.add_argument('--bench-version')
    p.add_argument('--agent',required=True);p.add_argument('--worker-model',required=True)
    p.add_argument('--sandbox',choices=('docker','modal','daytona'),default='docker')
    p.add_argument('--tokens-field',default='token_usage.total_tokens')
    p.add_argument('--calls-field',default='token_usage.calls')
    p.add_argument('--timeout',type=int,default=1200)
    p.add_argument('--allow-run',action='store_true')
    args=p.parse_args(argv)
    checkout=Path(args.checkout).resolve();response=Path(args.response).resolve()
    if response.exists(): p.error('Refusing to reuse an external harness response')
    actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if actual!=args.commit or not re.fullmatch('[0-9a-f]{40}',args.commit): p.error('SkillsBench checkout commit mismatch')
    if subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True).strip():
        p.error('Benchmark checkout must be clean; candidates use separate skill overlays')
    request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    fields(request,{'schema_version','phase','tasks','worker_skill','fresh_environment_required','evidence_contract','grader_visible_to_worker'},
                   {'schema_version','phase','tasks','worker_skill','fresh_environment_required','grader_visible_to_worker'})
    if request['schema_version']!=1 or request['phase'] not in ('support','reward','query'):
        p.error('Invalid request schema or phase')
    if request['fresh_environment_required'] is not True or request['grader_visible_to_worker'] is not False:
        p.error('Unsafe execution contract')
    if args.allow_run:
        if not args.bench_version: p.error('Pin --bench-version before execution')
        version=subprocess.check_output([args.bench_bin,'--version'],text=True).strip()
        if version!=args.bench_version: p.error('BenchFlow version output differs from explicit pin')
        help_text=subprocess.check_output([args.bench_bin,'eval','run','--help'],text=True)
        for flag in ('--jobs-dir','--skills-dir','--skill-mode','--tasks-dir'):
            if flag not in help_text: p.error('Installed BenchFlow lacks required CLI contract: '+flag)
    root=response.parent/'skillsbench_jobs';root.mkdir(exist_ok=False)
    plans=[];results=[]
    for index,task in enumerate(request['tasks']):
        plan=plan_task(task,request['worker_skill'],checkout,root/f'{index:04d}',args.bench_bin,
                       args.agent,args.worker_model,args.sandbox);plans.append(plan)
        if args.allow_run:
            with (root/f'{index:04d}'/'bench.log').open('w') as log:
                subprocess.run(plan['argv'],check=True,timeout=args.timeout,stdout=log,stderr=subprocess.STDOUT,shell=False)
            files=list(Path(plan['jobs_dir']).rglob('result.json'))
            if len(files)!=1: raise RuntimeError('Expected one official rollout result; no best-of/retry selection')
            results.append(normalize_result(files[0],task['id'],request['phase'],args.tokens_field,args.calls_field))
    write_json(root/'plans.json',plans)
    if args.allow_run: write_json(response,dict(schema_version=1,results=results))
    else: print(json.dumps(dict(status='offline_skillsbench_plan_only',plans=len(plans),response_written=False)))


if __name__=='__main__': main()
