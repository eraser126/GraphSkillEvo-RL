"""Fixed worker bridges. Only normalized support evidence crosses into editing."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import uuid

from .accounting import write_json
from .schema import ContractError, fields, render_worker
from .evidence import Trace, events_from_conversation


def validate_rows(tasks, rows, phase):
    if not isinstance(rows,list) or len(rows)!=len(tasks):
        raise RuntimeError('Incomplete worker panel; no fabricated zero rewards')
    expected={t['id'] for t in tasks}; found=set()
    for row in rows:
        fields(row, {'id','score','official_accuracy','outcome','trace','tokens','calls','error'}, {'id','score'})
        if row['id'] not in expected or row['id'] in found: raise RuntimeError('Worker task ID mismatch')
        found.add(row['id'])
        if row.get('error'): raise RuntimeError('Worker infrastructure failure')
        if not isinstance(row['score'],(float,int)) or not math.isfinite(row['score']) or not 0<=row['score']<=1:
            raise RuntimeError('Invalid worker score')
        if 'official_accuracy' in row and row['official_accuracy'] not in (0, 1):
            raise RuntimeError('Invalid binary official accuracy')
        if phase=='support':
            trace=Trace.create(row['trace'])
            if trace.public()['task_id']!=row['id'] or trace.public()['success']!=(row.get('official_accuracy', row['score'])==1):
                raise ContractError('Trace/outcome mismatch')
    return rows


class OriginalBackend:
    """Reuse original adapter rollout/scoring; no hidden-reference reflection builder."""
    def __init__(self, adapter, root, ledger, seed=42):
        self.adapter,self.root,self.ledger,self.seed=adapter,Path(root),ledger,seed
        self.root.mkdir(parents=True,exist_ok=True)

    def run(self, graph, tasks, phase):
        if phase not in ('support','reward','query'): raise ValueError('Unknown panel role')
        from graphskillevo.datasets.base import BatchSpec
        out=self.root / f'{phase}_{uuid.uuid4().hex}'
        out.mkdir()
        items=[]
        for t in tasks:
            p=deepcopy(t['payload'])
            if 'id' in p and str(p['id'])!=t['id']: raise ContractError('Payload ID mismatch')
            p['id']=t['id'];items.append(p)
        batch=BatchSpec(phase='train' if phase!='query' else 'eval',split='train' if phase!='query' else 'valid_unseen',
                        seed=self.seed,batch_size=len(items),payload=items)
        def execute():
            env=self.adapter.build_env_from_batch(batch,out_root=str(out))
            try: return self.adapter.rollout(env,render_worker(graph),str(out),use_eval_feedback=False)
            finally:
                close=getattr(env,'close',None)
                if callable(close): close()
        raw=self.ledger.call('worker_'+phase,execute)
        rows=[]
        for r in raw:
            if not isinstance(r,dict): r=vars(r)
            if r.get('error'): raise RuntimeError('Original worker reported infrastructure failure')
            if 'hard' not in r: raise RuntimeError('Original worker missing hard score')
            row=dict(id=str(r['id']),score=float(r['hard']))
            if phase=='support':
                directory=out/'predictions'/row['id']
                captured=directory/'editor_trace.json'
                if captured.is_file(): row['trace']=json.loads(captured.read_text(encoding='utf-8'))
                else:
                    p=directory/'conversation.json'
                    conversation=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else r.get('conversation',[])
                    row['trace']=dict(task_id=row['id'],events=events_from_conversation(conversation),
                                      snapshots=[],success=row['score']==1)
            rows.append(row)
        return validate_rows(tasks,rows,phase)


class CommandBackend:
    """Version-independent external harness contract, including SkillsBench/BenchFlow.

    A trusted wrapper receives a request with visible task handles and worker skill,
    executes each task in a fresh isolated environment, grades outside the agent,
    and writes normalized JSON. This bridge does NOT claim that BenchFlow natively
    accepts our schema. See docs/EDITOR_PREPARATION.md for integration requirements.
    """
    def __init__(self, argv, root, ledger, timeout=600):
        if not isinstance(argv,list) or not argv or not all(isinstance(x,str) for x in argv):
            raise ContractError('Runner must be an explicit argv list, never shell text')
        if not any('{request}' in x for x in argv) or not any('{response}' in x for x in argv):
            raise ContractError('Runner argv must include request and response placeholders')
        self.argv,self.root,self.ledger,self.timeout=argv,Path(root),ledger,timeout
        self.root.mkdir(parents=True,exist_ok=True)

    def run(self,graph,tasks,phase):
        if phase not in ('support','reward','query'): raise ValueError('Unknown panel role')
        out=self.root/f'{phase}_{uuid.uuid4().hex}';out.mkdir()
        request,response=out/'request.json',out/'response.json'
        write_json(request,dict(schema_version=1,phase=phase,tasks=deepcopy(tasks),
                                worker_skill=render_worker(graph),fresh_environment_required=True,
                                evidence_contract='graphskillevo.editor.evidence.Trace v1',
                                grader_visible_to_worker=False))
        # argv replacement is literal, with shell=False. No response file is reused.
        argv=[a.replace('{request}',str(request.resolve())).replace('{response}',str(response.resolve())) for a in self.argv]
        def execute():
            with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
                subprocess.run(argv,check=True,timeout=self.timeout,cwd=out,stdout=stdout,stderr=stderr,shell=False)
            raw=json.loads(response.read_text(encoding='utf-8'))
            fields(raw,{'schema_version','results'},{'schema_version','results'})
            if raw['schema_version']!=1: raise ContractError('Unknown harness response version')
            return validate_rows(tasks,raw['results'],phase)
        rows=self.ledger.call('external_harness_'+phase,execute)
        # Wrappers must include their internal calls/tokens; outer launch is separately recorded.
        for row in rows:
            if 'tokens' not in row or 'calls' not in row:
                raise ContractError('External harness must report internal usage, even if zero')
            self.ledger.add('external_worker_usage',tokens=row['tokens'],calls=row['calls'])
        return rows


def score(backend, graph, tasks, phase):
    if not tasks: raise ContractError('Cannot score an empty panel')
    rows=backend.run(graph,tasks,phase)
    return sum(r['score'] for r in rows)/len(rows)
