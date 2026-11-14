"""Single-panel, multi-task editing. No execution on import; legacy path is preserved."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

from .schema import ContractError, fields, fingerprint, validate_graph, render_worker
from .evidence import Trace, SnapshotStore
from .episode import propose
from .runner import candidate_audit
from .accounting import write_json


def load_batch_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    fields(data, {'schema_version','benchmark','training_allowed','provenance','graph','splits'},
           {'schema_version','benchmark','training_allowed','provenance','graph','splits'})
    if data['schema_version'] != 2 or type(data['training_allowed']) is not bool:
        raise ContractError('Expected single-panel manifest v2')
    if not isinstance(data['provenance'],str) or not data['provenance']:
        raise ContractError('Data provenance required')
    fields(data['splits'], {'train','val','test'}, {'train','val','test'})
    ids, payloads = set(), set()
    for split, rows in data['splits'].items():
        if not isinstance(rows,list) or not rows: raise ContractError('Every split must contain tasks')
        for row in rows:
            fields(row, {'id','payload','family'}, {'id','payload','family'})
            tid = row['id']
            if not isinstance(tid,str) or not tid or tid in ('.','..') or '/' in tid or '\\' in tid:
                raise ContractError('Unsafe task ID')
            if not isinstance(row['payload'],dict) or not isinstance(row['family'],str) or not row['family']:
                raise ContractError('Task payload/family required')
            # ID alone is insufficient: ignore the ID/path when checking duplicate content.
            content = {k:v for k,v in row['payload'].items() if k not in ('id','spreadsheet_path')}
            identity = fingerprint(content)
            if tid in ids or identity in payloads: raise ContractError('Duplicate task or payload across dataset')
            ids.add(tid);payloads.add(identity)
            forbidden = {'answers','gold_answer','gold_answers','golden','reference_answer'} & set(row['payload'])
            if forbidden: raise ContractError('Private answer in editable task payload')
    data['graph'] = validate_graph(json.loads((path.parent/data['graph']).read_text()))
    if 'skillsbench' in data['benchmark'].lower() and data['training_allowed']:
        raise ContractError('SkillsBench is not permitted for weight training')
    return data


def batches(tasks, size, seed):
    """Family-local shuffle; retain every task, including small remainder batches."""
    if size < 1: raise ValueError('Positive batch size required')
    rng = random.Random(seed); groups = {}
    for t in tasks: groups.setdefault(t['family'],[]).append(t)
    result = [];singletons=[]
    for family in sorted(groups):
        rows = groups[family][:]; rng.shuffle(rows)
        local=[rows[i:i+size] for i in range(0,len(rows),size)]
        if size>=3 and len(local)>1 and len(local[-1])==1:
            local[-1].insert(0,local[-2].pop())
        for panel in local:
            if len(panel)==1 and size>1:singletons.extend(panel)
            else:result.append(panel)
    # Rare singleton families join a transparent mixed remainder panel; never drop tasks.
    while len(singletons)>=2:
        n=min(size,len(singletons));result.append(singletons[:n]);singletons=singletons[n:]
    if singletons:
        room=next((p for p in result if len(p)<size),None)
        if room is not None:room.extend(singletons)
        elif result and len(result[-1])>2:result.append([result[-1].pop()]+singletons)
        else:result.append(singletons)
    rng.shuffle(result)
    return result


def mean_score(rows):
    if not rows: raise ContractError('Empty evaluation panel')
    return sum(x['score'] for x in rows)/len(rows)


def paired_changes(before,after):
    a={x['id']:x['score'] for x in before};b={x['id']:x['score'] for x in after}
    if set(a)!=set(b): raise ContractError('Candidate panel differs from baseline')
    return dict(improved=sum(b[k]>a[k] for k in a),regressed=sum(b[k]<a[k] for k in a),
                tied=sum(b[k]==a[k] for k in a))


def run_group(editor,backend,engine,ledger,cfg,graph,tasks,root,update_fn=None,optimizer=None):
    """One baseline pass supplies BOTH evidence and reward, then K frozen-weight proposals."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    worker_tasks=[dict(id=t['id'],payload=deepcopy(t['payload'])) for t in tasks]
    baseline=backend.run(graph,worker_tasks,'support')
    traces=[Trace.create(x['trace']) for x in baseline]
    store=SnapshotStore(root/'evidence')
    for trace in traces:store.put(trace)
    candidates=[];estimated_exec=len(tasks)
    for index in range(cfg.group_size):
        calls,tokens=ledger.calls,ledger.tokens
        candidate=propose(graph,traces,engine,editor,ledger,cfg.variant)
        after=None;changes=None;reused=False
        if candidate.valid:
            reused=render_worker(candidate.graph)==render_worker(graph)
            rows=baseline if reused else backend.run(candidate.graph,worker_tasks,'reward')
            if not reused:estimated_exec+=len(tasks)
            after=mean_score(rows);changes=paired_changes(baseline,rows)
            candidate.reward=after-mean_score(baseline)
        else:candidate.reward=-cfg.invalid_penalty
        # Default cost_weight=0 for new config. Charges are explicitly reported.
        delta_calls,delta_tokens=ledger.calls-calls,ledger.tokens-tokens
        candidate.reward-=cfg.cost_weight*(delta_calls+delta_tokens/cfg.cost_unit)
        candidate.audit.update(protocol='single_panel_batch',task_ids=[t['id'] for t in tasks],
                               reward_before=mean_score(baseline),reward_after=after,
                               paired_changes=changes,baseline_reused=reused,calls=delta_calls,tokens=delta_tokens)
        write_json(root/f'candidate_{index}.json',candidate_audit(candidate,include_actions=True))
        candidates.append(candidate)
    # Validation/frozen candidates are complete before any optimizer step.
    metrics=update_fn(editor,candidates,optimizer,cfg.clip,cfg.epochs,cfg.max_grad_norm) if update_fn else {'updated':False,'reason':'frozen_candidate_collection'}
    metrics.update(task_ids=[t['id'] for t in tasks],baseline=mean_score(baseline),
                   rewards=[c.reward for c in candidates],expected_worker_executions=estimated_exec)
    write_json(root/'metrics.json',metrics)
    return candidates,metrics


def train_batches(editor,backend,engine,ledger,cfg,data,root, *, resume_first_validation=False):
    """Train only on train; val chooses checkpoint; test is never accessed here."""
    import torch
    from .grpo import update_group
    if not data['training_allowed']: raise ContractError('Training permission is false')
    root=Path(root);base=data['graph'];bank=[];best=float('-inf');completed=0
    random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    optimizer=torch.optim.AdamW([p for p in editor.model.parameters() if p.requires_grad],lr=cfg.learning_rate)
    planned=min(cfg.updates,cfg.training_epochs*len(batches(data['splits']['train'],cfg.batch_size,cfg.seed)))
    if not resume_first_validation:
        write_json(root/'run_identity.json',dict(config=cfg.to_dict(),manifest_hash=fingerprint(data),
                                               protocol='single_panel_batch',worker_frozen=True,test_used=False))
    try:
        for epoch in range(cfg.training_epochs):
            for panel in batches(data['splits']['train'],cfg.batch_size,cfg.seed+epoch):
                if completed>=cfg.updates:break
                # Optional bank contains train-generated states only. Original graph remains anchored.
                graph=bank[(completed//2)%len(bank)] if bank and completed%2 else base
                directory=root/'updates'/f'{completed:06d}'
                resume_validation = resume_first_validation and completed == 0
                if resume_validation:
                    candidates = []
                else:
                    candidates,metrics=run_group(editor,backend,engine,ledger,cfg,graph,panel,directory,update_group,optimizer)
                completed+=1
                if cfg.state_bank_size:
                    eligible=[c for c in candidates if c.valid and c.audit['reward_after'] is not None and c.audit['reward_after']>c.audit['reward_before']]
                    if eligible:
                        bank.append(deepcopy(max(eligible,key=lambda c:c.reward).graph));bank=bank[-cfg.state_bank_size:]
                if completed==1 or completed%cfg.dev_frequency==0 or completed==planned:
                    # Evaluate the UPDATED editor on train evidence, then score on val.
                    # Val outcomes select checkpoint only, never enter editor context/gradients.
                    frozen_path = directory/'frozen_editor.json'
                    if resume_validation:
                        frozen = json.loads(frozen_path.read_text())
                    else:
                        traces=[Trace.create(json.loads(p.read_text())) for p in (directory/'evidence').glob('*.json')]
                        frozen=candidate_audit(propose(graph,traces,engine,editor,ledger,cfg.variant,deterministic=True))
                        write_json(frozen_path,frozen)
                    pool=[base]+([frozen['graph']] if frozen['valid'] else [])
                    vals=[]
                    val_tasks=[dict(id=t['id'],payload=t['payload']) for t in data['splits']['val']]
                    for g in pool:vals.append(mean_score(backend.run(g,val_tasks,'query')))
                    winner=max(range(len(vals)),key=lambda i:vals[i])
                    write_json(directory/'dev.json',dict(scores=vals,test_used=False,parameters_updated=False))
                    if vals[winner]>best:
                        best=vals[winner];editor.save(root/'best_dev_model')
                        write_json(root/'selected_graph.json',pool[winner])
                checkpoint=root/f'checkpoint_{completed:06d}';editor.save(checkpoint)
                torch.save(dict(optimizer=optimizer.state_dict(),completed_updates=completed,
                                python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                config=cfg.to_dict(),manifest_hash=fingerprint(data),state_bank=bank,
                                calls=ledger.calls,tokens=ledger.tokens),checkpoint/'training_state.pt')
        write_json(root/'summary.json',dict(completed_updates=completed,test_used=False,training_complete=True))
    except Exception as exc:
        editor.save(root/'interrupted_model')
        write_json(root/'interrupted.json',dict(error_type=type(exc).__name__,completed_updates=completed,
                                              unfinished_group_discarded=True))
        raise
