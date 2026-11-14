"""Runtime orchestration; no work is performed on import."""
from dataclasses import asdict
from pathlib import Path
import json
import random

from .accounting import Ledger,write_json
from .backends import OriginalBackend,CommandBackend,score
from .evidence import Trace,SnapshotStore
from .episode import propose
from .schema import fingerprint,render_worker
from .verification import CheckEngine,FixedChatJudge


def build_runtime(cfg,root):
    ledger=Ledger(root,cfg.max_calls,cfg.max_tokens)
    if cfg.backend in ('original','spreadsheet') or cfg.fixed_semantic_judge:
        from graphskillevo.config import load_graphskillevo_config
        from graphskillevo.model_setup import configure_models
        base,_=load_graphskillevo_config(cfg.base_config)
        configure_models(base)
    if cfg.backend=='original':
        from graphskillevo.adapters import get_adapter
        adapter=get_adapter(base);adapter.setup(base)
        backend=OriginalBackend(adapter,Path(root)/'worker_runs',ledger,cfg.seed)
    elif cfg.backend=='spreadsheet':
        from .spreadsheet_backend import SpreadsheetBackend
        backend=SpreadsheetBackend(base['data_root'], 'vendor/SpreadsheetBench',
                                   'configs/editor/spreadsheet_official.lock.json',
                                   Path(root)/'worker_runs',ledger,max_turns=base.get('max_turns',30),
                                   soffice='/opt/libreoffice7.6/program/soffice',exec_timeout=cfg.exec_timeout)
    else: backend=CommandBackend(list(cfg.runner_argv),Path(root)/'worker_runs',ledger,cfg.timeout)
    engine=CheckEngine(FixedChatJudge(timeout=cfg.timeout) if cfg.fixed_semantic_judge else None,ledger)
    return backend,engine,ledger


def support_traces(backend,graph,tasks,store):
    rows=backend.run(graph,tasks,'support')
    traces=[Trace.create(r['trace']) for r in rows]
    for trace in traces: store.put(trace)
    return traces


def candidate_audit(candidate,include_actions=False):
    record=dict(valid=candidate.valid,reward=candidate.reward,audit=candidate.audit,
                graph=candidate.graph,turn_count=len(candidate.turns))
    if include_actions: record['turns']=[asdict(t) for t in candidate.turns]
    return record


def train(editor,backend,engine,ledger,cfg,manifest,root):
    import torch
    from .grpo import update_group
    root=Path(root);units=[u for u in manifest['items'] if u['split']=='train']
    if not units: raise ValueError('No permitted train skill units')
    params=[p for p in editor.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg.learning_rate)
    store=SnapshotStore(root/'evidence')
    random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    write_json(root/'run_identity.json',dict(config=cfg.to_dict(),manifest_hash=fingerprint(manifest),
                                            method='two_turn_llm_grpo',worker_frozen=True))
    completed=0
    best_dev=float('-inf')
    try:
        for update in range(cfg.updates):
            unit=units[update%len(units)]; graph=unit['graph']
            directory=root/'updates'/f'{update:06d}';directory.mkdir(parents=True)
            traces=support_traces(backend,graph,unit['support'],store)
            baseline=score(backend,graph,unit['reward'],'reward')
            candidates=[]
            # Freeze weights for the complete group. Never update between candidates.
            for index in range(cfg.group_size):
                before_calls,before_tokens=ledger.calls,ledger.tokens
                c=propose(graph,traces,engine,editor,ledger,cfg.variant)
                calls,tokens=ledger.calls-before_calls,ledger.tokens-before_tokens
                if c.valid:
                    # NOOP / check-only candidates do not need a noisy duplicate baseline execution.
                    same=render_worker(c.graph)==render_worker(graph)
                    after=baseline if same else score(backend,c.graph,unit['reward'],'reward')
                    calls,tokens=ledger.calls-before_calls,ledger.tokens-before_tokens
                    c.reward=after-baseline-cfg.cost_weight*(calls+tokens/cfg.cost_unit)
                else:
                    after=None;c.reward=-cfg.invalid_penalty-cfg.cost_weight*(calls+tokens/cfg.cost_unit)
                c.audit.update(reward_before=baseline,reward_after=after,calls=calls,tokens=tokens,
                               unit_id=unit['unit_id'],support_ids=[t['id'] for t in unit['support']],
                               reward_ids=[t['id'] for t in unit['reward']])
                write_json(directory/f'candidate_{index}.json',candidate_audit(c,include_actions=True))
                candidates.append(c)
            metrics=update_group(editor,candidates,optimizer,cfg.clip,cfg.epochs,cfg.max_grad_norm)
            write_json(directory/'metrics.json',metrics)
            completed=update+1
            dev_units=[u for u in manifest['items'] if u['split']=='dev']
            if dev_units and completed%cfg.dev_frequency==0:
                dev_scores=[]
                for dev in dev_units:
                    dev_traces=support_traces(backend,dev['graph'],dev['support'],store)
                    frozen=propose(dev['graph'],dev_traces,engine,editor,ledger,cfg.variant,deterministic=True)
                    dev_scores.append(score(backend,frozen.graph,dev['query'],'query'))
                dev_mean=sum(dev_scores)/len(dev_scores)
                write_json(directory/'dev_selection.json',dict(mean=dev_mean,scores=dev_scores,
                           editor_updated=False,used_for_checkpoint_selection_only=True))
                if dev_mean>best_dev:
                    best_dev=dev_mean;editor.save(root/'best_dev_model')
            checkpoint=root/f'checkpoint_{completed:06d}';editor.save(checkpoint)
            torch.save(dict(optimizer=optimizer.state_dict(),completed_updates=completed,
                            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            manifest_hash=fingerprint(manifest),config=cfg.to_dict(),best_dev=best_dev if best_dev!=float('-inf') else None,
                            calls=ledger.calls,tokens=ledger.tokens),
                       checkpoint/'training_state.pt')
        result=dict(status='training_complete',updates=completed,query_evaluated=False,
                    calls=ledger.calls,tokens=ledger.tokens)
        write_json(root/'summary.json',result);return result
    except Exception as exc:
        # Discard unfinished group; persist state rather than treating an outage as zero reward.
        editor.save(root/'interrupted_model')
        write_json(root/'interrupted.json',dict(error_type=type(exc).__name__,completed_updates=completed,
                                              unfinished_group_discarded=True,calls=ledger.calls,tokens=ledger.tokens))
        raise


def adapt(editor,backend,engine,ledger,cfg,unit,root):
    """A single frozen two-turn proposal; query is not accessed or used for selection."""
    root=Path(root);store=SnapshotStore(root/'evidence')
    # Even an externally supplied unit's reward/query fields never reach propose().
    traces=support_traces(backend,unit['graph'],unit['support'],store)
    if hasattr(editor,'model'):
        editor.model.eval()
        for p in editor.model.parameters(): p.requires_grad_(False)
    candidate=propose(unit['graph'],traces,engine,editor,ledger,cfg.variant,deterministic=True)
    artifact=dict(schema_version=1,unit_id=unit['unit_id'],graph=candidate.graph,
                   unit_fingerprint=fingerprint(unit),
                   graph_hash=fingerprint(candidate.graph),source_graph_hash=fingerprint(unit['graph']),
                   support_ids=[t['id'] for t in unit['support']],query_used=False,
                   editor_parameters_updated=False,valid_edit=candidate.valid,
                   evaluation_protocol='one_candidate_fixed_budget')
    write_json(root/'adapted_skill.json',artifact)
    (root/'adapted_skill.md').write_text(render_worker(candidate.graph),encoding='utf-8')
    write_json(root/'edit_audit.json',candidate_audit(candidate,include_actions=True))
    return artifact


def evaluate_query(backend,unit,artifact,root):
    """Separate final command, no editor/checker update and no candidate selection."""
    if artifact['unit_id']!=unit['unit_id'] or artifact['graph_hash']!=fingerprint(artifact['graph']):
        raise ValueError('Frozen artifact identity/hash mismatch')
    if artifact.get('source_graph_hash')!=fingerprint(unit['graph']):
        raise ValueError('Frozen artifact source graph mismatch')
    if artifact.get('unit_fingerprint')!=fingerprint(unit):
        raise ValueError('Adaptation/evaluation manifest changed after export')
    expected=[t['id'] for t in unit['support']]
    if artifact.get('support_ids')!=expected or artifact.get('query_used') is not False:
        raise ValueError('Adaptation provenance mismatch')
    result=dict(unit_id=unit['unit_id'],graph_hash=artifact['graph_hash'],
                score=score(backend,artifact['graph'],unit['query'],'query'),
                task_ids=[t['id'] for t in unit['query']],selection_performed=False)
    write_json(Path(root)/'query_result.json',result);return result
