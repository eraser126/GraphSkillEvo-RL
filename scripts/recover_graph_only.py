"""Recover the fifth update and validation from archived run artifacts."""
from dataclasses import fields
import json
from pathlib import Path
import random
import shutil
import sys

import torch
from safetensors.torch import load_file

import scripts.run_spreadsheet_v2_batch as entry
from graphskillevo.editor.accounting import write_json
from graphskillevo.editor.batch import train_batches, validation_score
from graphskillevo.editor.episode import Candidate, EditorTurn
from graphskillevo.editor.grpo import update_group
from graphskillevo.editor.schema import fingerprint

previous = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
read = lambda path: json.loads(path.read_text())
checkpoint = previous/'checkpoint_000004'
batch = previous/'updates/000004'
saved_query, = sorted((previous/'worker_runs').glob('query_*'))


def recover(editor, backend, engine, ledger, cfg, data, root, **kwargs):
    state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=False)
    assert state['manifest_hash'] == fingerprint(data)
    assert state['config'] == cfg.to_dict()
    assert state['completed_updates'] == state['parameter_updates'] == 4
    metrics = read(batch/'metrics.json')
    assert metrics['updated'] and metrics['parameter_updates'] == 5
    assert not (batch/'dev.json').exists()
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    optimizer = torch.optim.AdamW([p for p in editor.model.parameters() if p.requires_grad],lr=cfg.learning_rate)
    optimizer.load_state_dict(state['optimizer'])
    names = {f.name for f in fields(EditorTurn)}
    candidates = []
    for path in sorted(batch.glob('candidate_*.json')):
        item = read(path)
        turns = [EditorTurn(**{k:v for k,v in t.items() if k in names}) for t in item['turns']]
        candidates.append(Candidate(graph=item['graph'],turns=turns,valid=item['valid'],audit=item['audit'],reward=item['reward']))
    assert len(candidates) == 4 and [c.reward for c in candidates] == metrics['rewards']
    replay = update_group(editor,candidates,optimizer,cfg.clip,cfg.epochs,cfg.max_grad_norm)
    assert replay['updated']
    pre = root/'checkpoint_000005_prevalidation'
    editor.save(pre)
    replayed = load_file(str(pre/'adapter_model.safetensors'))
    actual = load_file(str(previous/'interrupted_model/adapter_model.safetensors'))
    assert replayed.keys() == actual.keys()
    mismatches = [k for k,v in actual.items() if not torch.equal(v,replayed[k])]
    max_abs = max((actual[k]-replayed[k]).abs().max().item() for k in mismatches) if mismatches else 0.0
    prior = load_file(str(checkpoint/'adapter_model.safetensors'))
    delta_norm = sum((actual[k]-prior[k]).double().square().sum().item() for k in actual)**0.5
    error_norm = sum((actual[k]-replayed[k]).double().square().sum().item() for k in actual)**0.5
    relative_update_error = error_norm / delta_norm
    write_json(root/'replay_audit.json',dict(mismatched_tensors=len(mismatches),
              max_absolute_difference=max_abs,relative_update_error=relative_update_error))
    assert max_abs <= 5e-7 and relative_update_error < 0.10, (max_abs,relative_update_error)
    # Preserve the exact adapter saved by the original run, with reconstructed AdamW moments.
    shutil.copyfile(previous/'interrupted_model/adapter_model.safetensors',pre/'adapter_model.safetensors')
    from peft.utils.save_and_load import set_peft_model_state_dict
    set_peft_model_state_dict(editor.model, actual)
    state.update(optimizer=optimizer.state_dict(),completed_updates=5,parameter_updates=5,
                 python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 best=(-1.0,-1.0))
    torch.save(state,pre/'training_state.pt')
    shutil.copytree(previous/'updates',root/'updates')
    shutil.copyfile(previous/'run_identity.json',root/'run_identity.json')
    with ledger.path.open('x') as dst:
        dst.write((previous/'editor_costs.jsonl').read_text())
    original = entry.Ledger(previous,cfg.max_calls,cfg.max_tokens)
    ledger.calls,ledger.tokens = original.calls,original.tokens
    write_json(root/'continuation.json',dict(previous_run=str(previous),completed_batches=5,parameter_updates=5,
                                             replay_adapter_max_abs=max_abs,validation_source=str(saved_query)))
    print('Recovered batch 5 optimizer and model',flush=True)
    tasks = [dict(id=t['id'],payload=t['payload']) for t in data['splits']['val']]
    frozen = read(batch/'frozen_editor.json')
    pool = [data['graph']]+([frozen['graph']] if frozen['valid'] else [])
    saved = {}
    for task in tasks:
        tid = task['id']
        folder = saved_query/tid
        if (folder/'official_result.json').exists():
            saved[tid] = read(folder/'official_result.json')
        elif (folder/'conversation.json').exists():
            from scripts.run_spreadsheet_v2_baseline import valid_workbook
            raw = folder/'output.xlsx'
            assert raw.is_file() and not valid_workbook(raw),tid
            saved[tid] = dict(id=tid,accuracy=0.0,modification_accuracy=0.0,
                              regression_accuracy=0.0,missing_output=False,invalid_output=True)
    write_json(root/'recovered_validation_scores.json',saved)
    remaining = [t for t in tasks if t['id'] not in saved]
    rows = [dict(id=tid,score=entry.training_score(result),official_accuracy=result['accuracy'])
            for tid,result in saved.items()]
    if remaining:
        rows.extend(backend.run(pool[0],remaining,'query'))
    assert {r['id'] for r in rows} == {t['id'] for t in tasks}
    vals = [validation_score(rows)]
    for graph in pool[1:]:
        vals.append(validation_score(backend.run(graph,tasks,'query')))
    winner = max(range(len(vals)),key=lambda i:vals[i])
    write_json(root/'updates/000004/dev.json',dict(scores=[s[0] for s in vals],training_scores=[s[1] for s in vals],
                                                  parameters_updated=False,test_used=False,parameter_updates=5))
    editor.save(root/'best_dev_model')
    write_json(root/'selected_graph.json',pool[winner])
    state.update(best=vals[winner],python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(),calls=ledger.calls,tokens=ledger.tokens)
    editor.save(root/'checkpoint_000005')
    torch.save(state,root/'checkpoint_000005/training_state.pt')
    return train_batches(editor,backend,engine,ledger,cfg,data,root,resume_state=state,**kwargs)


if __name__ == "__main__":
    entry.train_batches = recover
    entry.main(['--mode','train','--variant','no_checks',
                '--baseline-run',str(previous.parent/'baseline_cache'),'--out-root',str(destination),
                '--model-path',str(checkpoint),'--allow-api','--allow-training'])
