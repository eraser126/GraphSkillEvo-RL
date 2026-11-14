"""Train the skill editor on four-task SpreadsheetBench 2 Debugging panels."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import uuid

from graphskillevo.editor.accounting import Ledger, write_json
from graphskillevo.editor.backends import validate_rows
from graphskillevo.editor.batch import batches, run_group, train_batches
from graphskillevo.editor.config import EditorConfig
from graphskillevo.editor.schema import apply_patch, fingerprint, render_worker, validate_graph
from graphskillevo.editor.verification import CheckEngine
from scripts.run_spreadsheet_v2_baseline import LO, grouped_splits, run_tasks


def training_score(result):
    """Score progress while discounting edits that damage cells meant to stay intact."""
    if type(result['missing_output']) is not bool or result['accuracy'] not in (0, 1):
        raise ValueError('Invalid V2 delivery or accuracy record')
    repair = result['modification_accuracy']
    preservation = result['regression_accuracy']
    if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1
           for x in (repair, preservation)):
        raise ValueError('Invalid V2 partial scores')
    return 0.0 if result['missing_output'] else repair * preservation


def load_experiment(dataset, graph_path):
    split_ids, _ = grouped_splits(dataset)
    rows = json.loads((dataset / 'dataset.json').read_text(encoding='utf-8'))
    rows_by_id = {str(row['id']): row for row in rows}
    # Only instructions enter the training manifest. Reference paths and scoring
    # regions are resolved by the worker from private official metadata.
    splits = {
        name: [dict(id=str(tid), family='debugging',
                    payload={'instruction': rows_by_id[str(tid)]['instruction']}) for tid in ids]
        for name, ids in split_ids.items()
    }
    graph = validate_graph(json.loads(graph_path.read_text(encoding='utf-8')))
    return dict(schema_version=2, benchmark='SpreadsheetBench 2 Debugging',
                provenance='Official Debugging data. Reference-workbook groups split 6/2/2 with seed 42.',
                graph=graph, splits=splits), rows_by_id


def load_baseline_results(root, data):
    """Load a user-selected historical initial-skill baseline for this task split."""
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    split_ids = {k: [r['id'] for r in rows] for k, rows in data['splits'].items()}
    if protocol['splits'] != split_ids or protocol['max_turns'] != 30:
        raise ValueError('Baseline split or worker turn limit differs from this experiment')
    if 'initial_graph_hash' in protocol and protocol['initial_graph_hash'] != fingerprint(data['graph']):
        raise ValueError('Baseline graph differs from this experiment')
    train = {r['id']: r for r in data['splits']['train']}
    node_ids = {n['node_id'] for n in data['graph']['nodes']}
    cached = {}
    for tid in protocol['pilot']:
        if tid not in train or tid in cached:
            raise ValueError(f'Invalid baseline task: {tid}')
        score_path = root/tid/'official_result.json'
        if not score_path.is_file():
            continue  # A stopped task without a score must be executed normally.
        score = json.loads(score_path.read_text(encoding='utf-8'))
        trace = json.loads((root/tid/'editor_trace.json').read_text(encoding='utf-8'))
        observed_nodes = {node for e in trace['events'] for node in e['node_ids']}
        contexts = [e['observation'] for e in trace['events'] if e['action']=='TASK_CONTEXT']
        if (score['id'] != tid or score['accuracy'] not in (0, 1)
                or not observed_nodes <= node_ids or contexts != [train[tid]['payload']['instruction'][:8000]]):
            raise ValueError(f'Baseline result does not match current task or graph nodes: {tid}')
        row = dict(id=tid, score=training_score(score), official_accuracy=score['accuracy'], trace=trace)
        validate_rows([train[tid]], [row], 'support')
        cached[tid] = (score, trace)
    return cached


def load_patch_examples(path, data):
    record = json.loads(path.read_text(encoding='utf-8'))
    graph = data['graph']
    if record['schema_version'] != 1 or record['base_graph_hash'] != fingerprint(graph) or not record['examples']:
        raise ValueError('Patch examples must match the initial graph')
    training_ids = {task['id'] for task in data['splits']['train']}
    pairs = []
    for example in record['examples']:
        if example['task_id'] not in training_ids:
            raise ValueError('Patch examples must use training tasks')
        current = graph
        for phase in ('checks', 'graph'):
            patch = example['patches'][phase]
            if not patch['operations'] or not patch['evidence_ids'] or not set(patch['evidence_ids']) <= set(example['allowed_evidence_ids']):
                raise ValueError('Patch examples require an edit and grounded evidence')
            current = apply_patch(current, patch, phase)
        pairs.append(example['patches'])
    return pairs


class DebuggingBackend:
    use_execution_guidance = True

    def __init__(self, dataset, vendor, rows_by_id, root, ledger, exec_timeout, base_config,
                 initial_graph_hash=None, baseline_run=None, baseline_results=None, *, node_checks=True):
        self.dataset = dataset
        self.vendor = vendor
        self.rows_by_id = rows_by_id
        self.root = root
        self.ledger = ledger
        self.exec_timeout = exec_timeout
        self.base_config = base_config
        self.initial_graph_hash = initial_graph_hash
        self.baseline_run = baseline_run
        self.baseline_results = baseline_results or {}
        self.node_checks = node_checks

    def run(self, graph, tasks, phase):
        if phase not in ('support', 'reward', 'query'):
            raise ValueError(f'Invalid worker phase: {phase}')
        panel_dir = self.root / f'{phase}_{uuid.uuid4().hex}'
        panel_dir.mkdir(parents=True, exist_ok=False)
        use_baseline = phase == 'support' and fingerprint(graph) == self.initial_graph_hash
        scores = {}
        official_tasks = []
        for task in tasks:
            tid = task['id']
            if use_baseline and tid in self.baseline_results:
                score, trace = self.baseline_results[tid]
                write_json(panel_dir/tid/'official_result.json', score)
                write_json(panel_dir/tid/'editor_trace.json', trace)
                write_json(panel_dir/tid/'baseline_reuse.json', {'source': str(self.baseline_run/tid)})
                scores[tid] = score
                print(f'Reuse initial-skill result {tid}: accuracy={score["accuracy"]}', flush=True)
            else:
                official_tasks.append(self.rows_by_id[tid])
        reused = len(scores)
        if official_tasks:
            fresh = run_tasks(self.vendor, self.dataset, graph, self.ledger, official_tasks,
                              panel_dir, stage=f'spreadsheet_v2_{phase}', exec_timeout=self.exec_timeout,
                              workers=2, base_config=self.base_config, node_checks=self.node_checks)
            scores.update((score['id'], score) for score in fresh)
        rows = []
        for task in tasks:
            tid = task['id']
            official = scores[tid]
            row = dict(id=tid, score=training_score(official), official_accuracy=official['accuracy'])
            if phase == 'support':
                path = panel_dir/tid/'editor_trace.json'
                row['trace'] = json.loads(path.read_text(encoding='utf-8'))
                row['outcome'] = {key: official[key] for key in (
                    'missing_output', 'modification_accuracy', 'regression_accuracy', 'tool_calls')}
            rows.append(row)
        validated = validate_rows(tasks, rows, phase)
        if reused:
            write_json(panel_dir/'summary.json', {'completed':len(rows), 'planned':len(tasks),
                       'passed':sum(r['official_accuracy'] for r in rows), 'reused':reused,
                       'results':[scores[t['id']] for t in tasks]})
        return validated


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=('preflight', 'candidates', 'train'), default='preflight')
    parser.add_argument('--dataset', type=Path, default=Path('data/spreadsheetbench_v2/recalculated/Debugging'))
    parser.add_argument('--graph', type=Path, default=Path('examples/editor/spreadsheetbench_v2_debugging_initial/initial_graph.json'))
    parser.add_argument('--vendor', type=Path, default=Path('vendor/SpreadsheetBench-2'))
    parser.add_argument('--config', type=Path, default=Path('configs/editor/spreadsheet_batch.yaml'))
    parser.add_argument('--out-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--baseline-run', type=Path, help='Reuse scored initial-graph results from this baseline results directory')
    parser.add_argument('--validation-per-workbook', type=int, default=4)
    parser.add_argument('--patch-examples', type=Path, help='Validated historical editor patch pairs from training traces')
    parser.add_argument('--variant', choices=('full', 'fixed_checks', 'no_checks'),
                        help='Use no_checks for graph editing without node verifiers')
    parser.add_argument('--continue-after-first-batch', type=Path,
                        help='Continue a matching run whose first batch completed without updating the editor')
    parser.add_argument('--batch-index', type=int, default=0, help='Panel to use in candidates mode')
    parser.add_argument('--updates', type=int, help='Override the number of training panels to run')
    parser.add_argument('--max-calls', type=int, help='Override the total recorded call budget')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--allow-training', action='store_true')
    args = parser.parse_args(argv)

    cfg = EditorConfig.load(args.config)
    overrides = {key: value for key, value in [('updates', args.updates), ('max_calls', args.max_calls),
                                               ('variant', args.variant)]
                 if value is not None}
    cfg = replace(cfg, **overrides).validate()
    if cfg.batch_size != 4 or cfg.group_size != 4:
        parser.error('This protocol requires four tasks and four candidates per panel')
    dataset = args.dataset.resolve()
    data, rows_by_id = load_experiment(dataset, args.graph)
    default_examples = args.graph.with_name('verified_patch_examples.json')
    examples_path = args.patch_examples or (default_examples if default_examples.is_file() else None)
    examples = load_patch_examples(examples_path, data) if examples_path else []
    baseline_run = args.baseline_run.resolve() if args.baseline_run else None
    baseline_results = load_baseline_results(baseline_run, data) if baseline_run else {}
    source_graph_hash = fingerprint(data['graph'])
    if cfg.variant == 'no_checks':
        worker_guidance = render_worker(data['graph'])
        for node in data['graph']['nodes']:
            node['checks'] = []
        if render_worker(data['graph']) != worker_guidance:
            raise ValueError('Removing checks changed initial worker guidance; cached scores cannot be reused')
        examples = [{'graph': pair['graph']} for pair in examples]
    validation_groups = {}
    for task in data['splits']['val']:
        workbook = rows_by_id[task['id']]['golden_response_path']
        validation_groups.setdefault(workbook, []).append(task)
    if not validation_groups or not 1 <= args.validation_per_workbook <= min(map(len, validation_groups.values())):
        parser.error('Invalid validation task count per workbook')
    data['splits']['val'] = [
        tasks[min(len(tasks) - 1, (slot * len(tasks)) // args.validation_per_workbook + offset % 2)]
        for offset, tasks in enumerate(validation_groups.values())
        for slot in range(args.validation_per_workbook)
    ]
    panels = batches(data['splits']['train'], cfg.batch_size, cfg.seed)
    if any(len(panel) != 4 for panel in panels):
        raise ValueError('Training split did not yield complete four-task panels')
    if not 0 <= args.batch_index < len(panels):
        parser.error('Invalid --batch-index')
    protocol = dict(benchmark=data['benchmark'], split_sizes={k: len(v) for k,v in data['splits'].items()},
                    training_panels=[[r['id'] for r in panel] for panel in panels], candidates_per_panel=4,
                    worker_executions_per_panel_at_most=20, task_workers=2, max_calls=cfg.max_calls,
                    max_tokens=cfg.max_tokens,
                    reward='mean delta in official modification_accuracy * regression_accuracy; missing output scores zero',
                    validation_selection='official task accuracy, then mean training score',
                    validation_task_ids=[r['id'] for r in data['splits']['val']],
                    baseline_run=str(baseline_run) if baseline_run else None,
                    cached_baseline_task_ids=sorted(baseline_results),
                    variant=cfg.variant, source_graph_hash=source_graph_hash,
                    initial_graph_hash=fingerprint(data['graph']))
    if examples:
        protocol['patch_examples'] = dict(count=len(examples), hash=fingerprint(examples))
    if args.mode == 'preflight':
        print(json.dumps(protocol, indent=2))
        return

    if not args.allow_api or args.out_root is None or args.model_path is None:
        parser.error('Execution requires --allow-api, --out-root and a local --model-path')
    if args.mode == 'train' and not args.allow_training:
        parser.error('Training requires --allow-training')
    previous_batch = None
    if args.continue_after_first_batch:
        previous = args.continue_after_first_batch.resolve()
        previous_batch = previous/'updates/000000'
        old_protocol = json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
        old_identity = json.loads((previous/'run_identity.json').read_text(encoding='utf-8'))
        old_metrics = json.loads((previous_batch/'metrics.json').read_text(encoding='utf-8'))
        if (args.mode != 'train' or cfg.updates < 2 or len(panels) < 2 or cfg.state_bank_size != 0
                or old_identity['config'] != json.loads(json.dumps(cfg.to_dict()))
                or any(old_protocol[k] != protocol[k] for k in ('initial_graph_hash', 'training_panels', 'baseline_run'))
                or old_metrics['updated'] is not False or old_metrics['task_ids'] != protocol['training_panels'][0]
                or (previous/'updates/000001').exists() or (previous_batch/'dev.json').exists()):
            parser.error('Previous run must match and have completed only the first batch, without a parameter update or validation')
    if not args.model_path.is_dir():
        parser.error('--model-path must be a local directory')
    if args.mode == 'train':
        import torch
        if not cfg.device.startswith('cuda') or not torch.cuda.is_available():
            parser.error('Editor training requires CUDA')
    vendor = args.vendor.resolve()
    for path in (vendor/'evaluation/evaluation.py', vendor/'evaluation/open_spreadsheet.py', LO/'soffice', LO/'python'):
        if not path.is_file():
            parser.error(f'Missing V2 runtime dependency: {path}')
    from graphskillevo.config import load_graphskillevo_config
    from graphskillevo.editor.policy import LocalLLMEditor
    from graphskillevo.model_setup import configure_models

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'protocol.json', {**protocol, 'mode':args.mode,
                                     'selected_panel':args.batch_index if args.mode=='candidates' else None})
    if previous_batch:
        preserved = root/'updates/000000'
        shutil.copytree(previous_batch, preserved, ignore=shutil.ignore_patterns('frozen_editor.json'))
        previous_ledger = previous/'editor_costs.jsonl'
        if previous_ledger.is_file():
            shutil.copyfile(previous_ledger, root/'editor_costs.jsonl')
        write_json(preserved/'metrics.json', {**old_metrics, 'parameter_updates': 0})
        write_json(root/'continuation.json', dict(previous_run=str(previous), skipped_validation=True,
                                                 completed_batches=1, parameter_updates=0))
    base, _ = load_graphskillevo_config(cfg.base_config)
    configure_models(base)
    ledger = Ledger(root, cfg.max_calls, cfg.max_tokens)
    backend = DebuggingBackend(dataset, vendor, rows_by_id, root/'worker_runs', ledger,
                               cfg.exec_timeout, cfg.base_config, protocol['initial_graph_hash'],
                               baseline_run, baseline_results, node_checks=cfg.variant != 'no_checks')
    engine = CheckEngine(ledger=ledger)
    editor = LocalLLMEditor.load(str(args.model_path), cfg.device, cfg.max_new_tokens,
                                 cfg.max_context, cfg.lora_rank)
    if args.mode == 'candidates':
        run_group(editor, backend, engine, ledger, cfg, data['graph'], panels[args.batch_index], root/'group', examples=examples)
    else:
        data['training_allowed'] = True
        train_batches(editor, backend, engine, ledger, cfg, data, root,
                      completed_no_update_batches=int(previous_batch is not None), examples=examples)


if __name__ == '__main__':
    main()
