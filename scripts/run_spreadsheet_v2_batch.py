"""Train the skill editor on four-task SpreadsheetBench 2 Debugging panels."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import uuid

from graphskillevo.editor.accounting import Ledger, write_json
from graphskillevo.editor.backends import validate_rows
from graphskillevo.editor.batch import batches, run_group, train_batches
from graphskillevo.editor.config import EditorConfig
from graphskillevo.editor.schema import fingerprint, validate_graph
from graphskillevo.editor.verification import CheckEngine
from scripts.run_spreadsheet_v2_baseline import LO, grouped_splits, load_module, run_tasks, score_task


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
        row = dict(id=tid, score=score['accuracy'], trace=trace)
        validate_rows([train[tid]], [row], 'support')
        cached[tid] = (score, trace)
    return cached


class DebuggingBackend:
    def __init__(self, dataset, vendor, rows_by_id, root, ledger, exec_timeout, base_config,
                 initial_graph_hash=None, baseline_run=None, baseline_results=None, resume_query=None):
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
        self.resume_query = resume_query

    def finish_validation(self, graph, tasks):
        panel = self.resume_query
        self.resume_query = None
        grader = load_module(self.vendor/'evaluation/evaluation.py', 'spreadsheet_v2_official_eval')
        scores = []
        remaining = []
        for task in tasks:
            tid = task['id']
            directory = panel/tid
            score_path = directory/'official_result.json'
            if score_path.is_file():
                score = json.loads(score_path.read_text())
            elif (directory/'conversation.json').is_file():
                # Validation does not consume node traces. Keep the saved agent
                # transcript and grade the existing output without inventing evidence.
                turns = len(json.loads((directory/'conversation.json').read_text()))
                score = score_task(self.vendor, grader, self.dataset, graph, panel,
                                   self.rows_by_id[tid], directory, None, turns)
            elif not directory.exists():
                remaining.append(self.rows_by_id[tid])
                continue
            else:
                raise ValueError(f'No recoverable validation transcript for {tid}')
            if remaining or score['id'] != tid or score['accuracy'] not in (0, 1):
                raise ValueError(f'Invalid saved validation result: {tid}')
            scores.append(score)
        print(f'Resume first validation: {len(scores)} saved executions, {len(remaining)} remaining', flush=True)
        if remaining:
            scores = run_tasks(self.vendor, self.dataset, graph, self.ledger, remaining,
                               panel, stage='spreadsheet_v2_query', exec_timeout=self.exec_timeout,
                               workers=2, base_config=self.base_config, prior_results=scores)
        return validate_rows(tasks, [dict(id=s['id'], score=s['accuracy']) for s in scores], 'query')

    def run(self, graph, tasks, phase):
        if phase not in ('support', 'reward', 'query'):
            raise ValueError(f'Invalid worker phase: {phase}')
        if phase == 'query' and self.resume_query is not None:
            if fingerprint(graph) != self.initial_graph_hash:
                raise ValueError('Saved validation requires the initial graph')
            return self.finish_validation(graph, tasks)
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
                              workers=2, base_config=self.base_config)
            scores.update((score['id'], score) for score in fresh)
        rows = []
        for task in tasks:
            tid = task['id']
            row = dict(id=tid, score=scores[tid]['accuracy'])
            if phase == 'support':
                path = panel_dir/tid/'editor_trace.json'
                row['trace'] = json.loads(path.read_text(encoding='utf-8'))
            rows.append(row)
        validated = validate_rows(tasks, rows, phase)
        if reused:
            write_json(panel_dir/'summary.json', {'completed':len(rows), 'planned':len(tasks),
                       'passed':sum(r['score']==1 for r in rows), 'reused':reused,
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
    parser.add_argument('--batch-index', type=int, default=0, help='Panel to use in candidates mode')
    parser.add_argument('--updates', type=int, help='Override the number of training panels to run')
    parser.add_argument('--max-calls', type=int, help='Override the total recorded call budget')
    parser.add_argument('--resume-first-validation', action='store_true',
                        help='Resume an interrupted first validation after a skipped optimizer step')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--allow-training', action='store_true')
    args = parser.parse_args(argv)

    cfg = EditorConfig.load(args.config)
    overrides = {key: value for key, value in [('updates', args.updates), ('max_calls', args.max_calls)]
                 if value is not None}
    cfg = replace(cfg, **overrides).validate()
    if cfg.batch_size != 4 or cfg.group_size != 4:
        parser.error('This protocol requires four tasks and four candidates per panel')
    dataset = args.dataset.resolve()
    data, rows_by_id = load_experiment(dataset, args.graph)
    baseline_run = args.baseline_run.resolve() if args.baseline_run else None
    baseline_results = load_baseline_results(baseline_run, data) if baseline_run else {}
    panels = batches(data['splits']['train'], cfg.batch_size, cfg.seed)
    if any(len(panel) != 4 for panel in panels):
        raise ValueError('Training split did not yield complete four-task panels')
    if not 0 <= args.batch_index < len(panels):
        parser.error('Invalid --batch-index')
    protocol = dict(benchmark=data['benchmark'], split_sizes={k: len(v) for k,v in data['splits'].items()},
                    training_panels=[[r['id'] for r in panel] for panel in panels], candidates_per_panel=4,
                    worker_executions_per_panel_at_most=20, task_workers=2, max_calls=cfg.max_calls,
                    max_tokens=cfg.max_tokens, reward='difference in official task accuracy',
                    baseline_run=str(baseline_run) if baseline_run else None,
                    cached_baseline_task_ids=sorted(baseline_results),
                    initial_graph_hash=fingerprint(data['graph']))
    if args.mode == 'preflight':
        print(json.dumps(protocol, indent=2))
        return

    if not args.allow_api or args.out_root is None or args.model_path is None:
        parser.error('Execution requires --allow-api, --out-root and a local --model-path')
    if args.mode == 'train' and not args.allow_training:
        parser.error('Training requires --allow-training')
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
    saved_protocol = {**protocol, 'mode':args.mode,
                      'selected_panel':args.batch_index if args.mode=='candidates' else None}
    resume_query = None
    if args.resume_first_validation:
        if args.mode != 'train' or cfg.state_bank_size:
            parser.error('First-validation recovery requires train mode and no state bank')
        if json.loads((root/'protocol.json').read_text()) != saved_protocol:
            raise ValueError('Saved protocol differs from the requested experiment')
        data['training_allowed'] = True
        identity = json.loads((root/'run_identity.json').read_text())
        if identity['config'] != json.loads(json.dumps(cfg.to_dict())) or identity['manifest_hash'] != fingerprint(data):
            raise ValueError('Saved training config or manifest differs')
        interrupted = json.loads((root/'interrupted.json').read_text())
        first = root/'updates'/'000000'
        metrics = json.loads((first/'metrics.json').read_text())
        queries = list((root/'worker_runs').glob('query_*'))
        if (interrupted['completed_updates'] != 1 or metrics['updated']
                or metrics['task_ids'] != saved_protocol['training_panels'][0]
                or (first/'dev.json').exists() or any(root.glob('checkpoint_*'))
                or len(queries) != 1
                or json.loads((first/'frozen_editor.json').read_text())['valid']):
            raise ValueError('Run is outside the supported first-validation recovery point')
        if args.model_path.resolve() != root/'interrupted_model':
            raise ValueError('Load the interrupted run adapter to recover its parameters')
        resume_query = queries[0]
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'protocol.json', saved_protocol)
    base, _ = load_graphskillevo_config(cfg.base_config)
    configure_models(base)
    ledger = Ledger(root, cfg.max_calls, cfg.max_tokens)
    backend = DebuggingBackend(dataset, vendor, rows_by_id, root/'worker_runs', ledger,
                               cfg.exec_timeout, cfg.base_config, protocol['initial_graph_hash'],
                               baseline_run, baseline_results, resume_query)
    engine = CheckEngine(ledger=ledger)
    editor = LocalLLMEditor.load(str(args.model_path), cfg.device, cfg.max_new_tokens,
                                 cfg.max_context, cfg.lora_rank)
    if args.mode == 'candidates':
        run_group(editor, backend, engine, ledger, cfg, data['graph'], panels[args.batch_index], root/'group')
    else:
        data['training_allowed'] = True
        train_batches(editor, backend, engine, ledger, cfg, data, root,
                      resume_first_validation=args.resume_first_validation)


if __name__ == '__main__':
    main()
