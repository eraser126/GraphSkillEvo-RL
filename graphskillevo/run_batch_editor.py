"""Single-panel batch editor. Defaults to offline preflight, never downloads weights."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--mode',choices=['preflight','collect','candidates','train','adapt','evaluate'],default='preflight')
    p.add_argument('--out-root');p.add_argument('--model-path');p.add_argument('--graph')
    p.add_argument('--api-editor',action='store_true');p.add_argument('--allow-api',action='store_true')
    p.add_argument('--allow-training',action='store_true');p.add_argument('--allow-test',action='store_true')
    p.add_argument('--batch-index',type=int,default=0)
    args=p.parse_args(argv)
    from .editor.config import EditorConfig
    from .editor.batch import load_batch_manifest,batches,run_group,train_batches,mean_score
    from .editor.schema import validate_graph,fingerprint
    from .editor.accounting import write_json
    cfg=EditorConfig.load(args.config);data=load_batch_manifest(args.manifest)
    panels=batches(data['splits']['train'],cfg.batch_size,cfg.seed)
    if args.mode=='preflight':
        print(json.dumps(dict(protocol='single_panel_batch',splits={k:len(v) for k,v in data['splits'].items()},
                              batch_sizes=[len(x) for x in panels],group_size=cfg.group_size,
                              training_allowed=data['training_allowed'],model_loaded=False,api_called=False,
                              training_started=False,runtime_readiness='not_proven'),indent=2));return
    if not args.allow_api:p.error('Explicit --allow-api required')
    if not args.out_root:p.error('A NEW --out-root is required')
    root=Path(args.out_root).resolve()
    if root.exists():p.error('Output directory already exists; never overwrite or silently resume')
    if args.mode=='train':
        if not args.allow_training or not data['training_allowed']:p.error('Training/data permission required')
        if args.api_editor or not args.model_path:p.error('GRPO requires the current LOCAL editor weights')
        if not cfg.device.startswith('cuda'):p.error('Production editor training requires an explicit CUDA environment')
        import torch
        if not torch.cuda.is_available():p.error('CUDA unavailable; training was NOT started')
    if args.mode in ('train','adapt','candidates') and not (bool(args.model_path)^args.api_editor):
        p.error('Choose one local editor or frozen API editor')
    if args.mode=='evaluate' and (not args.graph or not args.allow_test):
        p.error('Frozen --graph and --allow-test are required; no test-time selection')
    if args.batch_index<0 or args.batch_index>=len(panels):p.error('Invalid batch index')
    root.mkdir(parents=True,exist_ok=False)
    from .editor.runner import build_runtime,candidate_audit
    from .editor.evidence import Trace
    from .editor.episode import propose
    backend,engine,ledger=build_runtime(cfg,root)
    if args.mode=='evaluate':
        graph=validate_graph(json.loads(Path(args.graph).read_text()))
        rows=backend.run(graph,[dict(id=t['id'],payload=t['payload']) for t in data['splits']['test']],'query')
        write_json(root/'test.json',dict(score=mean_score(rows),graph_hash=fingerprint(graph),
                                       results=rows,editor_updated=False,selection_performed=False));return
    panel=panels[args.batch_index]
    if args.mode=='collect':
        rows=backend.run(data['graph'],[dict(id=t['id'],payload=t['payload']) for t in panel],'support')
        write_json(root/'collection.json',dict(protocol='single_panel_batch',results=rows,training_started=False));return
    from .editor.policy import ChatEditor,LocalLLMEditor
    editor=ChatEditor(cfg.max_new_tokens,cfg.timeout) if args.api_editor else LocalLLMEditor.load(
        args.model_path,cfg.device,cfg.max_new_tokens,cfg.max_context,cfg.lora_rank)
    if args.mode=='train':train_batches(editor,backend,engine,ledger,cfg,data,root)
    elif args.mode=='candidates':run_group(editor,backend,engine,ledger,cfg,data['graph'],panel,root/'group')
    else:
        rows=backend.run(data['graph'],[dict(id=t['id'],payload=t['payload']) for t in panel],'support')
        candidate=propose(data['graph'],[Trace.create(x['trace']) for x in rows],engine,editor,ledger,cfg.variant,deterministic=True)
        write_json(root/'adaptation.json',candidate_audit(candidate));write_json(root/'adapted_graph.json',candidate.graph)


if __name__=='__main__':main()
