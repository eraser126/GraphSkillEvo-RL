"""Prepared evidence/LLM editor path. Default preflight never loads models or calls APIs."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--mode',choices=['preflight','replay','adapt','train','evaluate'],default='preflight')
    p.add_argument('--out-root')
    p.add_argument('--unit-id')
    p.add_argument('--model-path')
    p.add_argument('--api-editor',action='store_true',help='Untrained frozen baseline, never GRPO')
    p.add_argument('--allow-api',action='store_true')
    p.add_argument('--allow-training',action='store_true')
    p.add_argument('--replay-evidence')
    p.add_argument('--replay-responses')
    p.add_argument('--artifact')
    args=p.parse_args(argv)
    from graphskillevo.editor.config import EditorConfig
    from graphskillevo.editor.data import load_manifest
    cfg=EditorConfig.load(args.config);manifest=load_manifest(args.manifest)
    if args.mode=='preflight':
        import importlib.util
        print(json.dumps(dict(status='offline_preflight_valid',unit_count=len(manifest['items']),
                         method='two_turn_llm_grpo',model_loaded=False,api_called=False,training_started=False,
                         optional_dependencies={x:importlib.util.find_spec(x) is not None for x in ('torch','transformers','peft')},
                         runtime_readiness='not_proven_by_offline_preflight'),indent=2))
        return
    if not args.out_root: p.error('--out-root required')
    if args.mode not in ('replay',) and not args.allow_api:
        p.error('Runtime blocked without --allow-api (also guards external harness calls)')
    if args.mode=='train' and not args.allow_training: p.error('Training blocked without --allow-training')
    if args.mode=='train' and (args.api_editor or not args.model_path): p.error('Training requires a local LLM; API editor is frozen only')
    if args.mode=='adapt' and not (bool(args.model_path) ^ args.api_editor): p.error('Choose one local model or frozen API editor')
    if args.mode=='evaluate' and not args.artifact: p.error('Final evaluation requires frozen --artifact')
    if args.mode=='replay' and (not args.replay_evidence or not args.replay_responses):
        p.error('Offline replay requires explicit evidence and scripted responses')
    units=[u for u in manifest['items'] if u['unit_id']==args.unit_id]
    if args.mode!='train' and len(units)!=1: p.error('Choose a unique --unit-id')
    if args.mode=='train' and not any(u['split']=='train' for u in manifest['items']): p.error('No permitted training units')
    root=Path(args.out_root).resolve()
    if root.exists() and any(root.iterdir()): p.error('Choose a new output directory; do not overwrite prior runs')
    from graphskillevo.editor.accounting import write_json
    root.mkdir(parents=True,exist_ok=True)
    if args.mode=='replay':
        from graphskillevo.editor.evidence import Trace
        from graphskillevo.editor.policy import ScriptedEditor
        from graphskillevo.editor.verification import CheckEngine
        from graphskillevo.editor.episode import propose
        from graphskillevo.editor.runner import candidate_audit
        traces=[Trace.create(x) for x in json.loads(Path(args.replay_evidence).read_text(encoding='utf-8'))]
        expected={t['id'] for t in units[0]['support']}
        if {t.public()['task_id'] for t in traces}!=expected: p.error('Replay evidence differs from support task IDs')
        responses=json.loads(Path(args.replay_responses).read_text(encoding='utf-8'))
        editor=ScriptedEditor([json.dumps(x) for x in responses])
        c=propose(units[0]['graph'],traces,CheckEngine(),editor,variant=cfg.variant)
        write_json(root/'replay.json',candidate_audit(c))
        print(json.dumps(dict(status='offline_contract_replay',valid=c.valid,training_started=False,
                              api_called=False,benchmark_evaluated=False)))
        return
    from graphskillevo.editor.runner import build_runtime,train,adapt,evaluate_query
    backend,engine,ledger=build_runtime(cfg,root)
    if args.mode=='evaluate':
        artifact=json.loads(Path(args.artifact).read_text(encoding='utf-8'))
        print(evaluate_query(backend,units[0],artifact,root));return
    from graphskillevo.editor.policy import LocalLLMEditor,ChatEditor
    editor=ChatEditor(cfg.max_new_tokens,cfg.timeout) if args.api_editor else LocalLLMEditor.load(
        args.model_path,cfg.device,cfg.max_new_tokens,cfg.max_context,cfg.lora_rank)
    result=train(editor,backend,engine,ledger,cfg,manifest,root) if args.mode=='train' else adapt(
        editor,backend,engine,ledger,cfg,units[0],root)
    print(json.dumps({k:v for k,v in result.items() if k!='graph'},ensure_ascii=False))


if __name__=='__main__': main()
