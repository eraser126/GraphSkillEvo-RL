"""Independent opt-in PPO entry point; the original EA entry remains untouched."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--rl-config", required=True)
    parser.add_argument("--mode", choices=("train", "search"), default="search")
    parser.add_argument("--controller", choices=("ppo", "random", "round_robin"), default="ppo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--allow-api", action="store_true", help="Explicitly authorize paid model calls")
    parser.add_argument("--check-config", action="store_true", help="Offline validation only; no adapter/API setup")
    parser.add_argument("--cfg-options", nargs="+", default=[])
    args = parser.parse_args()
    from graphskillevo.config import load_graphskillevo_config
    from graphskillevo.rl.config import RLConfig
    cfg, evolution = load_graphskillevo_config(args.base_config, cfg_options=args.cfg_options,
                                               legacy_overrides={"out_root": args.out_root})
    rl = RLConfig.load(args.rl_config)
    if args.check_config:
        print({"status": "offline_config_valid", "env": cfg["env"],
               "target": cfg["target_model"], "optimizer": cfg["optimizer_model"],
               "action_count": 3 + 2 * rl.archive_capacity, "api_called": False})
        return
    if not args.allow_api:
        parser.error("No API calls permitted without --allow-api; use --check-config for offline preflight")
    if args.mode == "train" and args.controller != "ppo":
        parser.error("Training is for PPO; use search for random/round_robin baselines")
    if args.mode == "search" and args.controller == "ppo" and not args.checkpoint:
        parser.error("PPO search requires --checkpoint")
    if args.resume and (args.mode != "train" or not args.checkpoint):
        parser.error("--resume requires train mode and a trusted checkpoint")
    from graphskillevo.rl.evaluator_bridge import write_json
    from graphskillevo.rl.trainer import build_trainer
    root = Path(args.out_root).resolve()
    if (root / "resolved_config.json").exists() and not args.resume:
        parser.error("Output directory already used; choose a fresh directory or explicitly --resume")
    root.mkdir(parents=True, exist_ok=True)
    safe_cfg = {k: ("<redacted>" if any(s in k.lower() for s in ("api_key", "password", "secret")) else v)
                for k, v in cfg.items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        commit = "unknown"
    write_json(root / "resolved_config.json", {"base": safe_cfg, "rl": rl.to_dict(), "commit": commit,
                                              "mode": args.mode, "controller": args.controller})
    trainer = build_trainer(cfg, evolution, rl, root)
    if args.checkpoint:
        trainer.load(args.checkpoint, resume=args.resume)
    summary = trainer.train() if args.mode == "train" else trainer.search(args.controller)
    write_json(root / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
