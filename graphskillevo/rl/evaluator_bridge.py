"""Fixed task panels and accounting around the original evaluator primitives."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import math
import subprocess
import time

from graphskillevo.evaluation import compute_score, build_rollout_reflection_context
from .types import Score, digest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root), cfg
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "cost_ledger.jsonl"
        self.tokens = self.calls = 0
        self.entries = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.tokens += row["tokens"]
                self.calls += row["calls"]
                self.entries += 1

    def check(self):
        if self.tokens >= self.cfg.max_run_tokens or self.calls >= self.cfg.max_run_calls:
            raise BudgetExceeded("Run budget exhausted at operation boundary")

    @contextmanager
    def measure(self, stage, **metadata):
        from graphskillevo.model import get_token_summary
        self.check()
        before = get_token_summary().get("_total", {})
        started, status = time.monotonic(), "ok"
        try:
            yield
        except Exception as exc:
            status = type(exc).__name__  # Do not persist credential-bearing exception messages.
            raise
        finally:
            after = get_token_summary().get("_total", {})
            tokens = max(0, after.get("total_tokens", 0) - before.get("total_tokens", 0))
            calls = max(0, after.get("calls", 0) - before.get("calls", 0))
            self.tokens += tokens
            self.calls += calls
            self.entries += 1
            append_json(self.path, {"entry": self.entries, "stage": stage, "status": status,
                                    "tokens": tokens, "calls": calls,
                                    "wall_seconds": time.monotonic() - started, **metadata})


class EvaluatorBridge:
    def __init__(self, adapter, loader, root, base_cfg, rl_cfg, ledger):
        if loader is None:
            raise ValueError("RL requires a dataloader with explicit stable task IDs")
        self.adapter, self.loader = adapter, loader
        self.root, self.cfg, self.rl_cfg, self.ledger = Path(root), base_cfg, rl_cfg, ledger
        try:
            self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            self.commit = "unknown"
        # Hash all settings for cache isolation; do not write raw secrets.
        self.fingerprint = digest(json.dumps(base_cfg, sort_keys=True, default=str) + self.commit)
        self.panel = None

    def make_panel(self, seed, split="train"):
        if split == "train":
            batch = self.loader.build_train_batch(batch_size=self.rl_cfg.train_probe_size,
                                                  seed=seed, out_root=str(self.root))
        elif split == "valid_seen":
            batch = self.loader.build_eval_batch(env_num=0, split=split, seed=seed, out_root=str(self.root))
        else:
            raise ValueError("RL bridge forbids test feedback; evaluate exported skills separately")
        items = batch.payload
        if not isinstance(items, list) or not items or not all(isinstance(x, dict) and "id" in x for x in items):
            raise ValueError("Fixed panels require a nonempty list payload with explicit task IDs")
        ids = [str(x["id"]) for x in items]
        if len(set(ids)) != len(ids) or batch.batch_size != len(ids):
            raise ValueError("Duplicate panel IDs or inconsistent batch size")
        full = json.dumps(asdict(batch), sort_keys=True, default=str)
        panel_id = digest(full + self.fingerprint)
        write_json(self.root / "panels" / f"{panel_id}.json",
                   {"panel_id": panel_id, "split": split, "seed": seed, "ids": ids,
                    "payload_hash": digest(full), "base_fingerprint": self.fingerprint})
        return deepcopy(batch), panel_id

    def set_train_panel(self, seed):
        self.panel, self.panel_id = self.make_panel(seed)

    def evaluate(self, content, panel=None, panel_id=None):
        batch = deepcopy(self.panel if panel is None else panel)
        panel_id = self.panel_id if panel_id is None else panel_id
        if batch is None:
            raise RuntimeError("No fixed panel selected")
        k = min(self.rl_cfg.reflection_failure_k, len(batch.payload) - 1) if batch.split == "train" else 0
        key = digest(json.dumps({"skill": digest(content), "panel": panel_id,
                                 "settings": self.fingerprint, "k": k,
                                 "hidden": self.rl_cfg.include_hidden_reference}, sort_keys=True))
        cached = self.root / "evaluation_cache" / f"{key}.json"
        if cached.exists():
            append_json(self.root / "cache_hits.jsonl", {"key": key, "panel_id": panel_id, "tokens": 0})
            return Score(**json.loads(cached.read_text()))
        self.ledger.check()
        # Fresh output directory after aborted attempts: native adapters resume by task ID.
        out = self.root / "evaluations" / f"{key}_{self.ledger.entries:08d}"
        out.mkdir(parents=True, exist_ok=True)
        with self.ledger.measure("evaluation", panel_id=panel_id, skill_id=digest(content), tasks=len(batch.payload)):
            env = self.adapter.build_env_from_batch(deepcopy(batch), out_root=str(out))
            try:
                rows = self.adapter.rollout(env, content, str(out), use_eval_feedback=False)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        if len(rows) != len(batch.payload):
            raise RuntimeError("Incomplete rollout panel; refusing to assign zero reward")
        result_ids = [str(x.get("id")) if isinstance(x, dict) else str(getattr(x, "id", None)) for x in rows]
        if set(result_ids) != {str(x["id"]) for x in batch.payload}:
            raise RuntimeError("Rollout IDs differ from fixed panel")
        # Scorer-specific error fields must not be silently treated as capability failures.
        for row in rows:
            error = row.get("error") if isinstance(row, dict) else getattr(row, "error", None)
            if error:
                raise RuntimeError("Rollout reported infrastructure/execution error; inspect protected run files")
        hard, soft = compute_score(rows)
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (hard, soft)):
            raise ValueError("Nonfinite/out-of-range panel score")
        reflection = build_rollout_reflection_context(
            rows, str(out), source_items=batch.payload, adapter=self.adapter,
            failure_k=k, seed=batch.seed, include_hidden_reference=self.rl_cfg.include_hidden_reference)
        score = Score(hard, soft, reflection)
        write_json(cached, asdict(score))
        write_json(out / "score.json", {"hard": hard, "soft": soft, "n": len(rows), "panel_id": panel_id})
        return score
