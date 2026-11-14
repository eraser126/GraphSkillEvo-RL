"""Explicit opt-in API search/training, with episode-boundary checkpoints."""
from pathlib import Path
import json
import random
import subprocess
import numpy as np
import torch

from .buffer import RolloutBuffer
from .env import SkillEditEnv
from .features import observation_dim
from .policy import ActorCritic
from .ppo import ppo_update
from .evaluator_bridge import append_json, write_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class RLTrainer:
    def __init__(self, backend, cfg, root):
        self.backend, self.cfg, self.root = backend, cfg, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.env = SkillEditEnv(backend, cfg)
        self.policy = ActorCritic(observation_dim(cfg), self.env.action_count, cfg.hidden_sizes).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.update, self.episode_counter, self.best_dev = 0, 0, float("-inf")

    def checkpoint(self, name):
        data = {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rl_config": self.cfg.to_dict(), "update": self.update,
                "episode_counter": self.episode_counter, "best_dev": self.best_dev,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "ledger_tokens": self.backend.tokens, "ledger_calls": self.backend.calls,
                "base_fingerprint": self.backend.evaluator.fingerprint}
        temp = self.root / f"{name}.tmp"
        torch.save(data, temp)
        temp.replace(self.root / name)

    def load(self, path, resume=False):
        # Load only checkpoints produced by this trusted checkout.
        data = torch.load(path, map_location=self.cfg.device, weights_only=False)
        old = data["rl_config"]
        for key in ("archive_capacity", "text_hash_dim", "max_text_chars", "hidden_sizes"):
            if old[key] != self.cfg.to_dict()[key]:
                raise ValueError(f"Checkpoint incompatible: {key}")
        self.policy.load_state_dict(data["policy"])
        if resume:
            allowed = {"updates", "max_run_tokens", "max_run_calls", "device"}
            if any(old[k] != v for k, v in self.cfg.to_dict().items() if k not in allowed):
                raise ValueError("Resume must retain the training configuration")
            if data["base_fingerprint"] != self.backend.evaluator.fingerprint:
                raise ValueError("Resume model/data configuration fingerprint changed")
            if self.backend.tokens < data["ledger_tokens"] or self.backend.calls < data["ledger_calls"]:
                raise ValueError("Cost ledger missing; refusing to reset spent budget")
            self.optimizer.load_state_dict(data["optimizer"])
            self.update, self.episode_counter = data["update"], data["episode_counter"]
            self.best_dev = data["best_dev"]
            random.setstate(data["python_rng"])
            np.random.set_state(data["numpy_rng"])
            torch.set_rng_state(data["torch_rng"].cpu())
            if torch.cuda.is_available() and data["cuda_rng"]:
                torch.cuda.set_rng_state_all(data["cuda_rng"])

    def episode(self, seed, controller="ppo", buffer=None, deterministic=False, prefix="search"):
        episode_id = self.episode_counter
        self.episode_counter += 1
        directory = self.root / "episodes" / f"{episode_id:06d}_{prefix}"
        directory.mkdir(parents=True, exist_ok=True)
        start_tokens, start_calls = self.backend.tokens, self.backend.calls
        obs, reset_info = self.env.reset(seed)
        write_json(directory / "reset.json", {**reset_info, "seed": seed,
                                             "reset_tokens": self.backend.tokens - start_tokens})
        total_reward = 0.0
        for step in range(self.cfg.max_edits):
            if controller == "ppo":
                action, logp, value = self.policy.sample(obs, deterministic=deterministic)
            else:
                legal = np.flatnonzero(obs["mask"])
                edits = legal[legal != self.env.action_count - 1]
                if controller == "random":
                    action = int(np.random.choice(legal))
                elif controller == "round_robin":
                    # Fixed operator family schedule, donor sampled uniformly among legal slots.
                    k = self.cfg.archive_capacity
                    groups = [[0], [1], list(range(2, k + 2)), list(range(k + 2, 2 * k + 2))]
                    available = [a for a in groups[step % 4] if obs["mask"][a]]
                    action = int(np.random.choice(available or list(edits)))
                else:
                    raise ValueError("Unknown controller")
                logp, value = 0.0, 0.0
            next_obs, reward, terminal, truncated, info = self.env.step(action)
            total_reward += reward
            if buffer is not None:
                next_value = 0.0 if terminal else self.policy.sample(next_obs, deterministic=True)[2]
                buffer.add(obs, action, logp, value, reward, terminal, next_value)
            append_json(directory / "edits.jsonl", {**info, "action": action,
                        "mask": obs["mask"].tolist(), "old_logp": logp, "value": value,
                        "policy_update": self.update})
            obs = next_obs
            if terminal or truncated:
                break
        if buffer is not None:
            buffer.finish_episode(self.cfg.gamma, self.cfg.gae_lambda)
        (directory / "best_skill.md").write_text(self.env.best.content, encoding="utf-8")
        summary = {"episode_id": episode_id, "seed": seed, "controller": controller,
                   "reward": total_reward, "best_train_score": self.env.scores[self.env.best.id].hard,
                   "best_id": self.env.best.id, "initial_best": self.env.initial_best_score,
                   "tokens": self.backend.tokens - start_tokens, "calls": self.backend.calls - start_calls}
        write_json(directory / "summary.json", summary)
        return self.env.best.content, summary

    def dev_evaluate(self):
        # Search feedback remains train-only. Holdout score never enters the edit environment.
        evaluator = self.backend.evaluator
        panel, panel_id = evaluator.make_panel(self.cfg.seed + 2000000, "valid_seen")
        scores = []
        for i in range(self.cfg.dev_episodes):
            skill, _ = self.episode(self.cfg.seed + 1000000 + i, deterministic=True, prefix="dev")
            scores.append(evaluator.evaluate(skill, panel, panel_id).hard)
        mean = float(np.mean(scores))
        append_json(self.root / "dev.jsonl", {"update": self.update, "mean": mean, "scores": scores})
        return mean

    def train(self):
        try:
            return self._train()
        except Exception as exc:
            # Discard unfinished on-policy data, retain consumed budget/RNG/next episode ID.
            self.checkpoint("policy_interrupted.pt")
            write_json(self.root / "interrupted.json", {"error_type": type(exc).__name__,
                       "update": self.update, "next_episode": self.episode_counter,
                       "tokens": self.backend.tokens, "calls": self.backend.calls})
            raise

    def _train(self):
        for update in range(self.update, self.cfg.updates):
            buffer = RolloutBuffer()
            # No partial-batch PPO update after any API fault. Costs remain durably logged.
            for _ in range(self.cfg.episodes_per_update):
                seed = self.cfg.seed + self.cfg.panel_seed_offset + self.episode_counter
                self.episode(seed, buffer=buffer, prefix="train")
            metrics = ppo_update(self.policy, self.optimizer, buffer, self.cfg)
            self.update = update + 1
            append_json(self.root / "metrics.jsonl", {"update": self.update, **metrics,
                                                     "tokens": self.backend.tokens, "calls": self.backend.calls})
            if self.cfg.dev_check_every_updates and self.update % self.cfg.dev_check_every_updates == 0:
                dev = self.dev_evaluate()
                if dev > self.best_dev:
                    self.best_dev = dev
                    self.checkpoint("policy_best.pt")
            self.checkpoint("policy_last.pt")
            print(json.dumps({"update": self.update, **metrics}), flush=True)
        return {"updates": self.update, "episodes": self.episode_counter,
                "tokens": self.backend.tokens, "calls": self.backend.calls}

    def search(self, controller):
        evaluator = self.backend.evaluator
        dev_panel, dev_id = evaluator.make_panel(self.cfg.seed + 2000000, "valid_seen")
        candidates = []
        best_content, best_score = None, float("-inf")
        for i in range(self.cfg.search_episodes):
            content, record = self.episode(self.cfg.seed + self.cfg.panel_seed_offset + i,
                                          controller=controller, deterministic=True)
            score = evaluator.evaluate(content, dev_panel, dev_id).hard
            candidates.append({**record, "dev_score": score})
            if score > best_score:
                best_content, best_score = content, score
        (self.root / "best_skill.md").write_text(best_content, encoding="utf-8")
        return {"selected_by": "valid_seen", "dev_score": best_score, "candidates": candidates,
                "tokens": self.backend.tokens, "calls": self.backend.calls,
                "test_evaluated": False}


def build_trainer(base_cfg, evolution_cfg, rl_cfg, root):
    from graphskillevo.adapters import get_adapter
    from graphskillevo.model_setup import configure_models
    from graphskillevo.model import reset_token_tracker
    from .evaluator_bridge import Ledger, EvaluatorBridge
    from .operator_bridge import OperatorBridge, RealBackend
    seed_all(rl_cfg.seed)
    configure_models(base_cfg)
    reset_token_tracker()
    adapter = get_adapter(base_cfg)
    adapter.setup(base_cfg)
    ledger = Ledger(root, rl_cfg)
    evaluator = EvaluatorBridge(adapter, adapter.get_dataloader(), root, base_cfg, rl_cfg, ledger)
    operators = OperatorBridge(base_cfg, evolution_cfg, rl_cfg, ledger)
    base_skill = Path(base_cfg["skill_init"]).read_text(encoding="utf-8")
    return RLTrainer(RealBackend(evaluator, operators, base_skill), rl_cfg, root)
