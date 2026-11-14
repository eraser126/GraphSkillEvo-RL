from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml


@dataclass
class RLConfig:
    seed: int = 42
    device: str = "cpu"
    max_edits: int = 4
    archive_capacity: int = 4
    initialization_pool_size: int = 4
    initial_pool_path: str = ""
    max_generator_calls_per_edit: int = 2
    max_initialization_calls: int = 4
    api_timeout: int = 120
    include_hidden_reference: bool = False
    max_episode_tokens: int = 500000
    max_run_tokens: int = 5000000
    max_run_calls: int = 10000
    cost_unit: float = 1000000.0
    cost_weight: float = 0.01
    invalid_penalty: float = 0.01
    text_hash_dim: int = 256
    max_text_chars: int = 20000
    learning_rate: float = 0.0003
    hidden_sizes: tuple = (128, 128)
    clip_range: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 16
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_update: int = 8
    updates: int = 4
    train_probe_size: int = 15
    reflection_failure_k: int = 5
    dev_check_every_updates: int = 1
    dev_episodes: int = 2
    search_episodes: int = 1
    panel_seed_offset: int = 0

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = raw.get("rl", raw)
        # Accept the nested design-document schema, rejecting unknown keys.
        nested = {"reward", "features", "ppo", "collection", "evaluation"}
        data = {}
        for k, v in raw.items():
            if k in nested:
                if not isinstance(v, dict):
                    raise ValueError(f"rl.{k} must be a mapping")
                for name, value in v.items():
                    if name in data:
                        raise ValueError(f"Duplicate RL option {name}")
                    data[name] = value
            else:
                if k in data:
                    raise ValueError(f"Duplicate RL option {k}")
                data[k] = v
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown RL options: {sorted(unknown)}")
        obj = cls(**data)
        obj.validate()
        return obj

    def validate(self):
        positive = ("max_edits", "archive_capacity", "initialization_pool_size",
                    "max_generator_calls_per_edit", "max_initialization_calls",
                    "api_timeout", "max_episode_tokens", "max_run_tokens", "max_run_calls",
                    "text_hash_dim", "max_text_chars", "update_epochs", "minibatch_size",
                    "episodes_per_update", "updates", "train_probe_size", "dev_episodes", "search_episodes")
        for key in positive:
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.initialization_pool_size > self.archive_capacity:
            raise ValueError("initialization_pool_size exceeds archive capacity")
        if not 0 <= self.reflection_failure_k < self.train_probe_size:
            raise ValueError("Require 0 <= reflection_failure_k < train_probe_size")
        if self.gamma != 1.0:
            raise ValueError("This finite-budget best-archive objective requires gamma=1")
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_range < 1:
            raise ValueError("Invalid GAE/PPO parameters")
        for key in ("learning_rate", "cost_unit", "max_grad_norm"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if any(getattr(self, k) < 0 for k in ("cost_weight", "invalid_penalty", "entropy_coef", "value_coef", "dev_check_every_updates")):
            raise ValueError("Penalties, coefficients and dev frequency must be nonnegative")
        if not self.hidden_sizes or any(int(x) <= 0 for x in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        return self

    def to_dict(self):
        return asdict(self)
