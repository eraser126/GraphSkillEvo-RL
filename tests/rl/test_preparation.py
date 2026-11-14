"""Non-training checks for PPO plumbing, checkpoint restore and the API guard."""
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import torch

from graphskillevo.rl.buffer import RolloutBuffer
from graphskillevo.rl.config import RLConfig
from graphskillevo.rl.policy import ActorCritic
from graphskillevo.rl.ppo import ppo_update
from graphskillevo.rl.trainer import RLTrainer, seed_all


def test_ppo_objective_with_no_weight_updates():
    cfg = RLConfig(update_epochs=1, minibatch_size=2)
    policy = ActorCritic(3, 4, [8])
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    buffer = RolloutBuffer()
    for reward in (0.0, 0.4):
        obs = {"features": np.zeros(3, dtype=np.float32), "mask": np.array([True, True, False, False])}
        action, logp, value = policy.sample(obs)
        buffer.add(obs, action, logp, value, reward, True, 0)
        buffer.finish_episode(1, 0.95)
    class NoUpdateOptimizer:
        calls = 0
        def zero_grad(self):
            policy.zero_grad()
        def step(self):
            self.calls += 1  # Intentionally no parameter update: preparation, not training.
    optimizer = NoUpdateOptimizer()
    metrics = ppo_update(policy, optimizer, buffer, cfg)
    assert optimizer.calls == 1
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())


def test_episode_boundary_checkpoint_restores_rng_and_policy(tmp_path):
    cfg = RLConfig()
    backend = SimpleNamespace(tokens=10, calls=1, evaluator=SimpleNamespace(fingerprint="unit"))
    seed_all(42)
    trainer = RLTrainer(backend, cfg, tmp_path)
    trainer.update, trainer.episode_counter = 2, 7
    trainer.checkpoint("unit.pt")
    expected = torch.rand(5)
    second = RLTrainer(backend, cfg, tmp_path)
    second.load(tmp_path / "unit.pt", resume=True)
    assert second.update == 2 and second.episode_counter == 7
    assert torch.equal(torch.rand(5), expected)
    assert all(torch.equal(v, second.policy.state_dict()[k]) for k, v in trainer.policy.state_dict().items())


def test_real_run_without_opt_in_exits_before_api_or_output_creation(tmp_path):
    root = tmp_path / "must_not_exist"
    result = subprocess.run([sys.executable, "-m", "graphskillevo.run_rl",
                             "--base-config", "configs/searchqa/default.yaml",
                             "--rl-config", "configs/rl/ppo_smoke.yaml",
                             "--out-root", str(root)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "No API calls permitted" in result.stderr
    assert not root.exists()
