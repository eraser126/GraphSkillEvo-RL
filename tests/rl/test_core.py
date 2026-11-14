"""Offline preparation checks only: no API calls and no training experiment."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from graphskillevo.graph_skill import assert_graph_skill
from graphskillevo.rl.archive import Archive
from graphskillevo.rl.buffer import RolloutBuffer
from graphskillevo.rl.config import RLConfig
from graphskillevo.rl.env import SkillEditEnv
from graphskillevo.rl.features import hash_text, observation_dim
from graphskillevo.rl.policy import ActorCritic
from graphskillevo.rl.types import Skill, Score, Edit


def skill(value):
    return (f"# Toy\n\n## Global Guidance\n{value}\n\n## Node Lists\n"
            "### operate\nFollow the instruction.\n\n## Task Graphs\n"
            "### default\n**Use when:** always\n**Workflow:**\n1. operate\n")


class MockBackend:
    def __init__(self, pool=None, invalid=False):
        self.tokens = self.calls = 0
        self.pool = pool or [skill("base"), skill("initial-good")]
        self.invalid = invalid
        self.calls_log = []

    def reset(self, seed):
        self.panel_id = f"train-{seed}"

    def initial_skills(self):
        return self.pool

    def evaluate(self, content):
        self.tokens += 10
        self.calls += 1
        hard = 0.8 if "repaired" in content else 0.4 if "initial-good" in content else 0.1
        return Score(hard, hard, "private train-side reflection")

    def edit(self, operator, current, donor, reflection):
        self.tokens += 5
        self.calls += 1
        self.calls_log.append((operator, donor, reflection))
        return Edit(None if self.invalid else skill("repaired"), operator, attempts=1)


def config(**kw):
    return RLConfig(archive_capacity=2, initialization_pool_size=2, **kw).validate()


def test_original_markdown_contract():
    assert_graph_skill(skill("base"))


def test_action_map_and_self_cross_mask():
    env = SkillEditEnv(MockBackend(), config())
    obs, _ = env.reset(1)
    assert len(obs["features"]) == observation_dim(env.cfg)
    assert obs["mask"].tolist() == [True, True, False, True, False, True, True]
    with pytest.raises(ValueError):
        env.step(2)
    env.step(5)
    assert env.backend.calls_log[-1][0] == "crossover_graph"
    assert "initial-good" in env.backend.calls_log[-1][1]


def test_fifo_pins_initial_and_deduplicates():
    archive = Archive(3)
    versions = [Skill(skill(str(i))) for i in range(5)]
    for version in versions:
        archive.add(version)
    assert [x.id for x in archive.items] == [versions[0].id, versions[3].id, versions[4].id]
    archive.add(versions[4])
    assert len(archive.items) == 3


def test_stop_gets_no_free_initialization_lift():
    env = SkillEditEnv(MockBackend(), config())
    env.reset(1)
    _, reward, terminal, _, info = env.step(6)
    assert reward == 0.0 and terminal
    assert info["initial_best_score"] == 0.4


def test_reward_matches_best_archive_and_costs():
    env = SkillEditEnv(MockBackend(), config(max_edits=2))
    env.reset(1)
    _, first, terminal, _, _ = env.step(1)
    assert not terminal
    _, second, terminal, _, _ = env.step(6)
    assert terminal
    assert first + second == pytest.approx(0.8 - 0.4 - 0.01 * 15 / 1000000)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_edit_self_transition_and_finite_horizon():
    env = SkillEditEnv(MockBackend(invalid=True), config(max_edits=1))
    env.reset(1)
    old = env.current.id
    _, reward, terminal, _, info = env.step(1)
    assert terminal and info["invalid"] and env.current.id == old and reward < 0


def test_legal_regression_is_retained_for_exploration():
    backend = MockBackend(pool=[skill("initial-good")])
    backend.edit = lambda *args: Edit(skill("base"), args[0], attempts=1)
    env = SkillEditEnv(backend, config())
    env.reset(1)
    env.step(1)
    assert env.scores[env.current.id].hard == 0.1
    assert env.scores[env.best.id].hard == 0.4


def test_infrastructure_failure_propagates_without_reward():
    backend = MockBackend()
    env = SkillEditEnv(backend, config())
    env.reset(1)
    backend.edit = lambda *a: (_ for _ in ()).throw(ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        env.step(0)
    assert env.steps == 0


def test_stable_features_and_snapshot_mask():
    assert np.array_equal(hash_text("abc ABC", 32, 100), hash_text("abc ABC", 32, 100))
    env = SkillEditEnv(MockBackend(), config())
    obs, _ = env.reset(1)
    buf = RolloutBuffer()
    buf.add(obs, 1, -1, 0, 1, True, 0)
    saved = buf.rows[0]["mask"].copy()
    obs["mask"][:] = False
    assert np.array_equal(buf.rows[0]["mask"], saved)


def test_gae_terminal_and_collection_bootstrap():
    obs = {"features": np.zeros(3), "mask": np.array([True, False])}
    b = RolloutBuffer()
    b.add(obs, 0, 0, 0.2, 1.0, True, 99)
    b.finish_episode(1, 0.95)
    assert b.rows[0]["return"] == pytest.approx(1.0)
    b.add(obs, 0, 0, 0.2, 0.1, False, 0.8)
    b.finish_episode(1, 0.95)
    assert b.rows[1]["return"] == pytest.approx(0.9)


def test_policy_masks_and_finite_backward_only():
    torch.manual_seed(42)
    p = ActorCritic(3, 4, [8])
    f = torch.zeros(2, 3)
    masks = torch.tensor([[True, False, False, False], [False, True, True, False]])
    dist, values = p.distribution_value(f, masks)
    assert dist.probs[0, 0] == 1
    assert dist.probs[1, 0] == 0
    loss = -dist.log_prob(torch.tensor([0, 1])).mean() + values.square().mean()
    loss.backward()  # Unit gradient check only; no optimizer/training run.
    assert all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None)
    with pytest.raises(ValueError):
        p.distribution_value(f, torch.zeros_like(masks))


def test_config_rejects_unknown_and_bad_reflection(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("rl:\n  wat: 1\n")
    with pytest.raises(ValueError):
        RLConfig.load(p)
    with pytest.raises(ValueError):
        RLConfig(train_probe_size=1).validate()


def test_bounded_original_initialization(monkeypatch):
    import graphskillevo.model as model
    from graphskillevo.operators import EvolutionOperators, GraphSkillGenerationError
    calls = []
    def broken(**kw):
        calls.append(kw)
        return "not JSON", {}
    monkeypatch.setattr(model, "chat_optimizer", broken)
    op = EvolutionOperators(env_name="searchqa", max_completion_tokens=10,
                            reasoning_effort="medium", max_initialization_calls=2, api_retries=1)
    with pytest.raises(GraphSkillGenerationError):
        op.initialize_population(n=1, base_skill=skill("base"))
    assert len(calls) == 2 and all(x["retries"] == 1 for x in calls)


def test_ledger_survives_restart_and_accounts_failures(tmp_path, monkeypatch):
    from graphskillevo.rl.evaluator_bridge import Ledger
    import graphskillevo.model as model
    counter = {"calls": 0, "total_tokens": 0}
    monkeypatch.setattr(model, "get_token_summary", lambda: {"_total": counter.copy()})
    ledger = Ledger(tmp_path, config())
    with pytest.raises(ConnectionError):
        with ledger.measure("unit"):
            counter.update(calls=1, total_tokens=7)
            raise ConnectionError()
    restored = Ledger(tmp_path, config())
    assert restored.tokens == 7 and restored.calls == 1


def test_fixed_panel_cache_isolation_and_test_guard(tmp_path, monkeypatch):
    from graphskillevo.datasets.base import BatchSpec
    from graphskillevo.rl.evaluator_bridge import Ledger, EvaluatorBridge
    import graphskillevo.model as model
    monkeypatch.setattr(model, "get_token_summary", lambda: {"_total": {}})
    class Loader:
        def build_train_batch(self, batch_size, seed, **kw):
            return BatchSpec("train", "train", seed, 2, [{"id": str(seed)}, {"id": str(seed + 1)}])
    class Adapter:
        calls = 0
        def build_env_from_batch(self, batch, **kw):
            return deepcopy(batch.payload)
        def rollout(self, env, content, out, **kw):
            self.calls += 1
            return [{"id": x["id"], "hard": 1, "soft": 1} for x in env]
    cfg = config()
    adapter = Adapter()
    bridge = EvaluatorBridge(adapter, Loader(), tmp_path, {"model": "test"}, cfg, Ledger(tmp_path, cfg))
    bridge.set_train_panel(1)
    original_id = bridge.panel_id
    bridge.evaluate(skill("base"))
    bridge.evaluate(skill("base"))
    assert adapter.calls == 1
    bridge.set_train_panel(2)
    bridge.evaluate(skill("base"))
    assert adapter.calls == 2 and bridge.panel_id != original_id
    with pytest.raises(ValueError):
        bridge.make_panel(3, "valid_unseen")


def test_original_ea_modules_import_without_using_rl():
    import graphskillevo.run
    from graphskillevo.selection import child_operator_sequence
    assert len(child_operator_sequence(4)) == 4


def test_real_bridge_bounded_invalid_and_api_exception(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from graphskillevo.config import EvolutionConfig
    from graphskillevo.operators import GraphSkillGenerationError
    from graphskillevo.rl.operator_bridge import OperatorBridge
    class Ledger:
        def measure(self, *args, **kw):
            return nullcontext()
    bridge = OperatorBridge({"env": "searchqa"}, EvolutionConfig(), config(), Ledger())
    calls = []
    def invalid(**kw):
        calls.append(1)
        raise GraphSkillGenerationError("bad")
    monkeypatch.setattr(bridge.ops, "mutation_graph", invalid)
    assert bridge.execute("mutation_graph", skill("a"), None, "").content is None
    assert len(calls) == 2
    monkeypatch.setattr(bridge.ops, "mutation_graph", lambda **kw: (_ for _ in ()).throw(ConnectionError()))
    with pytest.raises(ConnectionError):
        bridge.execute("mutation_graph", skill("a"), None, "")
