import math
import numpy as np
from graphskillevo.graph_skill import assert_graph_skill
from .archive import Archive
from .features import encode
from .types import Skill


class SkillEditEnv:
    """A search episode; all reward/feedback comes from a frozen train panel."""
    def __init__(self, backend, cfg):
        self.backend, self.cfg = backend, cfg
        self.action_count = 3 + 2 * cfg.archive_capacity
        self.done = True

    def _score(self, skill):
        if skill.id not in self.scores:
            result = self.backend.evaluate(skill.content)
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (result.hard, result.soft)):
                raise ValueError("Scorer returned invalid values")
            self.scores[skill.id] = result
        return self.scores[skill.id]

    def reset(self, seed):
        self.backend.reset(seed)
        self.start_tokens = self.backend.tokens
        self.steps, self.last_action, self.last_reward = 0, 0, 0.0
        self.last_valid, self.done = True, False
        self.scores = {}
        self.archive = Archive(self.cfg.archive_capacity)
        pool = self.backend.initial_skills()
        if not pool or len(pool) > self.cfg.archive_capacity:
            raise ValueError("Empty or oversized initialization pool")
        for content in pool:
            assert_graph_skill(content)
            skill = Skill(content)
            self._score(skill)
            self.archive.add(skill)
        self.current = self.archive.items[0]
        self.best = max(self.archive.items, key=lambda x: self.scores[x.id].hard)
        self.initial_best_score = self.scores[self.best.id].hard
        if self.backend.tokens - self.start_tokens >= self.cfg.max_episode_tokens:
            raise RuntimeError("Initialization already exhausted episode token budget")
        return encode(self), {"initial_ids": [s.id for s in self.archive.items],
                              "panel_id": self.backend.panel_id,
                              "initial_best": self.initial_best_score}

    def action_mask(self):
        mask = np.zeros(self.action_count, dtype=bool)
        mask[:2] = True
        mask[-1] = True
        for i, donor in enumerate(self.archive.slots()):
            if donor is not None and donor.id != self.current.id:
                mask[2 + i] = mask[2 + self.cfg.archive_capacity + i] = True
        return mask

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() required after terminal state")
        action = int(action)
        if not 0 <= action < self.action_count or not self.action_mask()[action]:
            raise ValueError("Masked or out-of-range action")
        cfg = self.cfg
        before = self.current.id
        donor_slots = [x.id if x else None for x in self.archive.slots()]
        tokens_before, calls_before = self.backend.tokens, self.backend.calls
        invalid, operator, donor_id, attempts, errors = False, "stop", None, 0, ()
        if action != self.action_count - 1:
            k = cfg.archive_capacity
            if action < 2:
                operator = ("mutation_other", "mutation_graph")[action]
                donor = None
            else:
                graph = action >= 2 + k
                slot = action - 2 - (k if graph else 0)
                donor = self.archive.slots()[slot]
                operator = "crossover_graph" if graph else "crossover_other"
            edit = self.backend.edit(operator, self.current.content,
                                     donor.content if donor else None,
                                     self.scores[self.current.id].reflection)
            attempts, errors = edit.attempts, edit.errors
            donor_id = donor.id if donor else None
            if edit.content is None:
                invalid = True
            else:
                try:
                    assert_graph_skill(edit.content)
                except ValueError:
                    invalid = True
                if not invalid:
                    new = Skill(edit.content, operator, self.steps + 1)
                    self._score(new)  # Infrastructure errors abort; never become zero reward.
                    self.current = new  # Accept short-term regressions for exploration.
                    self.archive.add(new)
                    if self.scores[new.id].hard > self.scores[self.best.id].hard:
                        self.best = new
        self.steps += 1
        spent = self.backend.tokens - tokens_before
        reward = -cfg.cost_weight * spent / cfg.cost_unit - cfg.invalid_penalty * invalid
        self.done = (action == self.action_count - 1 or self.steps >= cfg.max_edits
                     or self.backend.tokens - self.start_tokens >= cfg.max_episode_tokens)
        if self.done:
            reward += self.scores[self.best.id].hard - self.initial_best_score
        self.last_action, self.last_reward, self.last_valid = action, reward, not invalid
        info = {"step": self.steps, "panel_id": self.backend.panel_id,
                "before_id": before, "after_id": self.current.id, "best_id": self.best.id,
                "archive_slots_before": donor_slots, "operator": operator, "donor_id": donor_id,
                "invalid": invalid, "attempts": attempts, "errors": list(errors),
                "reward": reward, "tokens": spent, "calls": self.backend.calls - calls_before,
                "current_score": self.scores[self.current.id].hard,
                "best_score": self.scores[self.best.id].hard,
                "initial_best_score": self.initial_best_score, "terminal": self.done}
        return encode(self), float(reward), self.done, False, info
