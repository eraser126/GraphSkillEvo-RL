import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.rows = []

    def add(self, obs, action, logp, value, reward, terminated, next_value):
        self.rows.append({"features": obs["features"].copy(), "mask": obs["mask"].copy(),
                          "action": action, "logp": logp, "value": value, "reward": reward,
                          "terminated": terminated, "next_value": next_value})

    def finish_episode(self, gamma, lam):
        """Call at every episode boundary (including collection truncation)."""
        indices = [i for i, r in enumerate(self.rows) if "advantage" not in r]
        gae = 0.0
        for i in reversed(indices):
            r = self.rows[i]
            continuation = 0.0 if r["terminated"] else 1.0
            delta = r["reward"] + gamma * continuation * r["next_value"] - r["value"]
            gae = delta + gamma * lam * continuation * gae
            r["advantage"], r["return"] = gae, gae + r["value"]

    def arrays(self):
        if not self.rows or any("advantage" not in r for r in self.rows):
            raise ValueError("Empty or unfinished on-policy rollout")
        return {k: np.asarray([r[k] for r in self.rows]) for k in
                ("features", "mask", "action", "logp", "advantage", "return")}
