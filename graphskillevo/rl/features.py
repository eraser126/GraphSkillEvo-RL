import hashlib
import re
import numpy as np

OPERATORS = ("mutation_other", "mutation_graph", "crossover_other", "crossover_graph")


def hash_text(text, dim, limit):
    vec = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"\w+", text[:limit].lower()):
        key = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(key[:4], "little") % dim] += 1 if key[4] % 2 else -1
    norm = np.linalg.norm(vec)
    return vec / max(1.0, norm)


def observation_dim(cfg):
    return (cfg.archive_capacity + 1) * (10 + cfg.text_hash_dim) + 10


def encode(env):
    cfg = env.cfg
    parts = []
    for skill in [env.current] + env.archive.slots():
        if skill is None:
            parts.append(np.zeros(10 + cfg.text_hash_dim, dtype=np.float32))
            continue
        score = env.scores[skill.id]
        source = [float(skill.origin == op) for op in OPERATORS]
        structural = [len(skill.content) / cfg.max_text_chars,
                      len(re.findall(r"(?m)^### ", skill.content)) / 100,
                      skill.content.count("**Workflow:**") / 50,
                      skill.age / cfg.max_edits]
        parts.append(np.concatenate(([score.hard, score.soft], structural, source,
                                     hash_text(skill.content, cfg.text_hash_dim, cfg.max_text_chars))))
    spent = env.backend.tokens - env.start_tokens
    global_features = [env.steps / cfg.max_edits,
                       max(0, 1 - spent / cfg.max_episode_tokens),
                       env.scores[env.best.id].hard, env.initial_best_score,
                       env.last_action / env.action_count, env.last_reward,
                       float(env.last_valid), len(env.archive.items) / cfg.archive_capacity,
                       env.backend.calls / cfg.max_run_calls,
                       env.backend.tokens / cfg.max_run_tokens]
    return {"features": np.concatenate(parts + [global_features]).astype(np.float32),
            "mask": env.action_mask().copy()}
