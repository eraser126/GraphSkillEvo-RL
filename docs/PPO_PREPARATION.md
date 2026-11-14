# PPO skill-search implementation / preparation status

Prepared on 2026-10-31 on the original GraphSkillEvo checkout.
Upstream base: `722a8a913b05dc9d1b4078f0bec604cde4dc7529`.
Branch: `feature/ppo-skill-search`.

## Scope

The original EA entry point remains `python -m graphskillevo.run`.
The new entry point is `python -m graphskillevo.run_rl`.
PPO updates a small actor–critic, not the execution or editor LLM.
No RL population selection, graph backtracking or verifier learning is included.
The four original editing prompts and scope enforcement are reused.

**Preparation only. No PPO training, toy learning run, API rollout, or paid smoke
experiment is authorized/started during setup.** Unit tests use mocks and one
gradient check with no optimizer step. They are not evidence of research gains.

## Safe offline checks (no API)

```bash
cd /root/autodl-tmp/GraphSkillEvo
python -m graphskillevo.run_rl --help
python -m graphskillevo.run_rl --base-config configs/searchqa/default.yaml --rl-config configs/rl/ppo_minimal.yaml --out-root runs/preflight --check-config
python -m pytest tests/rl -q
```

Without `--allow-api`, any real train/search invocation is rejected before model
configuration or adapter setup. `--check-config` validates config and model names,
but does not validate raw datasets, credentials, endpoint reachability or rollout
compatibility. Configurable search defaults to deterministic actions for PPO;
random/round-robin baselines use the same bounded archive and action masks.

## Future commands — do not run until authorized

```bash
# Requires data materialization, endpoint/key configuration, and permission to spend API budget.
python -m graphskillevo.run_rl --base-config configs/searchqa/default.yaml --rl-config configs/rl/ppo_smoke.yaml --mode train --out-root runs/ppo_smoke --allow-api

# Freeze a trusted trained checkpoint; feedback remains on train, selection on dev.
python -m graphskillevo.run_rl --base-config configs/searchqa/default.yaml --rl-config configs/rl/ppo_smoke.yaml --mode search --controller ppo --checkpoint runs/ppo_smoke/policy_best.pt --out-root runs/ppo_search --allow-api

# Mechanism baselines (same environment/actions, no PPO updates).
python -m graphskillevo.run_rl --base-config configs/searchqa/default.yaml --rl-config configs/rl/ppo_smoke.yaml --mode search --controller random --out-root runs/random_search --allow-api
python -m graphskillevo.run_rl --base-config configs/searchqa/default.yaml --rl-config configs/rl/ppo_smoke.yaml --mode search --controller round_robin --out-root runs/round_robin_search --allow-api
```

Checkpoint files contain Python state; load only trusted checkpoints. Use the same
training output directory and `--resume --checkpoint .../policy_last.pt` for an
episode-boundary continuation. `policy_interrupted.pt` resumes from the next episode
after discarding the unfinished collection, with the actual spent ledger retained.
No test feedback is exposed by the RL bridge. Evaluate the exported `best_skill.md`
separately using the original evaluation script; never use test to select artifacts.

## Design choices

* Current version advances on every legal edit, including temporary regressions.
* Initial reference is pinned; other historical versions are FIFO, never fitness-selected.
* Invalid output is a bounded self-transition. Transport/auth/scoring failures abort,
  not a made-up zero reward.
* Episode return is terminal best-minus-initial-pool-best, minus actual recorded
  token cost and invalid penalties. Reset costs are included in the total ledger.
* Fixed train batches are rebuilt from a deep copy; IDs are checked against results.
  Unsupported adapters without explicit IDs fail fast rather than silently re-sample.
* Caches include full panel fingerprint, configuration/model identity, seed and
  feedback settings; cache hits are not independent repeated measurements.
* Numerical feature scaling and stable hashed text use no fitted normalization;
  checkpoint restores optimizer and Python/NumPy/Torch RNG at episode boundaries.
* `include_hidden_reference` defaults false and must be matched across baselines.

For matched mechanism comparisons, set `rl.initial_pool_path` in each run's config
to the same pre-generated `initial_pool.json` (with `base`, `skills`, `size` fields).
Without this option each fresh output directory generates its own initial pool;
such independent initializations must not be presented as a paired comparison.

## Budget limitations / checks needed before real experiments

Budgets are admission checks at operation boundaries, not a hard provider-level
billing cap: a complete task batch or initialization call can cross the token
threshold. Configure small batches and provider-side quotas. SDK/target adapter
transport retries may still occur internally; initialization and graph-generation
calls are bounded separately. The upstream token tracker records reported usage;
unreported failed requests may not provide billable token counts. Infrastructure
failures are separately logged; verify actual bills/rate limits with the provider.

Raw benchmark payloads are not installed by this patch. The repository contains
mostly split manifests. SearchQA/SpreadsheetBench need materialized examples;
ALFWorld needs its optional dependency and raw environment. Do not interpret an
offline preflight as an end-to-end data/API validation. Seeds do not force a hosted
LLM to be deterministic; repeat real evaluation when later authorized.

This small MLP/hashed-text observation is partially observable and does not prove
optimality. PPO-vs-random and PPO-vs-original-EA answer different experimental
questions. Include initialization, dev selection, invalid retries and PPO training
costs when reporting results.
