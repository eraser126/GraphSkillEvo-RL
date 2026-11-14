# Evidence grounded LLM skill editor

Prepared against the project design dated 2026-11-02. This path is separate from both the original EA (`graphskillevo.run`) and the earlier operator-selecting PPO (`graphskillevo.run_rl`). It trains a causal LLM's edit tokens with GRPO, not the worker agent or a small action-classification MLP.

## What is implemented

- Original Markdown import and compatible rendering; stable node IDs, conditional ordered workflows and versioned check conditions. No forced DAG. Cycles/revisits are legal.
- Atomic, phase-specific JSON patches: node insertion/deletion/instruction changes, workflow changes and check add/update/delete. Invalid transactions roll back completely.
- Worker rendering excludes check conditions. Markers provide guidance only, not a graph-constrained runtime.
- Immutable node-time snapshots, a harness-side recorder, conservative alignment, off-graph/skip/revisit reports and artifact-reference upstream candidates. Missing evidence remains unknown; no final-state reconstruction.
- Fixed bounded rule checker and optional fixed LLM checker, four check statuses and success/check contradictions. Checks run post hoc and never block the worker.
- Same LLM first edits checks, then receives rechecked evidence and edits the graph. Reward/query objects are not inputs to the edit function.
- Native PyTorch completion-token GRPO, group-relative advantages, clipped ratios, equal episode/turn weighting, dropout disabled, finite-gradient checks and optional LoRA. No KL penalty in this minimal version. Generation uses temperature 1 with no top-k/p warping so recorded and recomputed likelihoods match.
- Fresh on-policy groups, fixed reward panels, shared baseline, independent dev checkpoint selection, per-update model/optimizer/RNG snapshots and interruption metadata. Unfinished groups are discarded. Automatic resume is not exposed yet: keep cost logs and explicitly plan recovery before restarting a real run; never treat a new output root as continuation with a reset budget.
- Frozen one-candidate adaptation; separately invoked query evaluation against a hashed adapted artifact, no query-based skill selection.
- Original adapter bridge and a normalized external-harness command bridge. Ledger, manifest split validation and explicit runtime/training opt-ins.

## Safe offline checks

```bash
python -m graphskillevo.run_editor --config configs/editor/minimal.yaml --manifest examples/editor/manifest.json
python -m pytest tests/rl tests/editor -q
python -m graphskillevo.run_editor --config configs/editor/minimal.yaml --manifest examples/editor/manifest.json --mode replay --unit-id contract_fixture --replay-evidence examples/editor/evidence.json --replay-responses examples/editor/responses.json --out-root runs/editor_offline_replay
python -m scripts.editor_skill --input ckpt/gpt-5.4-nano/spreadsheetbench/best_skill.md --skill-id spreadsheet_skill --output /tmp/spreadsheet_editor_graph.json
```

Replay uses handwritten synthetic evidence and scripted patches. It is a contract check, not a toy training run, benchmark evaluation, trained editor or performance result. Output directories must be new. Default preflight does not even initialize adapters or models.

## Manifest contract

See `examples/editor/manifest.json`. Each unit declares split (`train`, `dev`, `test`, `explore`), skill/family IDs, provenance, an explicit training permission, graph path and support/reward/query lists. Task IDs and canonical payload hashes cannot overlap between panels. Skill/family/task identities cannot cross splits. File paths and aliases require a separate data audit; the validator cannot discover semantic near-duplicates.

The example is marked explore and forbidden for weight training. SkillsBench units are never accepted as train units by this implementation. Use separately audited permitted training material. Keep official graders/oracle/hidden answers outside worker/edit evidence. Only support records are projected into editor prompts; the private reward panel is evaluated after candidate generation and final query is a separate command.

## Worker and evidence integration

`OriginalBackend` calls the original `EnvAdapter.build_env_from_batch` and `rollout`; it uses official hard scores without building hidden-reference reflections. If the harness has written `predictions/<task_id>/editor_trace.json`, it loads the strict evidence contract. Otherwise it reads normal conversation logs, with empty snapshots. Such runs are explicitly evidence-limited; node checks return unknown rather than inventing intermediate values.

For richer traces, call `EvidenceRecorder.start`, `event`, and `end` in the worker's trusted harness hooks at the actual boundary. `end` can capture explicitly named worker-visible JSON/text files, checked to remain inside that visible workspace. Capture only allowed observations; the recorder is not a security sandbox and cannot infer whether arbitrary caller-supplied values contain hidden answers. Keep the recorder/output directory on the harness side. Call `finish(success)` only after grading and export `trace.public()`.

The alignment is deliberately conservative: a node start/end claim plus observable evidence is stronger than a bare label, but does not prove the node was correctly performed. Partial/missing markers remain uncertain. Artifact dependencies are observed read/write references; workflow predecessors alone are not causal ground truth.

## SkillsBench and external harness

SkillsBench/BenchFlow is not installed or secretly launched by these modules. Native BenchFlow does NOT directly accept our request schema. The `command` backend is the adapter boundary for a version-pinned trusted wrapper around the official harness.

Configure `runner_argv` as a list containing literal `{request}` and `{response}` placeholders. The bridge creates a fresh job directory and invokes it with `shell=False`. The request contains `schema_version`, panel role, explicit task handles, rendered worker skill, fresh-environment requirement and evidence contract. It contains no editable checks. A wrapper must:

1. Resolve each task handle against a fixed SkillsBench checkout and fresh container.
2. Replace only the chosen skill artifact, preserving other skills and the original official grader.
3. Keep oracle, tests and evaluator output unavailable to the worker.
4. Capture observable stage evidence through trusted hooks, run the official verifier outside the worker and return only normalized rows.
5. Report internal calls/tokens and fail loudly on infrastructure errors; no inferred zero scores.

Response JSON is `{ "schema_version": 1, "results": [...] }`. Each row has `id`, `score` in [0,1], `calls`, `tokens`; support rows additionally have `trace` matching `Trace.create`. A nonzero runner exit, timeout, missing response, duplicate/missing IDs, nonfinite score or malformed trace aborts the run. Do not pass CLI JSON through a shell or reuse response files.

The concrete wrapper is `python -m graphskillevo.editor.skillsbench`. It pins a clean SkillsBench checkout commit, creates an external skills overlay, replaces only the selected SKILL.md body while preserving frontmatter and other package resources, and constructs the documented `bench eval run` command. Without `--allow-run` it only writes execution plans and NEVER writes a fake scored response. With explicit execution it also checks the exact `bench --version` output and required CLI flags before invoking the harness. It refuses ambiguous multiple result files and missing trusted usage.

`configs/editor/skillsbench.template.yaml` shows the command configuration. Its explicit replacement strings and environment paths are integration placeholders; they are not provisioned resources and cannot be used for a real run unchanged. Worker task payloads must contain only `task_dir` (relative to the fixed checkout) and `target_skill` (relative package folder under environment/skills).

`--tokens-field` and `--calls-field` select nested trusted result fields for the actual pinned BenchFlow version; defaults are illustrative and may need adjustment. No missing field is silently treated as zero. Stage snapshots require trusted harness hooks; ordinary legacy conversation-only output remains evidence-limited.

This wrapper is unit-tested with filesystem fixtures, NOT verified by a real SkillsBench run. BenchFlow currently requires a separate Python 3.12+ tool environment, whereas this editor uses the original Python 3.10 environment. No BenchFlow/container execution is performed by preparation. Pin/install its environment and test the chosen real task before claiming end-to-end readiness. The benchmark data, independent adaptation examples and leakage audit are still required.

## Future real execution only after authorization

`adapt` needs `--allow-api`, a local model path OR `--api-editor`, a real unit and a fresh output root. The API editor is an untrained fixed baseline and cannot enter GRPO. Local model loading is offline-only, no remote-code trust and no automatic downloads. Optional LoRA checkpoints require their original local base directory.

`train` additionally needs `--allow-training`, a permitted train manifest and a LOCAL editor model. It never calls the query panels of test units; dev query panels may select a checkpoint and are labeled selection-only. `evaluate` requires an exported `--artifact` and checks graph and support provenance before querying the official score. It never loads an editor model.

The optional stack in `requirements-editor.txt` is a torch-2.1-compatible baseline, not a universal environment for every recent model. Check model architecture support and memory on the eventual GPU environment before use. The current no-GPU machine is for preparation only.

## Remaining limits

No real model training, model download, paid API request or real benchmark run was performed during preparation. Rule-only checks cannot judge arbitrary semantic requirements; configure a fixed judge for those or they return unknown. Native language model generation, optional LoRA and external harness execution need subsequent integration tests with the chosen real model and task.

Ledger limits are operation-boundary limits, not provider-enforced hard spending caps. External harness wrappers must report internal usage, API providers may omit usage for failed calls, and initial adapter retries may be hidden. No automatic version ranking by checker pass rate is implemented. The reward belongs to the two-turn episode; no precise per-node/check causal credit is claimed.

Reference APIs: [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation), [SkillsBench](https://github.com/benchflow-ai/skillsbench), [Skill-Alpha training](https://github.com/ejhshen/skill-alpha/blob/main/docs/TRAINING.md).
