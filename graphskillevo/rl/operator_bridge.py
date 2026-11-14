from graphskillevo.operators import EvolutionOperators, GraphSkillGenerationError
from graphskillevo.graph_skill import assert_graph_skill
from graphskillevo.task_context import build_initial_task_context
from .types import Edit


class OperatorBridge:
    def __init__(self, base_cfg, evolution_cfg, rl_cfg, ledger):
        self.cfg, self.rl_cfg, self.ledger = base_cfg, rl_cfg, ledger
        self.ops = EvolutionOperators(
            env_name=base_cfg["env"], max_completion_tokens=evolution_cfg.operator_max_completion_tokens,
            reasoning_effort=evolution_cfg.operator_reasoning_effort,
            graph_retry_attempts=evolution_cfg.graph_retry_attempts,
            max_initialization_calls=rl_cfg.max_initialization_calls,
            api_retries=1, api_timeout=rl_cfg.api_timeout)

    def initialize(self, base):
        assert_graph_skill(base)
        n = self.rl_cfg.initialization_pool_size - 1
        if n == 0:
            return [base]
        with self.ledger.measure("initialization"):
            skills, _ = self.ops.initialize_population(n=n, base_skill=base,
                                                       task_context=build_initial_task_context(self.cfg))
        pool, seen = [base], {base}
        for content in skills:
            if content not in seen:
                seen.add(content)
                pool.append(content)
        return pool

    def execute(self, operator, current, donor, reflection):
        errors = []
        for attempt in range(1, self.rl_cfg.max_generator_calls_per_edit + 1):
            try:
                with self.ledger.measure("operator", operator=operator, attempt=attempt):
                    if operator.startswith("mutation_"):
                        result = getattr(self.ops, operator)(parent=current, context=reflection or None)
                    else:
                        result = getattr(self.ops, operator)(parent_a=current, parent_b=donor, context=None)
                try:
                    assert_graph_skill(result.new_skill)
                except ValueError:
                    errors.append("SchemaValidationError")
                    continue
                return Edit(result.new_skill, operator, attempts=attempt, errors=tuple(errors))
            except GraphSkillGenerationError as exc:
                # Only generation/schema errors are recoverable as invalid edits.
                errors.append(type(exc).__name__)
        return Edit(None, operator, attempts=attempt, errors=tuple(errors))


class RealBackend:
    def __init__(self, evaluator, operator, base_skill):
        self.evaluator, self.operator, self.base_skill = evaluator, operator, base_skill
        self.ledger = evaluator.ledger
        self._pool = None

    @property
    def tokens(self):
        return self.ledger.tokens

    @property
    def calls(self):
        return self.ledger.calls

    @property
    def panel_id(self):
        return self.evaluator.panel_id

    def reset(self, seed):
        self.ledger.check()
        self.evaluator.set_train_panel(seed)

    def initial_skills(self):
        # Persisted immutable training-side initial pool is shared across episodes/controllers.
        from .evaluator_bridge import write_json
        import json
        from pathlib import Path
        source = self.operator.rl_cfg.initial_pool_path
        pool_path = Path(source) if source else self.evaluator.root / "initial_pool.json"
        if self._pool is None:
            if source and not pool_path.is_file():
                raise ValueError("Explicit shared initial_pool_path must exist")
            if pool_path.exists():
                data = json.loads(pool_path.read_text())
                if data["base"] != self.base_skill or data["size"] != self.operator.rl_cfg.initialization_pool_size:
                    raise ValueError("Initialization pool configuration changed")
                self._pool = data["skills"]
                if not self._pool or len(self._pool) > self.operator.rl_cfg.initialization_pool_size:
                    raise ValueError("Invalid shared initialization pool")
            else:
                self._pool = self.operator.initialize(self.base_skill)
                write_json(pool_path, {"base": self.base_skill, "skills": self._pool,
                                       "size": self.operator.rl_cfg.initialization_pool_size})
        return list(self._pool)

    def evaluate(self, content):
        return self.evaluator.evaluate(content)

    def edit(self, operator, current, donor, reflection):
        return self.operator.execute(operator, current, donor, reflection)
