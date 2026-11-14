"""Apply small backwards-compatible extensions on the pinned upstream checkout."""
from pathlib import Path

path = Path("graphskillevo/operators.py")
text = path.read_text()
if "max_initialization_calls: int | None" not in text:
    old = "        graph_retry_attempts: int = 3,\n    ) -> None:"
    new = ("        graph_retry_attempts: int = 3,\n"
           "        max_initialization_calls: int | None = None,\n"
           "        api_retries: int = 3,\n"
           "        api_timeout: int | None = None,\n    ) -> None:")
    assert text.count(old) == 1, "Upstream constructor changed; inspect before patching"
    text = text.replace(old, new)
    anchor = "        self.graph_retry_attempts = max(1, int(graph_retry_attempts))"
    text = text.replace(anchor, anchor + "\n        self.max_initialization_calls = max_initialization_calls\n"
                        "        self.api_retries = max(1, int(api_retries))\n"
                        "        self.api_timeout = api_timeout")
    anchor = "        while len(accepted) < n:\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor +
        "            if self.max_initialization_calls is not None and len(attempts) >= self.max_initialization_calls:\n"
        "                raise GraphSkillGenerationError(\n"
        "                    'Initialization generation-call budget exhausted',\n"
        "                    parsed={'accepted_count': len(accepted), 'attempt_count': len(attempts)},\n"
        "                )\n")
    assert text.count("retries=3,") == 2
    text = text.replace("retries=3,", "retries=self.api_retries,")
    text = text.replace("reasoning_effort=self.reasoning_effort,\n",
                        "reasoning_effort=self.reasoning_effort,\n" + " " * 12 + "timeout=self.api_timeout,\n")
    # Match indentation of the two calls without modifying their prompts/semantics.
    text = text.replace("                reasoning_effort=self.reasoning_effort,\n            timeout=",
                        "                reasoning_effort=self.reasoning_effort,\n                timeout=")
    path.write_text(text)

path = Path("pyproject.toml")
text = path.read_text()
if 'rl = ["torch>=2.1,<3"]' not in text:
    text = text.replace('[project.optional-dependencies]\n',
                        '[project.optional-dependencies]\nrl = ["torch>=2.1,<3"]\nrl_dev = ["pytest>=8,<10"]\n')
    path.write_text(text)
print("Backwards-compatible upstream extensions applied")
