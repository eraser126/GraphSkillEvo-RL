"""Idempotent addition of the optional LLM-editor dependency extra."""
from pathlib import Path

root=Path(__file__).resolve().parent
path=root/'pyproject.toml'
content=path.read_text(encoding='utf-8')
if '\neditor = [' not in content:
    anchor='[project.optional-dependencies]\n'
    if anchor not in content: raise RuntimeError('Unexpected upstream pyproject layout')
    extra='editor = ["torch>=2.1,<3", "transformers==4.44.2", "tokenizers==0.19.1", "accelerate==0.34.2", "peft==0.13.2", "safetensors==0.4.5"]\n'
    path.write_text(content.replace(anchor,anchor+extra,1),encoding='utf-8')
readme=root/'README.md'
body=readme.read_text(encoding='utf-8')
heading='## Evidence-grounded LLM editor preparation'
if heading not in body:
    body+='\n\n'+heading+'\n\n'
    body+='The independent `python -m graphskillevo.run_editor` entry implements the 2026-11-02 design: '
    body+='post-hoc node checks, two-turn LLM editing and completion-token GRPO. '
    body+='The original EA and earlier operator-selection PPO entries remain available. '
    body+='See [preparation and integration limits](docs/EDITOR_PREPARATION.md).\n\n'
    body+='Offline preflight (no models, APIs or training):\n\n```bash\n'
    body+='python -m graphskillevo.run_editor --config configs/editor/minimal.yaml --manifest examples/editor/manifest.json\n```\n'
    readme.write_text(body,encoding='utf-8')
