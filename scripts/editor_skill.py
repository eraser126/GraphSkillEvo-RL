"""Offline convert existing GraphSkillEvo skill Markdown to editable JSON."""
import argparse
from pathlib import Path
from graphskillevo.editor.schema import import_markdown,render_worker
from graphskillevo.editor.accounting import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--skill-id',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    target=Path(args.output)
    if target.exists(): p.error('Output exists; preserve original graph/version')
    g=import_markdown(Path(args.input).read_text(encoding='utf-8'),args.skill_id)
    render_worker(g);write_json(target,g)
    print({'status':'converted_offline','nodes':len(g['nodes']),'api_called':False})


if __name__=='__main__': main()
