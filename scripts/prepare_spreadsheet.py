"""Preparation only: validate downloaded data and compile the ORIGINAL graph."""
import argparse
import hashlib
import json
from pathlib import Path

from graphskillevo.editor.spreadsheet import initial_graph
from graphskillevo.editor.schema import render_worker


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', required=True, help='New directory; refuses to overwrite')
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = root/'graphskillevo/initial_skills/spreadsheetbench/init_skill.md'
    content = source.read_text()
    graph = initial_graph(content)
    (out/'initial_graph.json').write_text(json.dumps(graph, indent=2))
    (out/'worker_skill.md').write_text(render_worker(graph))
    provenance = dict(source=str(source.relative_to(root)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      nodes=len(graph['nodes']), workflows=graph['workflows'], source_kind='upstream_initial',
                      node_mapping=[dict(node_id=n['node_id'], name=n['name']) for n in graph['nodes']],
                      training_started=False, model_calls=0, benchmark_runs=0)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    manifest = dict(schema_version=1,items=[dict(
        unit_id='spreadsheet_preflight_fixture',split='explore',
        skill_ids=[graph['skill_id']],family_ids=['spreadsheet_initial'],
        benchmark='synthetic_preflight_only',training_allowed=False,
        provenance='Synthetic placeholder to validate configuration; contains no benchmark trajectories or labels.',
        graph='initial_graph.json',support=[dict(id='synthetic_placeholder',payload={'fixture':True})],
        reward=[],query=[])])
    (out/'preflight_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__': main()
