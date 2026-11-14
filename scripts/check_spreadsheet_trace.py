"""Recheck captured node evidence only; never runs an agent or grades a task."""
import argparse
import json
from pathlib import Path
from graphskillevo.editor.evidence import Trace
from graphskillevo.editor.schema import validate_graph
from graphskillevo.editor.verification import CheckEngine


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--graph', required=True)
    p.add_argument('--trace', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    graph = validate_graph(json.loads(Path(args.graph).read_text()))
    trace = Trace.create(json.loads(Path(args.trace).read_text()))
    reports = CheckEngine().run(graph, [trace])
    with Path(args.out).open('x') as out:
        json.dump(reports, out, indent=2)
    print(json.dumps({'checks': len(reports[0]['checks']), 'agent_called': False,
                      'task_graded': False, 'retry_triggered': False}))


if __name__ == '__main__': main()
