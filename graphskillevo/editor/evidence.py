"""Immutable observation snapshots and conservative alignment, never invented skips."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .schema import ContractError, fields, fingerprint


@dataclass(frozen=True)
class Trace:
    # JSON string makes nested evidence immutable, including across rechecks.
    encoded: str

    @classmethod
    def create(cls, record):
        fields(record, {'task_id', 'events', 'snapshots', 'success'},
               {'task_id', 'events', 'snapshots', 'success'})
        if not isinstance(record['task_id'], str) or type(record['success']) is not bool:
            raise ContractError('Invalid visible support outcome')
        if not isinstance(record['events'], list) or not isinstance(record['snapshots'], list):
            raise ContractError('Invalid evidence collections')
        previous = -1
        for e in record['events']:
            fields(e, {'index', 'kind', 'node_ids', 'action', 'observation', 'reads', 'writes'},
                   {'index', 'kind', 'node_ids', 'action', 'observation', 'reads', 'writes'})
            if type(e['index']) is not int or e['index'] <= previous:
                raise ContractError('Event indices must strictly increase')
            previous = e['index']
            if e['kind'] not in ('node_start', 'node_end', 'skip', 'step'):
                raise ContractError('Unknown event kind')
            for k in ('node_ids', 'reads', 'writes'):
                if not isinstance(e[k], list) or not all(isinstance(x, str) for x in e[k]):
                    raise ContractError('Event labels must be string lists')
            if not all(isinstance(e[k], str) for k in ('action', 'observation')):
                raise ContractError('Event text must be strings')
        seen = set()
        by_index = {e['index']: e for e in record['events']}
        for s in record['snapshots']:
            fields(s, {'snapshot_id', 'at_event', 'node_id', 'values', 'sha256'},
                   {'snapshot_id', 'at_event', 'node_id', 'values', 'sha256'})
            if not isinstance(s['values'], dict) or s['sha256'] != fingerprint(s['values']):
                raise ContractError('Snapshot hash mismatch')
            event = by_index.get(s['at_event'])
            if not event or event['kind'] != 'node_end' or s['node_id'] not in event['node_ids']:
                raise ContractError('Snapshot must belong to an actual marked node end')
            if s['snapshot_id'] in seen:
                raise ContractError('Duplicate snapshot ID')
            seen.add(s['snapshot_id'])
        return cls(json.dumps(deepcopy(record), sort_keys=True, ensure_ascii=False, allow_nan=False))

    def public(self):
        return json.loads(self.encoded)

    @property
    def id(self):
        return fingerprint(self.public())


def snapshot(node_id, at_event, values):
    # Values must be captured at that moment by the worker/harness, not reconstructed at task end.
    value = deepcopy(values)
    sha = fingerprint(value)
    return dict(snapshot_id=f's_{node_id}_{at_event}_{sha[:12]}', at_event=at_event,
                node_id=node_id, values=value, sha256=sha)


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def put(self, trace):
        path = self.root / (trace.id + '.json')
        if path.exists():
            if path.read_text(encoding='utf-8') != trace.encoded:
                raise ContractError('Immutable snapshot collision')
        else:
            with path.open('x', encoding='utf-8') as f:
                f.write(trace.encoded)
        return path


def align(graph, trace):
    record = trace.public(); events = record['events']
    known = {n['node_id'] for n in graph['nodes']}
    visits, statuses, off_graph, unmapped = [], {}, [], []
    for nid in known:
        starts, found, skipped = [], [], False
        for e in events:
            if nid not in e['node_ids']:
                continue
            if e['kind'] == 'node_start': starts.append(e['index'])
            elif e['kind'] == 'skip': skipped = True
            elif e['kind'] == 'node_end':
                start = starts.pop() if starts else None
                snaps = [s['snapshot_id'] for s in record['snapshots']
                         if s['at_event'] == e['index'] and s['node_id'] == nid]
                status = 'executed' if start is not None and (e['observation'] or snaps) else 'alignment_unknown'
                found.append(dict(node_id=nid, start=start, end=e['index'], status=status, snapshots=snaps))
        statuses[nid] = 'executed' if any(v['status'] == 'executed' for v in found) else (
            'alignment_unknown' if found or starts else 'skipped' if skipped else 'alignment_unknown')
        visits.extend(found)
    for e in events:
        if set(e['node_ids']) - known:
            off_graph.append(e['index'])
        elif e['kind']=='step' and not e['node_ids']:
            unmapped.append(e['index'])
    visits.sort(key=lambda v: v['end'])
    # Edges below indicate OBSERVED artifact use, not causal proof or workflow adjacency.
    writers, dependencies = {}, []
    for e in events:
        for artifact in e['reads']:
            if artifact in writers:
                dependencies.append(dict(source_event=writers[artifact], target_event=e['index'],
                                         artifact=artifact, basis='observed_artifact_reference'))
        for artifact in e['writes']: writers[artifact] = e['index']
    return dict(task_id=record['task_id'], trace_id=trace.id, statuses=statuses, visits=visits,
                off_graph=off_graph, unmapped_events=unmapped, dependencies=dependencies)


def events_from_conversation(conversation):
    """Legacy adapters get markers/text only. No fabricated stage snapshots."""
    events, active = [], set()
    for row in conversation:
        if not isinstance(row, dict): continue
        content = str(row.get('content', row.get('reasoning', '')))
        matches = list(re.finditer(r'\[NODE_(START|END|SKIP) ([A-Za-z][A-Za-z0-9_-]*)\]', content))
        for m in matches:
            events.append(dict(index=len(events), kind={'START':'node_start','END':'node_end','SKIP':'skip'}[m[1]],
                               node_ids=[m[2]], action='', observation='', reads=[], writes=[]))
            if m[1]=='START': active.add(m[2])
            elif m[1]=='END': active.discard(m[2])
        action = str(row.get('action', row.get('cmd', '')))
        observation = str(row.get('observation', row.get('obs', row.get('env_feedback', ''))))
        if action or observation or content:
            events.append(dict(index=len(events), kind='step', node_ids=sorted(active), action=action or content,
                               observation=observation, reads=[], writes=[]))
    return events
