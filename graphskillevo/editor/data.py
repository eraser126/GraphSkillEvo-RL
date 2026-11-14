"""Explicit skill-family and support/reward/query partitions."""
from pathlib import Path
import json
from .schema import ContractError, fields, fingerprint, validate_graph, render_worker


def load_manifest(path):
    path = Path(path).resolve(); data = json.loads(path.read_text(encoding='utf-8'))
    fields(data, {'schema_version', 'items'}, {'schema_version', 'items'})
    if data['schema_version'] != 1 or not isinstance(data['items'], list) or not data['items']:
        raise ContractError('Manifest must contain skill units')
    seen_ids, ownership = set(), {}
    for unit in data['items']:
        fields(unit, {'unit_id','split','skill_ids','family_ids','benchmark','training_allowed','provenance',
                      'graph','support','reward','query'},
               {'unit_id','split','skill_ids','family_ids','benchmark','training_allowed','provenance',
                'graph','support','reward','query'})
        if unit['split'] not in ('train','dev','test','explore') or unit['unit_id'] in seen_ids:
            raise ContractError('Invalid split or repeated unit')
        seen_ids.add(unit['unit_id'])
        if type(unit['training_allowed']) is not bool or not isinstance(unit['provenance'], str) or not unit['provenance']:
            raise ContractError('Explicit training permission/provenance required')
        # Conservative default: benchmark evaluation tasks are not model-training data.
        if 'skillsbench' in unit['benchmark'].lower() and unit['split']=='train':
            raise ContractError('SkillsBench units are reserved for exploration/evaluation, not weight training')
        if unit['split']=='train' and not unit['training_allowed']:
            raise ContractError('Training unit not marked as permitted')
        for kind in ('skill_ids','family_ids'):
            if not isinstance(unit[kind],list) or not unit[kind] or not all(isinstance(x,str) and x for x in unit[kind]):
                raise ContractError('Declare nonempty skill/family group IDs')
            for key in unit[kind]:
                identity = (kind,key)
                if identity in ownership and ownership[identity] != unit['split']:
                    raise ContractError('Skill/family leakage across splits')
                ownership[identity] = unit['split']
        graphpath = (path.parent / unit['graph']).resolve()
        unit['graph'] = validate_graph(json.loads(graphpath.read_text(encoding='utf-8')))
        render_worker(unit['graph'])
        tasks, payloads = set(), set()
        for role in ('support','reward','query'):
            if not isinstance(unit[role],list): raise ContractError('Task panels must be explicit lists')
            if role == 'support' and not unit[role]: raise ContractError('Support panel is required')
            for task in unit[role]:
                fields(task, {'id','payload'}, {'id','payload'})
                if not isinstance(task['id'],str) or not task['id'] or not isinstance(task['payload'],dict):
                    raise ContractError('Invalid task payload')
                tid = task['id']; ph = fingerprint(task['payload'])
                if '/' in tid or '\\' in tid or tid in ('.','..'):
                    raise ContractError('Task IDs must be safe path components')
                if tid in tasks or ph in payloads: raise ContractError('Task/payload overlap between panels')
                tasks.add(tid); payloads.add(ph)
                for identity in [('task',tid),('payload',ph)]:
                    if identity in ownership and ownership[identity] != unit['split']:
                        raise ContractError('Task leakage across skill-unit splits')
                    ownership[identity] = unit['split']
        if unit['split']=='train' and not unit['reward']:
            raise ContractError('Training requires an independent reward panel')
        if unit['split'] in ('dev','test') and not unit['query']:
            raise ContractError('Held-out units require independent query tasks')
    return data
