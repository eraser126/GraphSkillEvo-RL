"""Strict, JSON-only contracts for skills and transactional edits."""
from copy import deepcopy
import hashlib
import json
import re


class ContractError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ContractError(f"Expected keys {sorted(required)}; allowed {sorted(allowed)}")


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Nonempty string required: {label}")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", value):
        raise ContractError("Invalid stable identifier")
    return value


def check_condition(c):
    fields(c, {'check_id', 'applies_when', 'target', 'requirement', 'evidence_needed', 'method', 'rule'},
           {'check_id', 'applies_when', 'target', 'requirement', 'evidence_needed', 'method'})
    identifier(c['check_id'])
    for k in ('applies_when', 'target', 'requirement'):
        text(c[k], k)
    if not isinstance(c['evidence_needed'], list) or not all(isinstance(x, str) and x for x in c['evidence_needed']):
        raise ContractError('evidence_needed must be a list of snapshot keys')
    if c['method'] not in ('rule', 'llm', 'rule_then_llm'):
        raise ContractError('Unknown check method')
    if c['method'] in ('rule', 'rule_then_llm'):
        fields(c.get('rule'), {'kind', 'path', 'value'}, {'kind', 'path'})
        r = c['rule']
        if r['kind'] not in ('exists', 'nonempty', 'type', 'equals', 'spreadsheet') or not isinstance(r['path'], str):
            raise ContractError('Rule must be a bounded JSON predicate, never executable code')
        if r['kind'] in ('type', 'equals') and 'value' not in r:
            raise ContractError('Rule missing value')
        if r['kind'] == 'type' and r['value'] not in ('string','number','object','array','boolean'):
            raise ContractError('Unsupported rule JSON type')
        if r['kind'] == 'spreadsheet':
            from .spreadsheet import validate_spec
            validate_spec(r.get('value'))


def validate_graph(g, max_nodes=64, max_checks=128):
    fields(g, {'schema_version', 'skill_id', 'version', 'global_guidance', 'nodes', 'workflows'},
           {'schema_version', 'skill_id', 'version', 'global_guidance', 'nodes', 'workflows'})
    if g['schema_version'] != 1 or type(g['version']) is not int or g['version'] < 0:
        raise ContractError('Unsupported graph version')
    identifier(g['skill_id'])
    if not isinstance(g['global_guidance'], str):
        raise ContractError('global_guidance must be text')
    if not isinstance(g['nodes'], list) or not 0 < len(g['nodes']) <= max_nodes:
        raise ContractError('Node count outside limits')
    ids, names, check_ids = set(), set(), set()
    for n in g['nodes']:
        fields(n, {'node_id', 'name', 'purpose', 'instruction', 'evidence_expected', 'checks', 'version'},
               {'node_id', 'name', 'purpose', 'instruction', 'evidence_expected', 'checks', 'version'})
        nid = identifier(n['node_id'])
        text(n['name'], 'name')
        if nid in ids or n['name'] in names:
            raise ContractError('Duplicate node ID/name')
        ids.add(nid); names.add(n['name'])
        for k in ('name', 'purpose', 'instruction'):
            text(n[k], k)
        if '\n' in n['name'] or n['name'].startswith('#'):
            raise ContractError('Node names must fit Markdown headings')
        if type(n['version']) is not int or n['version'] < 0 or not isinstance(n['evidence_expected'], list):
            raise ContractError('Invalid node metadata')
        if not all(isinstance(x, str) for x in n['evidence_expected']) or not isinstance(n['checks'], list):
            raise ContractError('Invalid evidence/check list')
        for c in n['checks']:
            check_condition(c)
            if c['check_id'] in check_ids:
                raise ContractError('Check IDs must be unique within a skill')
            check_ids.add(c['check_id'])
    if len(check_ids) > max_checks:
        raise ContractError('Check count outside limits')
    if not isinstance(g['workflows'], list) or not g['workflows']:
        raise ContractError('At least one workflow required')
    wid, used = set(), set()
    for w in g['workflows']:
        fields(w, {'workflow_id', 'name', 'use_when', 'nodes'}, {'workflow_id', 'name', 'use_when', 'nodes'})
        identifier(w['workflow_id'])
        if w['workflow_id'] in wid:
            raise ContractError('Duplicate workflow ID')
        wid.add(w['workflow_id'])
        for k in ('name', 'use_when'):
            text(w[k], k)
            if '\n' in w[k]:
                raise ContractError('Workflow headings/conditions must be one line')
        if not isinstance(w['nodes'], list) or not w['nodes'] or any(x not in ids for x in w['nodes']):
            raise ContractError('Undefined/empty workflow references')
        used.update(w['nodes'])
    if ids != used:
        raise ContractError('Unused nodes; update workflows transactionally')
    # Cycles and repeated visits are intentionally allowed.
    if len(json.dumps(g, ensure_ascii=False)) > 200000:
        raise ContractError('Skill exceeds text budget')
    return g


def render_worker(g, markers=True):
    """Render only execution guidance, NEVER the editable checking channel."""
    validate_graph(g)
    names = {n['node_id']: n['name'] for n in g['nodes']}
    out = ['## Global Guidance', g['global_guidance']]
    if markers:
        out += ['When possible emit [NODE_START id] and [NODE_END id] for the IDs below. '
                'They record stage intent, not a restriction on tool use. Record [NODE_SKIP id] '
                'only for an explicit skip. You may revisit or explore outside the suggested workflow.']
    out += ['## Node Lists']
    for n in g['nodes']:
        out += [f"### {n['name']}", f"Stable node ID: {n['node_id']}" if markers else '', n['instruction']]
    out += ['## Task Graphs']
    for w in g['workflows']:
        out += [f"### {w['name']}", f"**Use when:** {w['use_when']}", '**Workflow:**']
        out += [f'{i}. {names[n]}' for i, n in enumerate(w['nodes'], 1)]
    content = '\n\n'.join(out) + '\n'
    from graphskillevo.graph_skill import assert_graph_skill
    try:
        assert_graph_skill(content)
    except ValueError as exc:
        raise ContractError('Rendered skill violates upstream graph document contract') from exc
    return content


def import_markdown(content, skill_id):
    from graphskillevo.graph_skill import split_graph_skill, assert_graph_skill
    assert_graph_skill(content)
    p = split_graph_skill(content)
    def blocks(s):
        matches = list(re.finditer(r'^###\s+(.+?)\s*$', s, re.M))
        return [(m[1], s[m.end():matches[i+1].start() if i+1 < len(matches) else len(s)].strip())
                for i, m in enumerate(matches)]
    nodes, lookup = [], {}
    for name, body in blocks(p.node_lists):
        nid = 'n_' + hashlib.sha256(name.encode()).hexdigest()[:12]
        lookup[name] = nid
        nodes.append(dict(node_id=nid, name=name, purpose=name, instruction=body or name,
                          evidence_expected=[], checks=[], version=0))
    workflows = []
    for i, (name, body) in enumerate(blocks(p.task_graphs)):
        condition = re.search(r'^\*\*Use when:\*\*\s*(.+)$', body, re.M)[1]
        refs = re.findall(r'^\d+\.\s+(.+?)\s*$', body, re.M)
        workflows.append(dict(workflow_id=f'w_{i}', name=name, use_when=condition,
                              nodes=[lookup[r] for r in refs]))
    guidance = p.global_guidance.split('## Global Guidance', 1)[1].strip()
    return validate_graph(dict(schema_version=1, skill_id=skill_id, version=0,
                               global_guidance=guidance, nodes=nodes, workflows=workflows))


def parse_patch(response):
    """Strict JSON only: no regex recovery that could silently change the action."""
    try:
        value = json.loads(response)
    except (ValueError, TypeError) as e:
        raise ContractError('Editor must output a single JSON patch') from e
    fields(value, {'base_version', 'operations', 'evidence_ids', 'rationale'},
           {'base_version', 'operations', 'evidence_ids', 'rationale'})
    if not isinstance(value['operations'], list) or not isinstance(value['evidence_ids'], list):
        raise ContractError('Patch operations/evidence_ids must be lists')
    if not all(isinstance(x, str) for x in value['evidence_ids']) or not isinstance(value['rationale'], str):
        raise ContractError('Invalid patch metadata')
    return value


def apply_patch(g, patch, phase):
    """Copy, apply, validate the entire transaction; never partially mutate input."""
    validate_graph(g)
    patch = parse_patch(json.dumps(patch, allow_nan=False))
    if type(patch['base_version']) is not int or patch['base_version'] != g['version']:
        raise ContractError('Stale patch version')
    if phase not in ('checks', 'graph'):
        raise ContractError('Invalid edit phase')
    result = deepcopy(g)
    for op in patch['operations']:
        fields(op, {'op', 'node_id', 'payload', 'check_id', 'workflow_id'}, {'op'})
        kind = op['op']
        check_ops = ('add_check', 'update_check', 'delete_check')
        graph_ops = ('update_node', 'insert_node', 'delete_node', 'set_workflow', 'delete_workflow', 'set_guidance')
        if kind not in (check_ops if phase == 'checks' else graph_ops):
            raise ContractError('Operation not permitted in this phase; use empty operations for NOOP')
        nodes = {n['node_id']: n for n in result['nodes']}
        if kind in check_ops + ('update_node', 'delete_node'):
            if op.get('node_id') not in nodes:
                raise ContractError('Unknown node target')
            n = nodes[op['node_id']]
        if kind in check_ops:
            checks = {c['check_id']: c for c in n['checks']}
            if kind == 'delete_check':
                if op.get('check_id') not in checks:
                    raise ContractError('Unknown check target')
                n['checks'] = [c for c in n['checks'] if c['check_id'] != op['check_id']]
            else:
                payload = deepcopy(op.get('payload')); check_condition(payload)
                cid = payload['check_id']
                if (kind == 'add_check' and cid in checks) or (kind == 'update_check' and cid not in checks):
                    raise ContractError('Check add/update target mismatch')
                n['checks'] = [c for c in n['checks'] if c['check_id'] != cid] + [payload]
            n['version'] += 1
        elif kind == 'update_node':
            p = op.get('payload')
            fields(p, {'name', 'purpose', 'instruction', 'evidence_expected'})
            if not p:
                raise ContractError('Empty node update')
            n.update(deepcopy(p)); n['version'] += 1
        elif kind == 'insert_node':
            p = deepcopy(op.get('payload'))
            if not isinstance(p, dict) or p.get('checks') != []:
                raise ContractError('Insert a node with empty checks; edit checks in the next cycle')
            result['nodes'].append(p)
        elif kind == 'delete_node':
            result['nodes'] = [x for x in result['nodes'] if x['node_id'] != n['node_id']]
        elif kind == 'set_guidance':
            result['global_guidance'] = text(op.get('payload'), 'global guidance')
        elif kind == 'set_workflow':
            p = deepcopy(op.get('payload'))
            if not isinstance(p, dict):
                raise ContractError('Workflow payload required')
            result['workflows'] = [w for w in result['workflows'] if w['workflow_id'] != p.get('workflow_id')] + [p]
        elif kind == 'delete_workflow':
            if op.get('workflow_id') not in {w['workflow_id'] for w in result['workflows']}:
                raise ContractError('Unknown workflow')
            result['workflows'] = [w for w in result['workflows'] if w['workflow_id'] != op['workflow_id']]
    if patch['operations']:
        result['version'] += 1
    validate_graph(result)
    render_worker(result)
    return result
