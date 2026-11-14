"""Evidence-led candidate ranking, explicitly not causal root-cause proof."""


def diagnose(graph, reports):
    candidates, contradictions, deviations = [], [], []
    workflow_edges = set()
    for w in graph['workflows']:
        workflow_edges.update(zip(w['nodes'], w['nodes'][1:]))
    for r in reports:
        alignment = r['alignment']
        checks = r['checks']; failed = [x for x in checks if x['status'] == 'fail']
        decisive = [x for x in checks if x['status'] in ('pass', 'fail')]
        if not r['success'] and not failed:
            contradictions.append(dict(task_id=r['task_id'], kind='failure_without_detected_check_failure',
                                       all_executed_checks_pass=bool(decisive) and all(x['status']=='pass' for x in decisive),
                                       has_unknown=any(x['status']=='unknown' for x in checks),
                                       uncovered_nodes=r['uncovered_nodes']))
        if r['success'] and failed:
            contradictions.append(dict(task_id=r['task_id'], kind='success_despite_check_failure',
                                       check_ids=[x['check_id'] for x in failed],
                                       alternatives=['overstrict_or_inapplicable', 'judge_error', 'agent_recovery']))
        for c in failed:
            candidates.append(dict(task_id=r['task_id'], node_id=c['node_id'], reason='observed_check_failure',
                                   check_id=c['check_id'], snapshot_id=c['snapshot_id'], causal_proof=False))
            end = c['at_event']
            # Follow actual artifact-reference chain, not mere workflow predecessor order.
            frontier = [end]
            for v in alignment['visits']:
                if v['node_id']==c['node_id'] and v['end']==end and v['start'] is not None:
                    frontier.extend(range(v['start'],v['end']+1))
            seen = set()
            while frontier:
                current = frontier.pop()
                for dep in alignment['dependencies']:
                    if dep['target_event'] != current or dep['source_event'] in seen: continue
                    seen.add(dep['source_event']); frontier.append(dep['source_event'])
                    for v in alignment['visits']:
                        if v['start'] is not None and v['start'] <= dep['source_event'] <= v['end']:
                            frontier.extend(i for i in range(v['start'],v['end']+1) if i not in seen)
                            candidates.append(dict(task_id=r['task_id'], node_id=v['node_id'],
                                                   reason='upstream_artifact_candidate', artifact=dep['artifact'],
                                                   causal_proof=False))
        order = [v['node_id'] for v in alignment['visits'] if v['status']=='executed']
        deviations.append(dict(task_id=r['task_id'], success=r['success'],
                               explicit_skips=[n for n,s in alignment['statuses'].items() if s=='skipped'],
                               unknown_nodes=[n for n,s in alignment['statuses'].items() if s=='alignment_unknown'],
                               revisits=sorted({n for n in order if order.count(n)>1}),
                               off_graph_events=alignment['off_graph'],
                               unmapped_events=alignment['unmapped_events'],
                               observed_nonworkflow_transitions=[list(e) for e in zip(order,order[1:])
                                                                if e not in workflow_edges]))
    return dict(candidates=candidates, contradictions=contradictions, deviations=deviations,
                interpretation='Candidates for repair; frequency, ordering and checks do not prove causality.')
