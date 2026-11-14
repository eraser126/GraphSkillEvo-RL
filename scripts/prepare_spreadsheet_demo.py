"""Select an offline demo subset inside existing splits; never execute tasks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re


# Instruction-based screening tags, not official labels or verified task semantics.
PATTERNS = {
    'text_processing': r'\b(text|string|substring|concatenat\w*|split|extract\w*|character\w*|trim|replace|LEFT\(|RIGHT\(|MID\()',
    'lookup_matching': r'\b(lookup\w*|vlookup|hlookup|xlookup|match\w*|index|offset|indirect|join\w*|correspond\w*|nth)\b',
    'aggregation_calculation': r'\b(sum\w*|total\w*|average\w*|count\w*|calculat\w*|maximum|minimum|aggregate\w*|subtotal\w*|percent\w*|round\w*|floor|ceiling|multiply|subtract\w*|divid\w*)\b',
    'conditional_logic': r'\b(if|ifs|sumif\w*|countif\w*|condition\w*|criteria|criterion|depend\w*|based on)\b',
    'date_time': r'\b(date\w*|time\w*|month\w*|year\w*|calendar|weekday\w*|duration\w*|hour\w*)\b',
    'cleaning_missing_duplicates': r'\b(blank\w*|empty|missing|duplicate\w*|unique|clean\w*|error\w*|invalid|#N/A|#VALUE)\b',
    'filter_sort': r'\b(filter\w*|sort\w*|rank\w*|ascending|descending|order by|delete\w*|remove\w*|exclud\w*)\b',
    'table_reshape': r'\b(transpos\w*|pivot\w*|unpivot\w*|rearrang\w*|restructur\w*|reshape\w*|consolidat\w*|insert\w*|merge\w*|stack\w*|append\w*|copy\w*|move\w*)\b',
    'cross_sheet': r'\b(multiple|different|other|separate|each|all|across|between|another|two|three)\s+(?:\w+\s+){0,3}(?:sheets|worksheets|workbooks)\b|\b(?:sheets|worksheets|workbooks)\b|(?:[A-Za-z0-9_]+|\x27[^\x27]+\x27)!\$?[A-Z]+\$?\d+',
    'formatting': r'\b(format\w*|color\w*|colour\w*|font\w*|bold|align\w*|decimal\w*|border\w*|highlight\w*|number format)\b',
}


def tags(task):
    text=task['payload']['instruction']
    return sorted(name for name,pattern in PATTERNS.items() if re.search(pattern,text,re.I))


def features(task):
    return {'operation:'+x for x in tags(task)}|{'official:'+task['payload']['instruction_type']}


def select(rows,size,seed):
    """Cover rare instruction tags twice when feasible, then fill official-type quotas."""
    if not 0<size<=len(rows):raise ValueError('Subset size outside original split')
    rng=random.Random(seed);pool=list(rows);rng.shuffle(pool)
    availability=Counter(f for row in rows for f in features(row))
    covered=Counter();chosen=[]
    official=Counter(row['payload']['instruction_type'] for row in rows)
    while len(chosen)<size:
        def score(row):
            fs=features(row)
            # Positive weight for each still-under-covered operation/type.
            coverage=sum(1/availability[f] for f in fs if covered[f]<min(2,availability[f]))
            typ='official:'+row['payload']['instruction_type']
            deficit=max(0,official[row['payload']['instruction_type']]*size/len(rows)-covered[typ])
            return (coverage,deficit)
        row=max(pool,key=score);pool.remove(row);chosen.append(row);covered.update(features(row))
    return chosen


def coverage(rows):
    return dict(tasks=len(rows),official_types=dict(Counter(t['payload']['instruction_type'] for t in rows)),
                operation_tags=dict(sorted(Counter(x for t in rows for x in tags(t)).items())),
                no_operation_tag=[t['id'] for t in rows if not tags(t)])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--val-size',type=int,default=16);p.add_argument('--test-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261109)
    a=p.parse_args();source=Path(a.manifest).resolve();out=Path(a.out).resolve()
    report_path=out.with_suffix('.selection.json')
    if out.exists() or report_path.exists():p.error('Output exists; choose new paths')
    data=json.loads(source.read_text(encoding='utf-8'))
    original=data['splits']
    selected={'train':original['train'],
              'val':select(original['val'],a.val_size,a.seed),
              'test':select(original['test'],a.test_size,a.seed+1)}
    import os
    data['graph']=os.path.relpath((source.parent/data['graph']).resolve(),out.parent)
    data['splits']=selected
    data['training_allowed']=False
    data['provenance']+=f' Demo subset seed={a.seed}: preserve all train IDs, select val/test within original splits using heuristic multi-label instruction coverage; no execution outcomes used.'
    report=dict(seed=a.seed,source=str(source),training_allowed=False,
                limitations=['Tags are heuristic and multi-label, not official operation categories.',
                             'Coverage is instruction keyword coverage, not verified semantic coverage or difficulty balance.',
                             'Original split membership is preserved; source-task similarity leakage has not been audited.'],
                patterns=PATTERNS,original={s:coverage(rows) for s,rows in original.items()},
                selected={s:coverage(rows) for s,rows in selected.items()},
                annotations={s:[dict(id=t['id'],official_type=t['payload']['instruction_type'],
                                    operation_tags=tags(t),instruction=t['payload']['instruction']) for t in rows]
                             for s,rows in selected.items()})
    for split,rows in selected.items():
        assert {t['id'] for t in rows}<={t['id'] for t in original[split]}
        report['selected'][split]['missing_available_tags']=sorted(set(report['original'][split]['operation_tags'])-set(report['selected'][split]['operation_tags']))
    from graphskillevo.editor.batch import load_batch_manifest
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    load_batch_manifest(out)
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report['selected'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
