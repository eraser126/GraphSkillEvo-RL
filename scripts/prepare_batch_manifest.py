"""Build task metadata, never trajectories, SFT examples or model labels."""
import argparse,json,re
from pathlib import Path


def family(text):
    # Transparent initial heuristic; labels are editable metadata, not learned ground truth.
    for name,pattern in [('multi_sheet',r'\b(sheet|sheets|worksheet|worksheets)\b'),
                         ('dates',r'\b(date|dates|time|month|year)\b'),
                         ('formulas',r'\b(formula|formulas|calculate|sum|total|average)\b'),
                         ('filter_sort',r'\b(sort|filter|duplicate|duplicates|remove)\b')]:
        if re.search(pattern,text,re.I):return name
    return 'general'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default='.')
    p.add_argument('--out',required=True);p.add_argument('--permit-training-data',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve()
    rows=json.loads((root/'data/spreadsheetbench_verified_400/dataset.json').read_text())
    by_id={str(x['id']):x for x in rows};splits={}
    for split in ['train','val','test']:
        ids=json.loads((root/'data/spreadsheetbench_id_split'/split/'items.json').read_text())
        tasks=[]
        for item in ids:
            tid=str(item['id'] if isinstance(item,dict) else item);row=by_id[tid]
            payload={k:row[k] for k in ['id','instruction','instruction_type','answer_position','answer_sheet'] if k in row}
            tasks.append(dict(id=tid,payload=payload,family=family(row['instruction'])))
        splits[split]=tasks
    import os
    data=dict(schema_version=2,benchmark='SpreadsheetBench Verified 400',training_allowed=a.permit_training_data,
              provenance='Official archive; preserve repository 80/40/280 ID splits. Family labels are instruction-keyword heuristics; no execution feedback used.',
              graph=os.path.relpath(root/'examples/editor/spreadsheet_verified400_initial/initial_graph.json',out.parent),splits=splits)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False)
    from graphskillevo.editor.batch import load_batch_manifest
    load_batch_manifest(out)
    print(json.dumps(dict(splits={k:len(v) for k,v in splits.items()},training_allowed=a.permit_training_data,agent_calls=0)))


if __name__=='__main__':main()
