"""Data integrity audit; no benchmark execution, scoring or learning data export."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from graphskillevo.editor.spreadsheet_official import DATA_SHA256, verified_case


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.data_root).resolve()
    sha = hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
    assert sha == DATA_SHA256, 'Unexpected dataset archive'
    data = json.loads((root/'dataset.json').read_text())
    ids = [str(x['id']) for x in data]
    assert len(ids) == len(set(ids)) == 400
    files, mappings = {}, []
    for row in data:
        tid = str(row['id'])
        assert tid not in ('.','..') and Path(tid).name == tid
        directory = root/'spreadsheet'/tid
        input_file, gold = verified_case(directory)
        mappings.append(dict(id=tid,input=str(input_file.relative_to(root)),gold=str(gold.relative_to(root))))
        for path in (input_file,gold):
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None, path
                assert '[Content_Types].xml' in z.namelist()
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    sets = {}
    for split in ['train','val','test']:
        rows = json.loads((Path(args.splits)/split/'items.json').read_text())
        sets[split] = {str(x['id']) if isinstance(x,dict) else str(x) for x in rows}
        assert sets[split] <= set(ids), 'Unknown split IDs'
    assert not any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')])
    assert set.union(*sets.values()) == set(ids), 'Incomplete original split coverage'
    report = dict(tasks=len(ids), workbook_files=len(files), archive_sha256=sha,
                  split_sizes={k:len(v) for k,v in sets.items()}, split_overlap=False,
                  benchmark_runs=0, sft_records_generated=0, case_mappings=mappings, files=files)
    with Path(args.out).open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','case_mappings')},indent=2))


if __name__ == '__main__': main()
