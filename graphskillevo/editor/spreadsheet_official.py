"""Pinned, unmodified upstream evaluator, called outside worker/check editing.

Upstream CLI assumes three original-release cases and input naming. Verified400
uses different paths: only file discovery/aggregation is adapted here.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from .schema import ContractError


OFFICIAL_COMMIT = '49b73a94775fb489063f60ca1865e3a650079a79'
DATA_SHA256 = '10ef893dd29cb13ab97143ea787e68cdc9574a13873ab9a54e50b31dc03fc949'


def verified_case(directory):
    directory = Path(directory)
    candidates = [(p, p.with_name(p.name.replace('_init.xlsx','_golden.xlsx')))
                  for p in sorted(directory.glob('*_init.xlsx'))]
    if (directory/'initial.xlsx').exists():
        candidates.append((directory/'initial.xlsx', directory/'golden.xlsx'))
    # Published archive SHA256 above contains one filename typo. Keep source files
    # unchanged and use this explicit, audited compatibility mapping only.
    if directory.name == '42930' and candidates == [(directory/'1_42930_init.xlsx', directory/'1_42930_golden.xlsx')]:
        candidates = [(directory/'1_42930_init.xlsx', directory/'1_43930_golden.xlsx')]
    if len(candidates) != 1 or not all(p.is_file() for p in candidates[0]):
        raise ContractError('Expected one complete Verified400 case: '+str(directory))
    return candidates[0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialSpreadsheetEvaluator:
    def __init__(self, vendor, lock):
        self.vendor = Path(vendor).resolve()
        locked = json.loads(Path(lock).read_text())
        if locked['commit'] != OFFICIAL_COMMIT:
            raise ContractError('Unexpected official evaluator version')
        for relative, sha in locked['files'].items():
            if hashlib.sha256((self.vendor / relative).read_bytes()).hexdigest() != sha:
                raise ContractError('Official source changed: '+relative)
        self.compare = _load(self.vendor/'evaluation/evaluation.py', 'spreadsheetbench_official').compare_workbooks
        self.recalc = _load(self.vendor/'evaluation/open_spreadsheet.py', 'spreadsheetbench_recalc')

    def grade(self, prediction, golden, instruction_type, answer_position, recalculate=True, soffice=None):
        # The official recalculator overwrites files: exclusively operate on a prediction copy.
        # Gold and raw prediction remain untouched; no details are fed back to the worker.
        with tempfile.TemporaryDirectory(prefix='spreadsheet-official-') as directory:
            copied = Path(directory)/'prediction.xlsx'
            if not Path(prediction).is_file():
                return dict(ok=False, source='official', reason='Missing output file')
            shutil.copy2(prediction, copied)
            if recalculate:
                executable = soffice or self.recalc.find_libreoffice()
                if not executable:
                    raise RuntimeError('Official recalculation backend unavailable')
                if not self.recalc.just_open_libreoffice(str(copied), executable):
                    raise RuntimeError('Official recalculation failed; infrastructure failure, not zero score')
            ok, _ = self.compare(str(golden), str(copied), instruction_type, answer_position)
            return dict(ok=bool(ok), source='official', reason='Official cell-value comparison',
                        official_commit=OFFICIAL_COMMIT, recalculated=bool(recalculate))
