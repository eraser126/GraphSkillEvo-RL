"""Spreadsheet worker bridge: original skill, checkpoints, independent scoring.

Prepared for later explicit runtime authorization. Importing never runs tasks.
"""
import json
from pathlib import Path
import shutil
import uuid

from .schema import ContractError, render_worker
from .spreadsheet import SpreadsheetEvidenceSession
from .spreadsheet_official import OfficialSpreadsheetEvaluator, verified_case
from .verification import CheckEngine


class SpreadsheetBackend:
    def __init__(self, data_root, vendor, lock, root, ledger, max_turns=30, soffice=None, exec_timeout=600):
        self.data_root = Path(data_root).resolve()
        self.grader = OfficialSpreadsheetEvaluator(vendor, lock)
        self.root, self.ledger = Path(root).resolve(), ledger
        self.max_turns, self.soffice = max_turns, soffice
        self.exec_timeout=exec_timeout

    def run(self, graph, tasks, phase):
        if phase not in ('support', 'reward', 'query'):
            raise ValueError('Unknown panel role')
        from graphskillevo.envs.spreadsheetbench.react_agent import run_react
        self.root.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in tasks:
            payload = task['payload']
            tid = str(task['id'])
            if not tid or Path(tid).name != tid or tid in ('.', '..'):
                raise ContractError('Invalid dataset task ID')
            case_dir = (self.data_root/'spreadsheet'/tid).resolve()
            if not case_dir.is_relative_to(self.data_root): raise ContractError('Invalid dataset path')
            # Use actual Verified400 names; refuse to silently select an arbitrary case.
            input_file, gold = verified_case(case_dir)
            if not input_file.is_file() or not gold.is_file():
                raise RuntimeError('Verified400 initial/golden files missing')
            output = self.root/f'{phase}_{uuid.uuid4().hex}'
            work = output/'workspace'; work.mkdir(parents=True, exist_ok=False)
            shutil.copy2(input_file, work/'input.xlsx')
            session = SpreadsheetEvidenceSession(graph, tid, work)
            session.observe('TASK_CONTEXT',json.dumps(dict(instruction=payload['instruction'],
                            instruction_type=payload.get('instruction_type',''),
                            output_contract=dict(input='input.xlsx',output='output.xlsx')),ensure_ascii=False))
            def execute():
                return run_react(instruction=payload['instruction'], input_path=str(work/'input.xlsx'),
                                 output_path=str(work/'output.xlsx'), work_dir=str(work),
                                 instruction_type=payload.get('instruction_type', ''),
                                 answer_position=payload.get('answer_position', ''),
                                 skill_content=render_worker(graph), max_turns=self.max_turns,
                                 evidence_session=session,exec_timeout=self.exec_timeout)
            raw = self.ledger.call('spreadsheet_worker_'+phase, execute)
            # No task evaluator or gold-comparison feedback is passed into run_react.
            answer_position = payload['answer_position']
            if payload.get('answer_sheet') and '!' not in answer_position:
                answer_position = ','.join(f"'{payload['answer_sheet']}'!{r}" for r in answer_position.split(','))
            result = self.grader.grade(work/'output.xlsx', gold, payload.get('instruction_type', ''),
                                       answer_position, soffice=self.soffice)
            trace = session.finish(result['ok'])
            reports = CheckEngine().run(graph, [trace])
            for name, value in [('editor_trace.json', trace.public()), ('node_checks.json', reports),
                                ('official_result.json', result), ('conversation.json', raw['conversation'])]:
                (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False))
            row = dict(id=tid, score=float(result['ok']))
            if phase == 'support': row['trace'] = trace.public()
            rows.append(row)
        from .backends import validate_rows
        return validate_rows(tasks, rows, phase)
