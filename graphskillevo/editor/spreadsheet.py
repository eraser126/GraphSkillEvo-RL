"""Worker-visible spreadsheet evidence and fixed deterministic check interfaces.

No gold workbooks, evaluator output, model calls, retries or arbitrary predicates.
Facts describe the captured stage, not the semantic correctness of the solution.
"""
import ast
import datetime
import hashlib
import math
from pathlib import Path
import re
import sys

from .recorder import EvidenceRecorder
from .schema import ContractError, import_markdown, validate_graph


CHECKS = {'readable', 'exists', 'no_cell_errors', 'formula_cache', 'syntax',
          'library_policy', 'paths_defined', 'save_declared', 'execution_success',
          'sheets_present'}


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) - {'check', 'allowed', 'names'}:
        raise ContractError('Invalid spreadsheet check parameters')
    if spec.get('check') not in CHECKS:
        raise ContractError('Unknown fixed spreadsheet check')
    for key in ('allowed', 'names'):
        if key in spec and (not isinstance(spec[key], list) or
                            not all(isinstance(x, str) and x for x in spec[key])):
            raise ContractError('Spreadsheet check list must contain strings')


def _cell(value):
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


def file_facts(path, max_bytes=32 * 1024 * 1024):
    p = Path(path)
    facts = {'path': p.name, 'exists': p.is_file()}
    if not facts['exists']:
        return facts
    facts['size'] = p.stat().st_size
    if facts['size'] > max_bytes:
        facts['capture_error'] = 'File exceeds capture byte limit'
        return facts
    facts['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return facts


def workbook_facts(path, max_cells=100000, max_sheets=64, sample_cells=40):
    import openpyxl
    from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula
    facts = file_facts(path)
    if not facts['exists'] or 'capture_error' in facts:
        return facts
    books = []
    try:
        wf = openpyxl.load_workbook(path, data_only=False, read_only=False, keep_links=False)
        books.append(wf)
        wv = openpyxl.load_workbook(path, data_only=True, read_only=False, keep_links=False)
        books.append(wv)
        facts.update(readable=True, sheet_names=wf.sheetnames, defined_names=list(wf.defined_names),
                     sheets=[], cell_errors=[],
                     formulas=[], uncached_formulas=[], scan_complete=True, scanned_cells=0)
        for ws in wf.worksheets[:max_sheets]:
            info = dict(name=ws.title, max_row=ws.max_row, max_column=ws.max_column,
                        merged_ranges=[str(x) for x in ws.merged_cells.ranges],
                        tables=list(ws.tables), sheet_state=ws.sheet_state, samples=[])
            facts['sheets'].append(info)
            # Bound both axes before iter_rows; malformed dimensions must not allocate huge rows.
            if ws.max_row * ws.max_column > max_cells - facts['scanned_cells']:
                facts['scan_complete'] = False
            cols = min(ws.max_column, max_cells)
            rows = min(ws.max_row, max(0, (max_cells-facts['scanned_cells']) // max(cols, 1)))
            for row in ws.iter_rows(max_row=rows, max_col=cols) if rows else []:
                for c in row:
                    facts['scanned_cells'] += 1
                    ref = f"'{ws.title}'!{c.coordinate}"
                    value = c.value
                    if isinstance(value, ArrayFormula):
                        value = dict(type='array', ref=value.ref, text=value.text)
                    elif isinstance(value, DataTableFormula):
                        value = dict(type='data_table', **dict(value))
                    else:
                        value = _cell(value)
                    if value is not None and len(info['samples']) < sample_cells:
                        info['samples'].append(dict(cell=c.coordinate, value=value))
                    if c.data_type == 'e':
                        facts['cell_errors'].append(dict(cell=ref, value=str(c.value)))
                    if c.data_type == 'f':
                        cached = wv[ws.title][c.coordinate]
                        facts['formulas'].append(dict(cell=ref, formula=value, cached=_cell(cached.value)))
                        if cached.value is None:
                            facts['uncached_formulas'].append(ref)
                        if cached.data_type == 'e':
                            facts['cell_errors'].append(dict(cell=ref, value=str(cached.value)))
            if facts['scanned_cells'] >= max_cells:
                break
        if len(facts['sheets']) != len(wf.worksheets):
            facts['scan_complete'] = False
    except Exception as exc:
        facts.update(readable=False, read_error=f'{type(exc).__name__}: {exc}')
    finally:
        for book in books:
            book.close()
    return facts


def script_facts(path):
    facts = file_facts(path, max_bytes=1000000)
    if not facts['exists'] or 'capture_error' in facts:
        return facts
    try:
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeError) as exc:
        facts.update(syntax_ok=False, parse_error=str(exc))
        return facts
    imports, assignments, saves = set(), set(), []
    dynamic = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(x.name.split('.')[0] for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or '').split('.')[0])
            dynamic |= bool(node.level)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assignments.update(x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ('exec', 'eval', '__import__'):
                dynamic = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('import_module',):
                dynamic = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('save', 'to_excel'):
                saves.append(dict(line=node.lineno, method=node.func.attr,
                                  uses_output_path=any(isinstance(x, ast.Name) and x.id == 'OUTPUT_PATH'
                                                       for x in ast.walk(node))))
    facts.update(syntax_ok=True, imports=sorted(imports), assigned_names=sorted(assignments),
                 save_calls=saves, dynamic_code=dynamic,
                 analysis_scope='static syntax only; does not prove runtime behavior')
    return facts


def evaluate_spec(spec, actual):
    """Return status/reason. References are added by the existing CheckEngine."""
    validate_spec(spec)
    if not isinstance(actual, dict):
        return 'unknown', 'Captured artifact metadata is missing'
    check = spec['check']
    def yes(ok, why): return ('pass' if ok else 'fail'), why
    if check == 'execution_success':
        if actual.get('timed_out'):
            return 'fail', 'Observed process timeout'
        if type(actual.get('returncode')) is not int:
            return 'unknown', 'No actual process return code captured'
        return yes(actual['returncode'] == 0, 'Observed process return code; not task correctness')
    if 'exists' not in actual or 'capture_error' in actual:
        return 'unknown', actual.get('capture_error', 'No file observation')
    if not actual['exists']:
        return 'fail', 'Requested artifact was absent at the actual checkpoint'
    if check == 'exists':
        return 'pass', 'Artifact exists at checkpoint; contents not validated'
    if check == 'readable':
        if 'readable' not in actual: return 'unknown', 'No workbook open result'
        return yes(actual['readable'], 'Workbook open result')
    if check in ('syntax', 'library_policy', 'paths_defined', 'save_declared'):
        if 'syntax_ok' not in actual: return 'unknown', 'Script AST not captured'
        if not actual['syntax_ok']: return 'fail', 'Python script cannot be parsed'
        if check == 'syntax': return 'pass', 'Python AST parses; runtime untested'
        if check == 'paths_defined':
            return yes({'INPUT_PATH', 'OUTPUT_PATH'} <= set(actual['assigned_names']), 'Path variable assignments in AST')
        if check == 'save_declared':
            return yes(any(c['uses_output_path'] for c in actual['save_calls']),
                       'Static output save call declaration; does not prove a file was saved')
        if actual['dynamic_code']: return 'unknown', 'Dynamic or relative imports cannot be validated statically'
        allowed = set(spec.get('allowed', ['openpyxl', 'pandas'])) | set(sys.stdlib_module_names)
        return yes(not (set(actual['imports']) - allowed), 'Static import roots compared with library policy')
    if not actual.get('readable'):
        return 'unknown', 'Workbook contents were not readable'
    if check == 'sheets_present':
        if not spec.get('names'): return 'not_applicable', 'No required sheet names configured'
        return yes(set(spec['names']) <= set(actual['sheet_names']), 'Required sheet names in captured workbook')
    if check == 'no_cell_errors':
        if actual['cell_errors']: return 'fail', 'Observed Excel error cells'
        if not actual['scan_complete']: return 'unknown', 'Bounded scan did not cover entire workbook'
        return 'pass', 'No Excel error cells in complete scan; not semantic correctness'
    if check == 'formula_cache':
        if actual['uncached_formulas']:
            return 'unknown', 'Formula cached values absent; blank result and missing calculation cannot be distinguished'
        if not actual['scan_complete']: return 'unknown', 'Formula scan incomplete'
        if not actual['formulas']: return 'not_applicable', 'Complete scan found no formula cells'
        return 'pass', 'Formula caches observed; freshness/correctness not established'
    raise ContractError('Unsupported check')


def initial_graph(content):
    """Import initial Markdown verbatim at node/workflow level. No best checkpoint."""
    graph = import_markdown(content, 'spreadsheetbench_initial')
    layout = [
        [('input', 'readable')],
        [('script', 'library_policy')],
        [('script', 'syntax'), ('script', 'paths_defined')],
        [('script', 'save_declared')],
        [('execution', 'execution_success'), ('output', 'exists')],
        [('output', 'readable'), ('output', 'no_cell_errors'), ('output', 'formula_cache')],
    ]
    expected = ['Explore Input File', 'Select Library', 'Write `solution.py`', 'Save Output', 'Execute', 'Confirm']
    if [n['name'] for n in graph['nodes']] != expected:
        raise ContractError('Initial upstream skill changed; review mapping before regeneration')
    for node, specs in zip(graph['nodes'], layout):
        node['evidence_expected'] = sorted({key for key, _ in specs})
        node['checks'] = [dict(check_id=f"{node['node_id']}_{name}", applies_when='always',
                               target=key, requirement=f'Captured {key}: {name}', evidence_needed=[key],
                               method='rule', rule=dict(kind='spreadsheet', path='/'+key, value={'check': name}))
                          for key, name in specs]
    return validate_graph(graph)


class SpreadsheetEvidenceSession:
    """Harness-owned checkpoint tool. No final files backfilled into earlier nodes."""
    def __init__(self, graph, task_id, workspace, input_name='input.xlsx', output_name='output.xlsx'):
        validate_graph(graph)
        self.graph = graph
        self.recorder = EvidenceRecorder(task_id, workspace)
        self.input_name, self.output_name = input_name, output_name
        self.active = set()
        self.execution = None

    def observe(self, action, observation='', reads=(), writes=()):
        return self.recorder.event('step', sorted(self.active), action=action[:12000],
                                   observation=observation[:8000], reads=reads, writes=writes)

    def _path(self, relative):
        p = (self.recorder.workspace / relative).resolve()
        if not p.is_relative_to(self.recorder.workspace):
            raise ContractError('Evidence path outside worker workspace')
        if any(re.search(r'gold|answer|oracle|verifier|^\.git$', part, re.I)
               for part in p.relative_to(self.recorder.workspace).parts):
            raise ContractError('Private evaluation artifacts cannot become node evidence')
        return p

    def checkpoint(self, node_id, phase):
        if node_id not in {n['node_id'] for n in self.graph['nodes']}:
            raise ContractError('Unknown node ID')
        if phase == 'start':
            self.execution = None
            self.active.add(node_id)
            self.recorder.start(node_id)
        elif phase == 'skip':
            self.active.discard(node_id)
            self.recorder.event('skip', [node_id])
        elif phase == 'end':
            values = dict(input=workbook_facts(self._path(self.input_name)),
                          script=script_facts(self._path('solution.py')),
                          output=workbook_facts(self._path(self.output_name)))
            script=self._path('solution.py')
            if script.is_file() and script.stat().st_size<=1000000:
                source=script.read_text(encoding='utf-8',errors='replace')
                values['script']['source']=source[:12000]
                values['script']['source_truncated']=len(source)>12000
            if self.execution is not None:
                # Only executions observed during this node visit may be attributed to it.
                if node_id in self.execution['node_ids']:
                    values['execution'] = self.execution
            self.recorder.end(node_id, observation='Harness captured files at explicit checkpoint', values=values)
            self.active.discard(node_id)
        else:
            raise ContractError('Unknown checkpoint phase')
        return 'Checkpoint recorded. Diagnostics do not trigger retry or task grading.'

    def process_observed(self, argv, returncode=None, timed_out=False, stdout='', stderr=''):
        self.execution = dict(argv=list(argv), returncode=returncode, timed_out=bool(timed_out),
                              stdout=stdout[:4000], stderr=stderr[:4000], node_ids=sorted(self.active))
        self.recorder.event('step', sorted(self.active), action=' '.join(argv),
                            observation=str(self.execution))

    def finish(self, official_success):
        return self.recorder.finish(official_success)
