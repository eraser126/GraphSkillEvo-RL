"""Synthetic files only: no benchmark solving, API requests or training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from graphskillevo.editor.spreadsheet import (
    initial_graph, workbook_facts, script_facts, SpreadsheetEvidenceSession, evaluate_spec)
from graphskillevo.editor.schema import ContractError, apply_patch, render_worker, import_markdown
from graphskillevo.editor.verification import CheckEngine
from graphskillevo.editor.episode import propose
from graphskillevo.editor.policy import ScriptedEditor


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def graph():
    return initial_graph((ROOT/'graphskillevo/initial_skills/spreadsheetbench/init_skill.md').read_text())


def workbook(path, formula=False, error=False):
    wb = openpyxl.Workbook()
    wb.active['A1'] = '=1+2' if formula else 3
    if error: wb.active['B1'] = '#DIV/0!'
    wb.save(path); wb.close()


def test_initial_preserves_nodes_and_workflow(graph):
    source = import_markdown((ROOT/'graphskillevo/initial_skills/spreadsheetbench/init_skill.md').read_text(), graph['skill_id'])
    assert len(graph['nodes']) == 6
    assert graph['workflows'] == source['workflows']
    assert graph['global_guidance'] == source['global_guidance']
    for a, b in zip(graph['nodes'], source['nodes']):
        for key in ('node_id', 'instruction', 'name'): assert a[key] == b[key]
    assert render_worker(graph) == render_worker(source)


def test_no_real_checkpoints_never_pass(graph, tmp_path):
    workbook(tmp_path/'output.xlsx')
    s = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    report = CheckEngine().run(graph, [s.finish(True)])[0]
    assert {x['status'] for x in report['checks']} == {'unknown'}


def test_immutable_checkpoint_not_backfilled(graph, tmp_path):
    s = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    nid = graph['nodes'][0]['node_id']
    s.checkpoint(nid, 'start'); s.checkpoint(nid, 'end')
    workbook(tmp_path/'input.xlsx')
    report = CheckEngine().run(graph, [s.finish(False)])[0]
    observed = [x for x in report['checks'] if x['node_id'] == nid]
    assert observed[0]['status'] == 'fail'
    assert observed[0]['snapshot_id'] and observed[0]['evidence_keys'] == ['input']


def test_error_and_truncated_scan(tmp_path):
    workbook(tmp_path/'error.xlsx', error=True)
    f = workbook_facts(tmp_path/'error.xlsx')
    assert evaluate_spec({'check':'no_cell_errors'}, f)[0] == 'fail'
    workbook(tmp_path/'ok.xlsx')
    wb = openpyxl.load_workbook(tmp_path/'ok.xlsx'); wb.active['Z100'] = 1; wb.save(tmp_path/'ok.xlsx'); wb.close()
    f = workbook_facts(tmp_path/'ok.xlsx', max_cells=10)
    assert not f['scan_complete']
    assert evaluate_spec({'check':'no_cell_errors'}, f)[0] == 'unknown'


def test_formula_cache_and_not_applicable(tmp_path):
    workbook(tmp_path/'f.xlsx', formula=True)
    facts = workbook_facts(tmp_path/'f.xlsx')
    assert evaluate_spec({'check':'formula_cache'}, facts)[0] == 'unknown'
    workbook(tmp_path/'literal.xlsx')
    facts = workbook_facts(tmp_path/'literal.xlsx')
    assert evaluate_spec({'check':'formula_cache'}, facts)[0] == 'not_applicable'
    assert evaluate_spec({'check':'sheets_present','names':['Absent']}, facts)[0] == 'fail'


def test_special_formulas_survive_node_checkpoint(graph, tmp_path):
    from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

    wb = openpyxl.Workbook()
    wb.active['A1'] = ArrayFormula('A1:A2', '=ROW(A1:A2)')
    wb.active['C1'] = DataTableFormula('C1:D2', r1='F1')
    wb.save(tmp_path/'input.xlsx')
    wb.close()

    session = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    node_id = graph['nodes'][0]['node_id']
    session.checkpoint(node_id, 'start')
    session.checkpoint(node_id, 'end')
    facts = session.finish(False).public()['snapshots'][0]['values']['input']
    formulas = [row['formula'] for row in facts['formulas']]
    assert formulas[0] == {'type': 'array', 'ref': 'A1:A2', 'text': '=ROW(A1:A2)'}
    assert formulas[1]['type'] == 'data_table'
    assert formulas[1]['ref'] == 'C1:D2' and formulas[1]['r1'] == 'F1'


def test_script_ast_never_executes(tmp_path):
    p = tmp_path/'solution.py'
    p.write_text("import openpyxl\nimport os\nINPUT_PATH='input.xlsx'\nOUTPUT_PATH='output.xlsx'\nwb.save(OUTPUT_PATH)\nraise RuntimeError('MUST NOT EXECUTE')\n")
    f = script_facts(p)
    for check in ['syntax', 'library_policy', 'paths_defined', 'save_declared']:
        assert evaluate_spec({'check':check}, f)[0] == 'pass'
    p.write_text("import requests\n")
    assert evaluate_spec({'check':'library_policy'}, script_facts(p))[0] == 'fail'
    p.write_text("exec('print(1)')")
    assert evaluate_spec({'check':'library_policy'}, script_facts(p))[0] == 'unknown'
    p.write_text('invalid(')
    assert evaluate_spec({'check':'syntax'}, script_facts(p))[0] == 'fail'


def test_execution_receipt_is_visit_specific(graph, tmp_path):
    s = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    execute, confirm = graph['nodes'][4]['node_id'], graph['nodes'][5]['node_id']
    s.checkpoint(execute, 'start'); s.process_observed(['python','solution.py'], 1)
    s.checkpoint(execute, 'end'); s.checkpoint(confirm, 'start'); s.checkpoint(confirm, 'end')
    trace = s.finish(False)
    assert 'execution' not in trace.public()['snapshots'][1]['values']
    assert evaluate_spec({'check':'execution_success'}, {})[0] == 'unknown'
    report = CheckEngine().run(graph, [trace])[0]
    assert any(x['status'] == 'fail' and x['check_id'].endswith('execution_success') for x in report['checks'])
    s.checkpoint(execute, 'start'); s.checkpoint(execute, 'end')
    assert 'execution' not in s.finish(False).public()['snapshots'][-1]['values']


def test_private_paths_and_symlinks_rejected(graph, tmp_path):
    outside = tmp_path.parent/'outside.xlsx'; workbook(outside)
    (tmp_path/'input.xlsx').symlink_to(outside)
    s = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    with pytest.raises(ContractError): s._path('input.xlsx')
    with pytest.raises(ContractError): s._path('golden.xlsx')
    with pytest.raises(ContractError): s._path('../outside.xlsx')


def test_edit_rechecks_same_evidence(graph, tmp_path):
    workbook(tmp_path/'output.xlsx', error=True)
    nid = graph['nodes'][5]['node_id']
    s = SpreadsheetEvidenceSession(graph, 'synthetic', tmp_path)
    s.checkpoint(nid,'start'); s.checkpoint(nid,'end'); trace = s.finish(False)
    original = trace.encoded
    check = deepcopy(graph['nodes'][5]['checks'][1]); check['rule']['value'] = {'check':'readable'}
    snapshot_id = trace.public()['snapshots'][0]['snapshot_id']
    p = dict(base_version=0,operations=[dict(op='update_check',node_id=nid,payload=check)],
             evidence_ids=[snapshot_id],rationale='Synthetic contract edit')
    noop = dict(base_version=1,operations=[],evidence_ids=[trace.id],rationale='No graph edit')
    candidate = propose(graph,[trace],CheckEngine(),ScriptedEditor([json.dumps(p),json.dumps(noop)]))
    assert candidate.valid and candidate.audit['changed_checks']
    assert trace.encoded == original
    assert [json.loads(t.prompt[1]['content'])['response_template']['base_version']
            for t in candidate.turns] == [0, 1]
    unknown = dict(p, evidence_ids=[snapshot_id + '_unknown'])
    rejected = propose(graph,[trace],CheckEngine(),ScriptedEditor([json.dumps(unknown)]))
    assert not rejected.valid and rejected.audit['error'] == 'Patch cites unknown evidence'
    check['rule']['value'] = {'check':'execute_arbitrary_code'}
    p['operations'][0]['payload'] = check
    with pytest.raises(ContractError): apply_patch(graph,p,'checks')


def test_react_checkpoints_diagnostic_only(graph, tmp_path, monkeypatch):
    from graphskillevo.envs.spreadsheetbench import react_agent as react
    nid = graph['nodes'][0]['node_id']
    def msg(phase):
        return SimpleNamespace(content='',tool_calls=[{'id':phase,'function':{'name':'node_checkpoint',
                               'arguments':json.dumps(dict(node_id=nid,phase=phase))}}])
    messages = iter([msg('start'),msg('end'),SimpleNamespace(content='done',tool_calls=[])])
    monkeypatch.setattr(react,'chat_target_messages',lambda **kwargs:(next(messages),None))
    s = SpreadsheetEvidenceSession(graph,'synthetic',tmp_path)
    result = react._react_loop('system','user',str(tmp_path),3,100,evidence_session=s)
    assert result['n_turns'] == 2
    assert all('retry' in x['obs'] for x in result['conversation'] if x['type']=='tool_call')
    assert CheckEngine().run(graph,[s.finish(False)])[0]['checks'][0]['status'] == 'fail'


def test_official_comparison_synthetic_only(tmp_path):
    from graphskillevo.editor.spreadsheet_official import OfficialSpreadsheetEvaluator
    vendor = ROOT/'vendor/SpreadsheetBench'
    if not vendor.exists(): pytest.skip('Pinned upstream sources are required')
    grader = OfficialSpreadsheetEvaluator(vendor,ROOT/'configs/editor/spreadsheet_official.lock.json')
    workbook(tmp_path/'prediction.xlsx'); workbook(tmp_path/'gold.xlsx')
    assert grader.grade(tmp_path/'prediction.xlsx',tmp_path/'gold.xlsx','Cell-Level Manipulation','Sheet!A1',recalculate=False)['ok']
    workbook(tmp_path/'prediction.xlsx',error=True)
    assert not grader.grade(tmp_path/'prediction.xlsx',tmp_path/'gold.xlsx','Cell-Level Manipulation','Sheet!B1',recalculate=False)['ok']


def test_verified_case_naming(tmp_path):
    from graphskillevo.editor.spreadsheet_official import verified_case
    workbook(tmp_path/'initial.xlsx');workbook(tmp_path/'golden.xlsx')
    assert verified_case(tmp_path)[0].name == 'initial.xlsx'
    workbook(tmp_path/'1_id_init.xlsx');workbook(tmp_path/'1_id_golden.xlsx')
    with pytest.raises(ContractError): verified_case(tmp_path)
    typo = tmp_path/'42930'; typo.mkdir()
    workbook(typo/'1_42930_init.xlsx');workbook(typo/'1_43930_golden.xlsx')
    assert verified_case(typo)[1].name == '1_43930_golden.xlsx'


def test_backend_uses_official_and_only_worker_visible_files(graph,tmp_path,monkeypatch):
    from graphskillevo.editor.spreadsheet_backend import SpreadsheetBackend
    from graphskillevo.editor.accounting import Ledger
    from graphskillevo.envs.spreadsheetbench import react_agent
    data = tmp_path/'data'; case = data/'spreadsheet'/'fake'; case.mkdir(parents=True)
    workbook(case/'1_fake_init.xlsx');workbook(case/'1_fake_golden.xlsx')
    calls = []
    def fake_worker(**kw):
        work = Path(kw['work_dir']); assert sorted(p.name for p in work.iterdir()) == ['input.xlsx']
        s = kw['evidence_session']; nid = graph['nodes'][5]['node_id']
        s.checkpoint(nid,'start'); workbook(work/'output.xlsx', error=True); s.checkpoint(nid,'end')
        calls.append(kw)
        return {'conversation':[]}
    monkeypatch.setattr(react_agent,'run_react',fake_worker)
    backend = SpreadsheetBackend(data,ROOT/'vendor/SpreadsheetBench',
                                ROOT/'configs/editor/spreadsheet_official.lock.json',tmp_path/'runs',Ledger(tmp_path/'ledger'))
    # Recalculation is independently tested using the actual upstream implementation.
    real_grade = backend.grader.grade
    monkeypatch.setattr(backend.grader,'grade',lambda *a,**k:real_grade(*a,recalculate=False))
    tasks = [dict(id='fake',payload={'instruction':'synthetic','answer_position':'A1','answer_sheet':'Sheet'})]
    rows = backend.run(graph,tasks,'support')
    assert len(calls) == 1 and rows[0]['score'] == 1
    reports = json.loads(next((tmp_path/'runs').glob('*/node_checks.json')).read_text())
    assert any(x['status']=='fail' for x in reports[0]['checks'])
    # A failed advisory check does not force the independent official score to zero.


def test_official_recalculation_preserves_source_files(tmp_path):
    import hashlib
    from graphskillevo.editor.spreadsheet_official import OfficialSpreadsheetEvaluator
    soffice = Path('/opt/libreoffice7.6/program/soffice')
    if not soffice.is_file(): pytest.skip('Prepared Linux LibreOffice 7.6 required')
    grader = OfficialSpreadsheetEvaluator(ROOT/'vendor/SpreadsheetBench',ROOT/'configs/editor/spreadsheet_official.lock.json')
    prediction, gold = tmp_path/'formula.xlsx', tmp_path/'gold.xlsx'
    workbook(prediction,formula=True);workbook(gold)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (prediction,gold)]
    assert grader.grade(prediction,gold,'Cell-Level Manipulation','Sheet!A1',soffice=str(soffice))['ok']
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (prediction,gold)]


def test_spreadsheet_runtime_config_uses_new_backend_without_api(tmp_path, monkeypatch):
    import graphskillevo.model_setup as setup
    from graphskillevo.editor.runner import build_runtime
    from graphskillevo.editor.config import EditorConfig
    from graphskillevo.editor.spreadsheet_backend import SpreadsheetBackend
    monkeypatch.setattr(setup,'configure_models',lambda base: None)
    cfg = EditorConfig.load(ROOT/'configs/editor/spreadsheet_prepared.yaml')
    backend, engine, _ = build_runtime(cfg,tmp_path)
    assert isinstance(backend,SpreadsheetBackend)
    assert engine.fixed_judge is None
    assert backend.data_root == (ROOT/'data/spreadsheetbench_verified_400').resolve()
