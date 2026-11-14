"""Run the initial skill on SpreadsheetBench 2 Debugging training tasks."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait
from contextlib import contextmanager
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from zipfile import is_zipfile

from graphskillevo.editor.accounting import Ledger, write_json
from graphskillevo.editor.evidence import Trace
from graphskillevo.editor.schema import render_worker, validate_graph
from graphskillevo.editor.spreadsheet import SpreadsheetEvidenceSession
from graphskillevo.editor.verification import CheckEngine


LO = Path('/opt/libreoffice7.6/program')
SYSTEM = """Audit and repair the supplied Excel workbook according to the task instruction.
Use the task workspace and Python with openpyxl. The input workbook is input.xlsx.
Write solution.py, run it with execute_solution, and save the repaired workbook as output.xlsx.
Preserve formulas and unrelated content. Formula caches may be empty until the workbook is recalculated.
Use bash to inspect the workbook and write_file to write scripts. Report the saved output when finished.
"""


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_file(dataset, relative):
    path = (dataset / relative).resolve()
    if not path.is_relative_to(dataset) or not path.is_file():
        raise ValueError(f'Missing or invalid workbook path: {relative}')
    return path


def grouped_splits(dataset):
    rows = json.loads((dataset / 'dataset.json').read_text())
    groups = defaultdict(list)
    if len(rows) != 100 or len({r['id'] for r in rows}) != 100:
        raise ValueError('Expected 100 unique Debugging tasks')
    for row in rows:
        case_file(dataset, row['spreadsheet_path'])
        groups[case_file(dataset, row['golden_response_path'])].append(row)
    keys = sorted(groups)
    if len(keys) != 10 or any(len(rows) != 10 for rows in groups.values()):
        raise ValueError('Expected ten source workbooks with ten variants each')
    random.Random(42).shuffle(keys)
    partitions = {'train':keys[:6], 'val':keys[6:8], 'test':keys[8:]}
    splits = {name:[r['id'] for key in group_keys for r in groups[key]]
              for name,group_keys in partitions.items()}
    pilot = []
    for i,key in enumerate(partitions['train']):
        variants = sorted(groups[key],key=lambda row:row['id'])
        pilot.extend(variants[(2*i+j)%len(variants)] for j in range(2))
    return splits,pilot


class WorkerTools:
    """Use an unprivileged process and a clean environment for task tools."""

    def __init__(self, work, packages):
        self.work = work
        self.env = {'PATH':f'{packages.parent / "bin"}:/usr/bin:/bin',
                    'PYTHONPATH':str(packages),'LANG':'C.UTF-8','TMPDIR':str(work)}

    def run(self, argv, timeout, input=None):
        return subprocess.run(argv,cwd=self.work,env=self.env,user=65534,group=65534,
                              extra_groups=[],umask=0o077,input=input,capture_output=True,
                              text=True,errors='replace',timeout=timeout)

    def write_file(self, name, content):
        path = (self.work / name).resolve()
        if not path.is_relative_to(self.work) or path == self.work or path == self.work/'input.xlsx':
            return 'Write rejected: choose a file within the task workspace other than input.xlsx.'
        code = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(sys.stdin.read(),encoding="utf-8")'
        result = self.run(['python','-c',code,str(path)],timeout=15,input=content)
        return f'File written: {name}' if result.returncode == 0 else result.stderr[:1000]


@contextmanager
def recalculation_service(vendor, root):
    recalc = load_module(vendor/'evaluation/open_spreadsheet.py','spreadsheet_v2_recalc')
    recalc.LIBREOFFICE_PROGRAM_DIR = str(LO)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',2002)) == 0:
            raise RuntimeError('LibreOffice UNO port 2002 is already in use')
    profile = (root/'lo-profile').as_uri()
    with (root/'libreoffice.log').open('a') as log:
        process = subprocess.Popen([str(LO/'soffice'),f'-env:UserInstallation={profile}',
                                    '--headless','--accept=socket,host=127.0.0.1,port=2002;urp;',
                                    '--norestore','--nofirststartwizard'],stdout=log,stderr=log)
        try:
            for _ in range(30):
                if recalc._check_service_ready():
                    break
                if process.poll() is not None:
                    raise RuntimeError('LibreOffice exited. See libreoffice.log.')
                time.sleep(1)
            else:
                raise RuntimeError('LibreOffice UNO service did not become ready')
            yield recalc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out-root',type=Path,required=True)
    parser.add_argument('--allow-api',action='store_true')
    args = parser.parse_args()
    repo = Path.cwd()
    dataset = (repo/'data/spreadsheetbench_v2/recalculated/Debugging').resolve()
    vendor = repo/'vendor/SpreadsheetBench-2'
    graph = validate_graph(json.loads((repo/'examples/editor/spreadsheetbench_v2_debugging_initial/initial_graph.json').read_text()))
    splits,pilot = grouped_splits(dataset)
    if not args.allow_api:
        print(json.dumps({'splits':{k:len(v) for k,v in splits.items()},'pilot':[r['id'] for r in pilot]}))
        return
    root = args.out_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',{'splits':splits,'pilot':[r['id'] for r in pilot],
                                    'split_seed':42,'grouped_by':'reference workbook','max_turns':30})
    ledger = Ledger(root,max_calls=10000,max_tokens=None)
    run_tasks(vendor,dataset,graph,ledger,pilot,root,stage='spreadsheet_v2_baseline')


def configure_worker(base_config):
    from graphskillevo.config import load_graphskillevo_config
    from graphskillevo.model_setup import configure_models

    base, _ = load_graphskillevo_config(base_config)
    configure_models(base)


def execute_task(dataset, graph, system, task, public, packages, result_dir,
                 stage, max_turns, exec_timeout):
    from graphskillevo.envs.spreadsheetbench.react_agent import _build_user, _react_loop

    tid = result_dir.name
    work = public/tid
    work.mkdir(mode=0o700)
    os.chown(work,65534,65534)
    source = case_file(dataset,task['spreadsheet_path'])
    input_file = work/'input.xlsx'
    shutil.copyfile(source,input_file)
    input_file.chmod(0o444)
    output_file = work/'output.xlsx'
    session = SpreadsheetEvidenceSession(graph,tid,work)
    session.observe('TASK_CONTEXT',task['instruction'])
    tools = WorkerTools(work,packages)
    probe = tools.run(['python','-c','import openpyxl; import os; assert os.getuid() == 65534; assert not os.access("/root",os.R_OK)'],timeout=15)
    if probe.returncode:
        raise RuntimeError('Worker tools failed preflight: '+probe.stderr)
    user = _build_user(task['instruction'],str(input_file),str(output_file),'','')
    print(f'Start {tid}',flush=True)
    usage = Ledger(result_dir, max_tokens=None)
    raw = usage.call(stage,lambda:_react_loop(
        system,user,str(work),max_turns,16384,evidence_session=session,exec_timeout=exec_timeout,tool_runner=tools))
    write_json(result_dir/'conversation.json',raw['conversation'])
    for name in ('solution.py','output.xlsx'):
        path = work/name
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path,result_dir/name)
    return session.finish(False).public(), raw['n_turns']


def score_task(vendor, grader, dataset, graph, root, task, result_dir, record, tool_calls):
    tid = result_dir.name
    prediction = result_dir/f'{tid}_output.xlsx'
    invalid_output = False
    if (result_dir/'output.xlsx').is_file():
        if not prediction.is_file():
            shutil.copyfile(result_dir/'output.xlsx',prediction)
        invalid_output = not is_zipfile(prediction)
        # The official grader rejects broken XLSX archives. Recalculation would
        # fail first and abort the batch before the grader can record the result.
        if not invalid_output:
            before = prediction.stat().st_mtime_ns
            with recalculation_service(vendor,root) as recalc:
                recalc.batch_open_files([str(prediction)])
            if prediction.stat().st_mtime_ns == before:
                raise RuntimeError(f'LibreOffice did not recalculate {tid}')
    score,missing = grader.process_single_item(task,str(dataset),str(result_dir),{},'Debugging')
    if record is not None:
        trace = Trace.create({**record, 'success': score['accuracy']==1})
        write_json(result_dir/'editor_trace.json',trace.public())
        write_json(result_dir/'node_checks.json',CheckEngine().run(graph,[trace]))
    result = {'id':tid,'accuracy':score['accuracy'],'missing_output':missing,
              'modification_accuracy':score['modification_accuracy'],
              'regression_accuracy':score['regression_accuracy'],'tool_calls':tool_calls}
    if invalid_output:
        result['output_error'] = 'invalid_xlsx_archive'
        result['error_message'] = score['error_message']
    write_json(result_dir/'official_result.json',result)
    return result


def run_tasks(vendor, dataset, graph, ledger, tasks, root, *, stage, max_turns=30,
              exec_timeout=600, workers=1, base_config='configs/spreadsheetbench/default.yaml',
              prior_results=()):
    """Run task tools in isolated processes and score completed waves serially."""
    if workers < 1:
        raise ValueError('At least one worker is required')
    grader = load_module(vendor/'evaluation/evaluation.py','spreadsheet_v2_official_eval')
    system = SYSTEM+'\n'+render_worker(graph)
    results = list(prior_results)
    with TemporaryDirectory(prefix='ssb2-worker-') as tmp:
        public = Path(tmp)
        public.chmod(0o711)
        packages = public/'packages'
        packages.mkdir(mode=0o755)
        import openpyxl, et_xmlfile
        for package in (openpyxl,et_xmlfile):
            source = Path(package.__file__).parent
            shutil.copytree(source,packages/source.name,ignore=shutil.ignore_patterns('__pycache__'))
        bin_dir = public/'bin'
        bin_dir.mkdir(mode=0o755)
        launcher = bin_dir/'python'
        launcher.write_text(f'#!/bin/sh\nexec {LO}/python "$@"\n')
        launcher.chmod(0o755)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=configure_worker, initargs=(str(base_config),)) as pool:
            for start in range(0, len(tasks), workers):
                ledger.check()
                pending = []
                for task in tasks[start:start+workers]:
                    tid = str(task['id'])
                    if not tid or Path(tid).name != tid or tid in ('.', '..'):
                        raise ValueError(f'Invalid task ID: {tid}')
                    result_dir = root/tid
                    result_dir.mkdir()
                    future = pool.submit(execute_task, dataset, graph, system, task, public,
                                         packages, result_dir, stage, max_turns, exec_timeout)
                    pending.append((task, result_dir, future))
                # Keep the privileged UNO service closed while task tools are running.
                wait([future for _, _, future in pending])
                # Worker routers have separate counters. Import each task's usage once,
                # even when an exception prevents scoring the wave.
                for _, result_dir, _ in pending:
                    path = result_dir/'editor_costs.jsonl'
                    if path.is_file():
                        for line in path.read_text(encoding='utf-8').splitlines():
                            ledger.add(**json.loads(line))
                for task, result_dir, future in pending:
                    record, tool_calls = future.result()
                    result = score_task(vendor, grader, dataset, graph, root, task,
                                        result_dir, record, tool_calls)
                    results.append(result)
                    write_json(root/'summary.json',{'completed':len(results),'planned':len(tasks)+len(prior_results),
                                                    'passed':sum(r['accuracy']==1 for r in results),'results':results})
                    print(json.dumps(result),flush=True)
    return results


if __name__=='__main__':
    main()
