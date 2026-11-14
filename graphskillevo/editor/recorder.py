"""Harness-side capture hook. Call at the actual event, never after task completion."""
from pathlib import Path
import json
from .evidence import Trace,snapshot
from .schema import ContractError


class EvidenceRecorder:
    def __init__(self,task_id,visible_workspace,max_file_bytes=100000):
        self.task_id,self.workspace=task_id,Path(visible_workspace).resolve()
        self.max_file_bytes=max_file_bytes
        self.events,self.snapshots=[],[]

    def event(self,kind,node_ids=(),action='',observation='',reads=(),writes=()):
        index=len(self.events)
        self.events.append(dict(index=index,kind=kind,node_ids=list(node_ids),action=action,
                                observation=observation,reads=list(reads),writes=list(writes)))
        return index

    def start(self,node_id): return self.event('node_start',[node_id])

    def end(self,node_id,observation='',visible_files=None,values=None):
        captured=dict(values or {})
        for label,relative in (visible_files or {}).items():
            path=(self.workspace/relative).resolve()
            if not path.is_relative_to(self.workspace): raise ContractError('Capture outside worker-visible workspace')
            if any(x.lower() in ('oracle','verifier','.git') for x in path.relative_to(self.workspace).parts):
                raise ContractError('Grader/oracle files cannot become node evidence')
            if not path.is_file() or path.stat().st_size>self.max_file_bytes:
                continue  # Explicitly missing; checks will return unknown.
            value=path.read_text(encoding='utf-8')
            captured[label]=json.loads(value) if path.suffix.lower()=='.json' else value
        index=self.event('node_end',[node_id],observation=observation)
        if captured: self.snapshots.append(snapshot(node_id,index,captured))
        return index

    def finish(self,success):
        return Trace.create(dict(task_id=self.task_id,events=self.events,snapshots=self.snapshots,success=success))
