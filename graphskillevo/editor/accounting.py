"""Durable boundary budget; failures remain infrastructure failures."""
from pathlib import Path
import json
import math
import time


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, root, max_calls=10000, max_tokens=1000000):
        self.path = Path(root) / 'editor_costs.jsonl'; self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.tokens = 0
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                r = json.loads(line); self.calls += r['calls']; self.tokens += r['tokens']

    def check(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded('Editor run call budget exhausted at operation boundary')
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceeded('Editor run token budget exhausted at operation boundary')

    def add(self, stage, tokens=0, calls=1, status='ok'):
        if type(tokens) is not int or tokens<0 or type(calls) is not int or calls<0:
            raise ValueError('Invalid cost accounting')
        self.calls += calls; self.tokens += tokens
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,calls=calls,tokens=tokens,status=status))+'\n')

    def call(self, stage, fn):
        self.check(); status='ok'; before=before_calls=0
        # Existing API router token tracker, if configured; no model/API initialization here.
        try:
            from graphskillevo.model import get_token_summary
            summary=get_token_summary().get('_total',{})
            before=summary.get('total_tokens',0);before_calls=summary.get('calls',0)
        except ImportError:
            get_token_summary=None
        try: return fn()
        except Exception as exc:
            status=type(exc).__name__; raise
        finally:
            summary=get_token_summary().get('_total',{}) if get_token_summary else {}
            after=summary.get('total_tokens',before)
            calls=max(1,int(summary.get('calls',before_calls)-before_calls))
            self.add(stage,max(0,int(after-before)),calls=calls,status=status)


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)
