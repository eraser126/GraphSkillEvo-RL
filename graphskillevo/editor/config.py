"""Offline-loadable configuration, no model initialization."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math
import yaml
from .schema import ContractError


@dataclass(frozen=True)
class EditorConfig:
    schema_version: int=1
    seed: int=42
    group_size: int=4
    updates: int=4
    dev_frequency: int=1
    learning_rate: float=1e-5
    clip: float=0.2
    epochs: int=1
    max_grad_norm: float=1.0
    cost_weight: float=0.01
    cost_unit: int=10000
    invalid_penalty: float=0.1
    max_calls: int=1000
    max_tokens: int | None=1000000
    max_new_tokens: int=512
    max_context: int=8192
    lora_rank: int=0
    device: str='cpu'
    variant: str='full'
    backend: str='original'
    base_config: str='configs/searchqa/default.yaml'
    runner_argv: tuple=()
    timeout: int=600
    fixed_semantic_judge: bool=False
    batch_size: int=4
    training_epochs: int=1
    state_bank_size: int=0
    exec_timeout: int=600

    @classmethod
    def load(cls,path):
        raw=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(raw,dict): raise ContractError('Editor config must be a mapping')
        try: cfg=cls(**raw)
        except TypeError as e: raise ContractError('Unknown editor config key') from e
        return cfg.validate()

    def validate(self):
        for name in ('seed','group_size','updates','dev_frequency','epochs','max_calls','max_new_tokens','max_context','timeout','cost_unit','batch_size','training_epochs','exec_timeout'):
            value=getattr(self,name)
            if type(value) is not int or value<0 or (name!='seed' and value==0):
                raise ContractError('Invalid integer setting: '+name)
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens <= 0):
            raise ContractError('max_tokens must be a positive integer or null')
        if self.schema_version!=1 or self.group_size<2 or self.max_new_tokens>=self.max_context:
            raise ContractError('Invalid schema/group/context setting')
        for name in ('learning_rate','clip','max_grad_norm','cost_weight','invalid_penalty'):
            x=getattr(self,name)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
                raise ContractError('Invalid numeric setting: '+name)
        if not 0<self.clip<1 or self.learning_rate<=0 or self.max_grad_norm<=0:
            raise ContractError('Invalid optimization setting')
        if self.backend not in ('original','command','spreadsheet') or self.variant not in ('full','fixed_checks','no_checks'):
            raise ContractError('Unknown backend/variant')
        if type(self.fixed_semantic_judge) is not bool or type(self.lora_rank) is not int or self.lora_rank<0:
            raise ContractError('Invalid judge/LoRA setting')
        if self.device!='cpu' and not self.device.startswith('cuda'): raise ContractError('Unsupported device')
        if type(self.state_bank_size) is not int or self.state_bank_size<0: raise ContractError('Invalid state bank size')
        if self.backend=='command' and (not self.runner_argv or not all(isinstance(x,str) for x in self.runner_argv)):
            raise ContractError('Command backend requires runner argv')
        return self

    def to_dict(self): return asdict(self)
