"""Synthetic CUDA backward only: no tasks, APIs, optimizer steps or saved weights."""
import argparse,gc,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cases',default='2048:2,6144:2048,14336:2048')
    p.add_argument('--allow-model-load',action='store_true')
    a=p.parse_args()
    if not a.allow_model_load:p.error('Explicit --allow-model-load required; no parameter update will occur')
    out=Path(a.out)
    if out.exists():p.error('Choose a new output file')
    import torch
    if not torch.cuda.is_available():p.error('CUDA required')
    from graphskillevo.editor.policy import LocalLLMEditor
    editor=LocalLLMEditor.load(a.model,device='cuda',max_new_tokens=2048,max_context=16384,lora_rank=8)
    # CPU copies verify parameters without adding a second GPU copy.
    before={n:v.detach().cpu().clone() for n,v in editor.model.named_parameters() if v.requires_grad}
    parameters=[v for v in editor.model.parameters() if v.requires_grad]
    # Reserve Adam first/second moment storage, without performing optimizer.step.
    moments=[torch.zeros_like(v) for v in parameters for _ in range(2)]
    results=[]
    for item in a.cases.split(','):
        prompt_length,completion_length=map(int,item.split(':'))
        new=None;loss=None
        editor.model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        t=time.time();row=dict(prompt_tokens=prompt_length,completion_tokens=completion_length)
        try:
            prompt=[32+i%64 for i in range(prompt_length)]
            completion=[96+i%64 for i in range(completion_length)]
            new=editor.logps(prompt,completion);loss=-new.mean();loss.backward();torch.cuda.synchronize()
            grads=[v.grad for v in parameters if v.grad is not None]
            finite=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            nonzero=any(bool(g.abs().max()>0) for g in grads)
            assert finite and nonzero,'Missing, nonfinite or zero adapter gradients'
            row.update(status='passed',finite=finite,nonzero_gradients=nonzero,gradient_tensors=len(grads))
            del new,loss
        except torch.cuda.OutOfMemoryError as exc:
            import traceback
            traceback.print_exc()
            row.update(status='oom',error=str(exc));new=None;loss=None;gc.collect()
        row.update(seconds=round(time.time()-t,2),peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
                   peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3))
        results.append(row);print(json.dumps(row),flush=True)
    unchanged=all(torch.equal(v.detach().cpu(),before[n]) for n,v in editor.model.named_parameters() if v.requires_grad)
    base=editor.model.get_base_model()
    report=dict(model=a.model,torch_version=torch.__version__,attention_implementation=base.config._attn_implementation,
                cases=results,weights_unchanged=unchanged,optimizer_step=False,
                adam_moments_reserved=True,gradient_checkpointing=editor.checkpointing,
                logits_selector=editor.logits_selector,logprob_chunk_size=editor.logprob_chunk_size,
                task_runs=0,api_calls=0)
    with out.open('x') as f:json.dump(report,f,indent=2)
    assert unchanged
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
