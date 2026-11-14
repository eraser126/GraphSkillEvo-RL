"""On-policy, completion-token GRPO for short two-turn LLM edit episodes."""
import math


def advantages(rewards):
    import torch
    values=torch.tensor(rewards,dtype=torch.float32)
    if values.numel()<2 or not torch.isfinite(values).all(): raise ValueError('GRPO needs a finite group of >=2')
    std=values.std(unbiased=False)
    return torch.zeros_like(values) if std<1e-8 else (values-values.mean())/(std+1e-8)


def clipped_loss(new_logps,old_logps,advantage,clip=0.2):
    import torch
    if new_logps.shape!=old_logps.shape or not new_logps.numel(): raise ValueError('Completion shape mismatch')
    if not torch.isfinite(new_logps).all() or not torch.isfinite(old_logps).all():
        raise ValueError('Nonfinite token likelihood')
    # Reject grossly off-policy data rather than silently clamping log ratios.
    delta=new_logps-old_logps
    if delta.abs().max()>30: raise ValueError('Likelihood ratio overflow/off-policy group')
    ratio=delta.exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()


def update_group(editor,candidates,optimizer,clip=0.2,epochs=1,max_grad_norm=1.0):
    import torch
    rewards=[c.reward for c in candidates]
    adv=advantages(rewards)
    if not adv.any(): return dict(updated=False,reason='zero_group_variance',loss=0.0)
    for c in candidates:
        if not c.turns: raise ValueError('Candidate has no sampled editor actions')
        for t in c.turns:
            if len(t.completion_ids)!=len(t.old_logps) or not t.prompt_ids or not t.completion_ids:
                raise ValueError('API/scripted actions have no trainable token likelihoods')
    losses=[]
    editor.model.eval()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total=0.0
        for index,c in enumerate(candidates):
            # Each episode has equal weight, and each turn equal weight within the episode.
            # Tokens in observations/prompts are never targets. No per-node reward is fabricated.
            for turn in c.turns:
                new=editor.logps(turn.prompt_ids,turn.completion_ids)
                old=torch.tensor(turn.old_logps,device=new.device,dtype=new.dtype)
                loss=clipped_loss(new,old,adv[index].to(new.device),clip)/(len(candidates)*len(c.turns))
                if not torch.isfinite(loss): raise ValueError('Nonfinite GRPO loss')
                loss.backward();total+=float(loss.detach())
                del new,old,loss
        parameters=[p for p in editor.model.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            optimizer.zero_grad(set_to_none=True);raise ValueError('Nonfinite editor gradient')
        torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm)
        optimizer.step();losses.append(total)
    return dict(updated=True,loss=sum(losses)/len(losses),rewards=rewards)
