import torch


def ppo_update(policy, optimizer, buffer, cfg):
    data = buffer.arrays()
    device = next(policy.parameters()).device
    t = {k: torch.as_tensor(v, device=device,
                           dtype=torch.bool if k == "mask" else torch.long if k == "action" else torch.float32)
         for k, v in data.items()}
    advantages = t["advantage"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics = []
    for _ in range(cfg.update_epochs):
        order = torch.randperm(len(advantages), device=device)
        for idx in order.split(cfg.minibatch_size):
            dist, value = policy.distribution_value(t["features"][idx], t["mask"][idx])
            logp = dist.log_prob(t["action"][idx])
            ratio = (logp - t["logp"][idx]).exp()
            pg = torch.minimum(ratio * advantages[idx],
                               ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantages[idx])
            value_loss = (value - t["return"][idx]).square().mean()
            entropy = dist.entropy().mean()
            loss = -pg.mean() + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(grad):
                raise FloatingPointError("Non-finite PPO gradients")
            optimizer.step()
            metrics.append([float(loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                            float(((ratio - 1) - (logp - t["logp"][idx])).mean().detach())])
    return dict(zip(("loss", "value_loss", "entropy", "approx_kl"),
                    [sum(x[i] for x in metrics) / len(metrics) for i in range(4)]))
