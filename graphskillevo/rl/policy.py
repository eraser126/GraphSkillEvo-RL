import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_count, hidden_sizes=(128, 128)):
        super().__init__()
        layers, previous = [], obs_dim
        for width in hidden_sizes:
            layers += [nn.Linear(previous, width), nn.Tanh()]
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.actor = nn.Linear(previous, action_count)
        self.critic = nn.Linear(previous, 1)

    def distribution_value(self, features, mask):
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Every observation needs a legal action")
        z = self.trunk(features)
        logits = self.actor(z).masked_fill(~mask, torch.finfo(z.dtype).min)
        return Categorical(logits=logits), self.critic(z).squeeze(-1)

    @torch.no_grad()
    def sample(self, obs, deterministic=False):
        device = next(self.parameters()).device
        features = torch.as_tensor(obs["features"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)
        dist, value = self.distribution_value(features, mask)
        action = dist.logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action)), float(value)
