# grpo_loss.py

"""
Group Relative Policy Optimization (GRPO) loss.

- We sample K responses per prompt.
- For each prompt i we have rewards r_{i,1..K}.
- We compute group mean: b_i = mean_k r_{i,k}.
- Advantages: A_{i,k} = r_{i,k} - b_i.
- Loss is standard policy gradient with optional entropy bonus:
      L = - E[ A_{i,k} * log pi(a_{i,k} | x_i) ] - beta * H
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class GRPOConfig:
    entropy_coef: float = 0.01  # exploration bonus


def grpo_loss(
    logprobs: torch.Tensor,     # shape [B * G]
    advantages: torch.Tensor,   # shape [B * G]
    entropies: torch.Tensor,    # shape [B * G]
    config: GRPOConfig = GRPOConfig(),
) -> (torch.Tensor, Dict[str, float]):

    # Detach advantages so we do not backprop through rewards
    advantages = advantages.detach()

    # Flatten
    logprobs = logprobs.view(-1)
    advantages = advantages.view(-1)
    entropies = entropies.view(-1)

    # Policy gradient loss: negative because we want to maximize A * log pi
    policy_loss = -torch.mean(advantages * logprobs)

    # Entropy bonus
    entropy_mean = torch.mean(entropies)
    total_loss = policy_loss - config.entropy_coef * entropy_mean

    with torch.no_grad():
        info = {
            "loss_total": float(total_loss.item()),
            "loss_policy": float(policy_loss.item()),
            "entropy": float(entropy_mean.item()),
        }

    return total_loss, info


if __name__ == "__main__":
    # tiny smoke test
    B, G = 4, 3
    logprobs = torch.zeros(B * G).normal_(mean=-1.0, std=0.5)
    rewards = torch.randn(B, G)
    # group baseline
    group_mean = rewards.mean(dim=1, keepdim=True)
    adv = (rewards - group_mean).view(-1)
    ent = torch.ones_like(logprobs) * 2.0

    loss, info = grpo_loss(logprobs, adv, ent)
    print("Test loss:", loss.item())
    print("Info:", info)
