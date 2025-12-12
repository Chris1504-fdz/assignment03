"""
dpo_loss.py

Assignment 3 - Part 3: Direct Preference Optimization (DPO)

Implements the DPO objective from:
  Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)

We optimize the policy parameters θ by minimizing:

  L_DPO(θ) = - E_{(x, y+, y-)} [ log σ( β * ( (log πθ(y+|x) - log πθ(y-|x))
                                               - (log π_ref(y+|x) - log π_ref(y-|x)) ) ) ]

Where:
- y+ is the chosen / preferred response
- y- is the rejected / less preferred response
- π_ref is a frozen reference model (usually the starting SFT policy)
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class DPOConfig:
    beta: float = 0.1    # temperature / scaling factor for preference strength


def dpo_loss(
    logp_pi_chosen: torch.Tensor,
    logp_pi_rejected: torch.Tensor,
    logp_ref_chosen: torch.Tensor,
    logp_ref_rejected: torch.Tensor,
    config: DPOConfig = DPOConfig(),
) -> (torch.Tensor, Dict[str, float]):
    """
    Args:
        logp_pi_chosen:    log πθ(y+|x), shape [B]
        logp_pi_rejected:  log πθ(y-|x), shape [B]
        logp_ref_chosen:   log π_ref(y+|x), shape [B]
        logp_ref_rejected: log π_ref(y-|x), shape [B]
        config:            DPO hyperparameters (beta)

    Returns:
        loss: scalar tensor (to backprop)
        info: dict of scalars for logging
    """
    # Ensure 1D
    logp_pi_chosen = logp_pi_chosen.view(-1)
    logp_pi_rejected = logp_pi_rejected.view(-1)
    logp_ref_chosen = logp_ref_chosen.view(-1)
    logp_ref_rejected = logp_ref_rejected.view(-1)

    # Policy log-ratio and reference log-ratio
    pi_log_ratio = logp_pi_chosen - logp_pi_rejected
    ref_log_ratio = logp_ref_chosen - logp_ref_rejected

    # "Advantage" term in DPO
    advantage = pi_log_ratio - ref_log_ratio  # [B]

    # DPO objective: we *maximize* log σ(beta * advantage), so loss is negative of that
    scaled_adv = config.beta * advantage
    loss = -torch.log(torch.sigmoid(scaled_adv) + 1e-8).mean()

    with torch.no_grad():
        # Preference accuracy: how often the policy scores chosen > rejected
        pref_acc = (pi_log_ratio > 0).float().mean().item()
        ref_pref_acc = (ref_log_ratio > 0).float().mean().item()
        mean_adv = advantage.mean().item()

        info = {
            "loss": float(loss.item()),
            "pref_acc": pref_acc,
            "ref_pref_acc": ref_pref_acc,
            "mean_advantage": mean_adv,
        }

    return loss, info


if __name__ == "__main__":
    # Tiny smoke test
    B = 4
    logp_pi_chosen = torch.tensor([0.0, -1.0, 0.5, 1.0])
    logp_pi_rejected = torch.tensor([-1.0, -1.0, 0.0, 0.0])
    logp_ref_chosen = torch.zeros(B)
    logp_ref_rejected = torch.zeros(B)

    loss, info = dpo_loss(
        logp_pi_chosen,
        logp_pi_rejected,
        logp_ref_chosen,
        logp_ref_rejected,
        DPOConfig(beta=0.1),
    )
    print("Test DPO loss:", loss.item())
    print("Info:", info)
