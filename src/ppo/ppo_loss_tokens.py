# ppo_loss_tokens.py

from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass
class PPOLossConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    max_log_ratio: float = 5.0  # token-level clamp


def ppo_loss_tokens(
    logprobs_new: torch.Tensor,   # [B, T-1]
    logprobs_old: torch.Tensor,   # [B, T-1] (no grad)
    advantages: torch.Tensor,     # [B]
    logprobs_ref: torch.Tensor,   # [B, T-1] (no grad)
    entropies: torch.Tensor,      # [B, T-1]
    mask: torch.Tensor,           # [B, T-1] bool
    config: PPOLossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    logprobs_old = logprobs_old.detach()
    logprobs_ref = logprobs_ref.detach()
    advantages = advantages.detach()

    # Broadcast advantages to tokens
    adv = advantages.view(-1, 1)  # [B, 1]

    # Ratio per token
    log_ratio = logprobs_new - logprobs_old
    log_ratio = torch.clamp(log_ratio, -config.max_log_ratio, config.max_log_ratio)
    ratio = torch.exp(log_ratio)

    clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)

    # PPO objective per token
    unclipped = ratio * adv
    clipped = clipped_ratio * adv
    obj = torch.minimum(unclipped, clipped)  # [B, T-1]

    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)

    policy_loss = -(obj * mask_f).sum() / denom

    # Clip fraction on masked tokens
    clipped_tokens = ((ratio > (1.0 + config.clip_ratio)) | (ratio < (1.0 - config.clip_ratio))) & mask
    clip_frac = clipped_tokens.float().sum() / denom

    # Token-level KL approx on masked tokens
    kl_tok = (logprobs_new - logprobs_ref)
    kl = (kl_tok * mask_f).sum() / denom
    # Prevent negative cancellation turning KL "off"
    kl = kl.abs()

    # Entropy bonus on masked tokens
    entropy = (entropies * mask_f).sum() / denom

    total_loss = policy_loss + config.kl_coef * kl - config.entropy_coef * entropy

    # NaN/Inf guard
    if not torch.isfinite(total_loss):
        total_loss = torch.zeros((), device=logprobs_new.device, requires_grad=True)

    with torch.no_grad():
        info = {
            "loss_total": float(total_loss.item()),
            "loss_policy": float(policy_loss.item()) if torch.isfinite(policy_loss) else float("nan"),
            "kl": float(kl.item()) if torch.isfinite(kl) else float("nan"),
            "entropy": float(entropy.item()) if torch.isfinite(entropy) else float("nan"),
            "clip_frac": float(clip_frac.item()) if torch.isfinite(clip_frac) else float("nan"),
            "n_tokens": float(denom.item()),
        }

    return total_loss, info
