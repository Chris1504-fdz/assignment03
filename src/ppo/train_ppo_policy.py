"""
train_ppo_policy.py

PPO-based RLHF (Task B) - token-level PPO (stable)

Key fixes:
- Token-level PPO (per-token logprobs + mask) instead of sequence-summed logprobs
- logprobs_old computed once (no grad) per rollout, logprobs_new recomputed during PPO updates
- Generation robustness: remove_invalid_values + renormalize_logits
- Avoid empty-prompt mask edge cases (ensure at least one attended token)
- Safer full attention mask even when pad_token_id == eos_token_id
- Timestamped logs to logs/*.txt
- Subsample 10k prompts
"""

import os
import sys
import atexit
from datetime import datetime
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from ppo_loss_tokens import ppo_loss_tokens, PPOLossConfig


# ============================
# CONFIG
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POLICY_MODEL_NAME = "gpt2-medium"
REWARD_MODEL_NAME = "gpt2-medium"
REWARD_CHECKPOINT = "reward_model/gpt2_reward_model.pt"

MAX_PROMPT_LEN = 128
MAX_GEN_LEN = 64

BATCH_SIZE = 8
NUM_EPOCHS = 1
LOG_INTERVAL = 20
LR = 1e-5

SUBSAMPLE_SEED = 0
MAX_PROMPTS = 10000

NUM_WORKERS = 4
PIN_MEMORY = True

PPO_UPDATES_PER_BATCH = 4

# Sampling params
TOP_P = 0.9
TEMPERATURE = 1.0

ppo_cfg = PPOLossConfig(
    clip_ratio=0.2,
    kl_coef=0.1,
    entropy_coef=0.01,
    max_log_ratio=5.0,
)


# ============================
# TIMESTAMPED LOGGING
# ============================

def _ts_prefix() -> str:
    now = datetime.now()
    ms = now.microsecond // 1000
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{ms:03d}" + " | "

class TimestampedTee:
    def __init__(self, *streams):
        self.streams = streams
        self._buf = ""

    def write(self, data: str):
        if not data:
            return
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            out = _ts_prefix() + line + "\n"
            for s in self.streams:
                try:
                    s.write(out)
                    s.flush()
                except Exception:
                    pass

    def flush(self):
        if self._buf:
            out = _ts_prefix() + self._buf
            self._buf = ""
            for s in self.streams:
                try:
                    s.write(out)
                    s.flush()
                except Exception:
                    pass
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_timestamped_logging() -> str:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = POLICY_MODEL_NAME.replace("/", "_")
    log_path = os.path.join("logs", f"ppo_token_{safe_model}_bs{BATCH_SIZE}_{ts}.txt")

    f = open(log_path, "w", buffering=1)
    orig_out = sys.stdout
    orig_err = sys.stderr

    sys.stdout = TimestampedTee(orig_out, f)
    sys.stderr = TimestampedTee(orig_err, f)

    atexit.register(f.close)
    return log_path


# ============================
# REWARD MODEL
# ============================

class GPT2RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        lengths = attention_mask.sum(dim=1)        # [B]
        last_indices = (lengths - 1).clamp(min=0)

        _, _, hidden_size = hidden_states.size()
        last_idx = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_size)
        last_hidden = hidden_states.gather(dim=1, index=last_idx).squeeze(1)

        rewards = self.value_head(last_hidden).squeeze(-1)  # [B]
        return rewards


# ============================
# DATASET
# ============================

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ============================
# MASK BUILDING
# ============================

def build_full_attention_mask(
    gen_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """
    Build attention mask for prompt+response.
    Works even when pad_token_id == eos_token_id.

    Keeps:
    - prompt padding as 0 (left pad)
    - prompt tokens as 1
    - response tokens as 1 up to and including first EOS in the response (if present)
    """
    B, T_full = gen_ids.shape
    T_prompt = prompt_attention_mask.shape[1]

    attn_full = torch.zeros((B, T_full), device=gen_ids.device, dtype=torch.long)
    attn_full[:, :T_prompt] = prompt_attention_mask.long()

    for b in range(B):
        resp = gen_ids[b, T_prompt:]
        eos_pos = (resp == eos_token_id).nonzero(as_tuple=False).squeeze(-1)
        if eos_pos.numel() > 0:
            resp_len = int(eos_pos[0].item()) + 1
        else:
            resp_len = resp.numel()
        if resp_len > 0:
            attn_full[b, T_prompt:T_prompt + resp_len] = 1

    return attn_full


# ============================
# TOKEN LOGPROBS + ENTROPY + MASK
# ============================

def compute_token_logprobs_entropy_and_mask(
    model: nn.Module,
    input_ids: torch.Tensor,             # [B, T]
    attention_mask: torch.Tensor,        # [B, T]
    prompt_attention_mask: torch.Tensor  # [B, T_prompt]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      token_logprobs: [B, T-1]  log p(x_{t+1} | x_{<=t})
      token_entropies: [B, T-1] entropy at position t
      token_mask: [B, T-1] bool mask for response tokens only (and attended)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits                       # [B, T, V]
    logp_all = torch.log_softmax(logits, dim=-1)  # [B, T, V]

    B, T, V = logp_all.shape

    targets = input_ids[:, 1:]        # [B, T-1]
    logp_t = logp_all[:, :-1, :]      # [B, T-1, V]

    token_logprobs = logp_t.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    probs_t = logp_t.exp()
    token_entropies = -(probs_t * logp_t).sum(dim=-1)  # [B, T-1]

    # token_logprobs[:, j] corresponds to predicting token index (j+1)
    T_prompt = prompt_attention_mask.shape[1]
    token_indices = torch.arange(1, T, device=input_ids.device)  # 1..T-1

    attended = attention_mask[:, 1:].bool()  # [B, T-1]
    is_response = (token_indices >= T_prompt).unsqueeze(0)  # [1, T-1]
    token_mask = attended & is_response  # [B, T-1]

    return token_logprobs, token_entropies, token_mask


# ============================
# MAIN
# ============================

def main():
    log_path = setup_timestamped_logging()
    print(f"Logging to: {log_path}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device count visible: {torch.cuda.device_count()}")

    os.makedirs("models", exist_ok=True)

    random.seed(SUBSAMPLE_SEED)
    torch.manual_seed(SUBSAMPLE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SUBSAMPLE_SEED)

    print("Loading HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    prompts = ds["prompt"] if "prompt" in ds.column_names else ds["chosen"]
    print(f"Total prompts in dataset: {len(prompts)}")

    prompts = [p for p in prompts if isinstance(p, str) and p.strip() != ""]
    print(f"After filtering empties: {len(prompts)}")

    if len(prompts) > MAX_PROMPTS:
        idx = random.sample(range(len(prompts)), MAX_PROMPTS)
        prompts = [prompts[i] for i in idx]
    print(f"Using {len(prompts)} prompts for PPO training")

    prompt_ds = PromptDataset(prompts)
    pin_memory = bool(PIN_MEMORY and torch.cuda.is_available())
    prompt_loader = DataLoader(
        prompt_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
    )

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id

    print("Loading policy model...")
    base_policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(DEVICE)
    base_policy.config.use_cache = False

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        policy = nn.DataParallel(base_policy)
        print("Policy: using DataParallel")
    else:
        policy = base_policy
        print("Policy: single GPU/CPU")

    print("Loading reference model (frozen, single GPU)...")
    ref_policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(DEVICE)
    ref_policy.config.use_cache = False
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    print("Loading reward model...")
    reward_model = GPT2RewardModel(REWARD_MODEL_NAME).to(DEVICE)
    state_dict = torch.load(REWARD_CHECKPOINT, map_location=DEVICE)
    reward_model.load_state_dict(state_dict)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    global_update_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"===== PPO Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        policy.train()

        for batch_prompts in prompt_loader:
            batch_prompts = list(batch_prompts)

            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LEN,
                return_tensors="pt",
            )
            input_ids_prompt = enc["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask_prompt = enc["attention_mask"].to(DEVICE, non_blocking=True)

            # Ensure at least one attended token per row
            row_sums = attention_mask_prompt.sum(dim=1)
            bad = (row_sums == 0).nonzero(as_tuple=False).squeeze(-1)
            if bad.numel() > 0:
                attention_mask_prompt[bad, -1] = 1

            # ---- GENERATE ----
            with torch.no_grad():
                gen_model = policy.module if isinstance(policy, nn.DataParallel) else policy
                gen_model.eval()

                gen_ids = gen_model.generate(
                    input_ids=input_ids_prompt,
                    attention_mask=attention_mask_prompt,
                    max_new_tokens=MAX_GEN_LEN,
                    do_sample=True,
                    top_p=TOP_P,
                    temperature=TEMPERATURE,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                    remove_invalid_values=True,
                    renormalize_logits=True,
                )

            policy.train()

            # Build full mask
            attn_full = build_full_attention_mask(gen_ids, attention_mask_prompt, eos_id)

            # ---- fixed rollout quantities (old/ref/reward) ----
            with torch.no_grad():
                logprobs_old_tok, ent_old_tok, mask_tok = compute_token_logprobs_entropy_and_mask(
                    policy, gen_ids, attn_full, attention_mask_prompt
                )
                logprobs_old_tok = logprobs_old_tok.detach()
                mask_tok = mask_tok.detach()

                logprobs_ref_tok, _, _ = compute_token_logprobs_entropy_and_mask(
                    ref_policy, gen_ids, attn_full, attention_mask_prompt
                )
                logprobs_ref_tok = logprobs_ref_tok.detach()

                full_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                reward_enc = tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_PROMPT_LEN + MAX_GEN_LEN,
                    return_tensors="pt",
                )
                r_input_ids = reward_enc["input_ids"].to(DEVICE, non_blocking=True)
                r_attention_mask = reward_enc["attention_mask"].to(DEVICE, non_blocking=True)
                rewards = reward_model(r_input_ids, r_attention_mask)

                advantages = rewards - rewards.mean()
                advantages = advantages / (rewards.std() + 1e-8)
                advantages = advantages.detach()

            # ---- PPO updates ----
            last_info = None
            for _ in range(PPO_UPDATES_PER_BATCH):
                logprobs_new_tok, ent_tok, _ = compute_token_logprobs_entropy_and_mask(
                    policy, gen_ids, attn_full, attention_mask_prompt
                )

                loss, info = ppo_loss_tokens(
                    logprobs_new=logprobs_new_tok,
                    logprobs_old=logprobs_old_tok,
                    advantages=advantages,
                    logprobs_ref=logprobs_ref_tok,
                    entropies=ent_tok,
                    mask=mask_tok,
                    config=ppo_cfg,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

                global_update_step += 1
                last_info = info

                if global_update_step % LOG_INTERVAL == 0:
                    print(
                        f"[Update {global_update_step}] "
                        f"loss_total = {last_info['loss_total']:.4f} | "
                        f"policy_loss = {last_info['loss_policy']:.4f} | "
                        f"kl = {last_info['kl']:.4f} | "
                        f"entropy = {last_info['entropy']:.4f} | "
                        f"clip_frac = {last_info['clip_frac']:.3f} | "
                        f"tokens = {last_info['n_tokens']:.0f} | "
                        f"mean_reward = {rewards.mean().item():.4f}"
                    )

        state_dict_to_save = policy.module.state_dict() if isinstance(policy, nn.DataParallel) else policy.state_dict()
        ckpt_path = f"models/ppo_policy_token_epoch{epoch+1}.pt"
        torch.save(state_dict_to_save, ckpt_path)
        print(f"Saved PPO policy checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
