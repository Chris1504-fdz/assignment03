import os
import re
import random
from typing import List
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


# ---------------- logging ----------------
def setup_logger(log_path: str = "logs/grpo_training_log.txt"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("grpo_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers if re-running in same process
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ---------------- config ----------------
SEED = 42

POLICY_MODEL_NAME = "gpt2-medium"
REWARD_MODEL_NAME = "gpt2-medium"
REWARD_CHECKPOINT = "reward_model/gpt2_reward_model.pt"

MAX_PROMPT_LEN = 128
MAX_NEW_TOKENS = 64

BATCH_SIZE = 2          # start small on 16 GB; increase to 4 if stable
GROUP_SIZE = 4          # GRPO group size
NUM_EPOCHS = 1

# GRPO/PPO-ish update
LR = 1e-5
UPDATE_EPOCHS = 2       # >1 makes clipping matter
CLIP_EPS = 0.2
KL_COEF = 0.02
MAX_GRAD_NORM = 1.0

TOP_P = 0.9
TEMPERATURE = 1.0
MAX_PROMPTS = 2000

LOG_INTERVAL = 20
SAVE_DIR = "models"

# AMP
USE_AMP = True
AMP_DTYPE = torch.float16  # RTX 5000: fp16 is typical


# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_first_human_turn(raw_prompt: str) -> str:
    if raw_prompt is None:
        return ""
    parts = re.split(r"\n\s*Assistant\s*:\s*", raw_prompt, maxsplit=1)
    first_chunk = parts[0].strip()
    first_chunk = re.sub(r"^\s*Human\s*:\s*", "", first_chunk).strip()
    first_chunk = re.sub(r"\s+", " ", first_chunk).strip()
    return first_chunk

def format_hh_prompt(question: str) -> str:
    question = question.strip()
    return f"Human: {question}\n\nAssistant:"

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------- reward model (same architecture you trained) ----------------
class GPT2RewardModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B,T,H]
        lengths = attention_mask.sum(dim=1)
        last_indices = (lengths - 1).clamp(min=0)

        B, T, H = hidden_states.size()
        idx = last_indices.view(-1, 1, 1).expand(-1, 1, H)
        last_hidden = hidden_states.gather(dim=1, index=idx).squeeze(1)
        rewards = self.value_head(last_hidden).squeeze(-1)
        return rewards


# ---------------- masking helpers (pad_id == eos_id) ----------------
@torch.no_grad()
def build_attention_mask_until_first_eos(
    gen_ids: torch.Tensor,
    attn_prompt: torch.Tensor,
    prompt_len_padded: int,
    eos_id: int,
) -> torch.Tensor:
    """
    attn_prompt is left-padded prompt mask [N, prompt_len_padded]
    gen_ids is [N, T_full] = prompt_len_padded + generated
    Returns attn_full that zeros tokens after first eos in the generated region.
    """
    N, T_full = gen_ids.shape
    L = T_full - prompt_len_padded

    attn_full = torch.ones((N, T_full), device=gen_ids.device, dtype=attn_prompt.dtype)
    attn_full[:, :prompt_len_padded] = attn_prompt

    gen_region = gen_ids[:, prompt_len_padded:]          # [N, L]
    eos_hits = (gen_region == eos_id)                    # [N, L]
    has_eos = eos_hits.any(dim=1)                        # [N]

    first_eos = torch.argmax(eos_hits.int(), dim=1)      # [N] meaningful only if has_eos=True
    pos = torch.arange(L, device=gen_ids.device).unsqueeze(0)  # [1, L]

    keep_gen = (~has_eos).unsqueeze(1) | (pos <= first_eos.unsqueeze(1))
    attn_full[:, prompt_len_padded:] = keep_gen.to(attn_full.dtype)
    return attn_full

def response_target_mask(attn_full: torch.Tensor, prompt_len_padded: int) -> torch.Tensor:
    """
    Next-token mask [N, T-1] selecting only response targets.
    First response target is predicted at t = prompt_len_padded - 1.
    """
    m = attn_full[:, 1:].float()  # [N, T-1]
    start = prompt_len_padded - 1
    if start > 0:
        m[:, :start] = 0.0
    return m


# ---------------- logprob helper (OOM-safe) ----------------
def per_token_logprob_fused_ce(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Returns per-token logprob for the actual next token: [N, T-1]
    Uses fused cross entropy (no log_softmax materialization).
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [N, T, V]
    N, T, V = logits.shape

    logits = logits[:, :-1, :].contiguous()   # [N, T-1, V]
    targets = input_ids[:, 1:].contiguous()   # [N, T-1]

    nll = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        reduction="none",
    ).view(N, T - 1)

    return -nll


def main():
    assert torch.cuda.is_available(), "Need CUDA for this script."

    set_seed(SEED)
    device = torch.device("cuda:0")

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # -------- data --------
    logger.info("Loading HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    raw_prompts = ds["prompt"] if "prompt" in ds.column_names else ds["chosen"]

    if len(raw_prompts) > MAX_PROMPTS:
        idx = random.sample(range(len(raw_prompts)), MAX_PROMPTS)
        raw_prompts = [raw_prompts[i] for i in idx]

    questions = [extract_first_human_turn(p) for p in raw_prompts]
    prompts = [format_hh_prompt(q) for q in questions]
    logger.info(f"Using {len(prompts)} prompts for GRPO training")

    loader = DataLoader(PromptDataset(prompts), batch_size=BATCH_SIZE, shuffle=True)

    # -------- tokenizer --------
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    # -------- models --------
    logger.info("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    policy.train()
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    logger.info("Loading reference model (frozen)...")
    ref = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(device)
    ref.eval()
    ref.config.use_cache = False
    for p in ref.parameters():
        p.requires_grad = False

    logger.info("Loading reward model (frozen)...")
    reward_model = GPT2RewardModel(REWARD_MODEL_NAME).to(device)
    reward_model.load_state_dict(torch.load(REWARD_CHECKPOINT, map_location=device))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    global_step = 0
    logger.info(f"===== GRPO (single GPU) | device={device} =====")

    for epoch in range(NUM_EPOCHS):
        logger.info(f"===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        for batch_prompts in loader:
            batch_prompts = list(batch_prompts)
            B = len(batch_prompts)

            enc = tok(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LEN,
                return_tensors="pt",
            )
            input_ids_prompt = enc["input_ids"].to(device)
            attn_prompt = enc["attention_mask"].to(device)
            T_prompt = input_ids_prompt.shape[1]

            # Repeat prompts G times
            input_rep = input_ids_prompt.repeat_interleave(GROUP_SIZE, dim=0)
            attn_rep = attn_prompt.repeat_interleave(GROUP_SIZE, dim=0)
            N = input_rep.shape[0]  # N = B*G

            # -------- generate (no grad). Disable checkpointing for generate. --------
            policy_was_training = policy.training
            policy.eval()
            policy.config.use_cache = True
            policy.gradient_checkpointing_disable()

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                gen_ids = policy.generate(
                    input_ids=input_rep,
                    attention_mask=attn_rep,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    top_p=TOP_P,
                    temperature=TEMPERATURE,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            gen_ids = gen_ids.clone()  # avoid inference-tensor backward issues

            # Restore training mode
            policy.train(policy_was_training)
            policy.config.use_cache = False
            policy.gradient_checkpointing_enable()

            attn_full = build_attention_mask_until_first_eos(gen_ids, attn_rep, T_prompt, eos_id)
            resp_mask = response_target_mask(attn_full, T_prompt)      # [N, T-1]
            denom = resp_mask.sum(dim=1).clamp(min=1.0)                # [N]

            # -------- rewards (no grad) --------
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                rewards = reward_model(gen_ids, attn_full)             # [N]

            # group-wise normalized advantages
            rewards_bg = rewards.view(B, GROUP_SIZE)
            mean = rewards_bg.mean(dim=1, keepdim=True)
            std = rewards_bg.std(dim=1, keepdim=True).clamp(min=1e-6)
            adv_bg = (rewards_bg - mean) / std
            advantages = adv_bg.view(-1).detach()                      # [N]

            # -------- snapshot old logprobs and reference logprobs (no grad) --------
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                logp_old_tok = per_token_logprob_fused_ce(policy, gen_ids, attn_full)   # [N, T-1]
                logp_old_seq = (logp_old_tok * resp_mask).sum(dim=1) / denom            # [N]

                logp_ref_tok = per_token_logprob_fused_ce(ref, gen_ids, attn_full)      # [N, T-1]
                logp_ref_seq = (logp_ref_tok * resp_mask).sum(dim=1) / denom            # [N]

            # -------- GRPO/PPO-style updates on this batch --------
            for _ in range(UPDATE_EPOCHS):
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE):
                    logp_pi_tok = per_token_logprob_fused_ce(policy, gen_ids, attn_full)   # [N, T-1]
                    logp_pi_seq = (logp_pi_tok * resp_mask).sum(dim=1) / denom             # [N]

                    ratio = torch.exp(logp_pi_seq - logp_old_seq)                           # [N]
                    ratio_clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)

                    pg_unclipped = ratio * advantages
                    pg_clipped = ratio_clipped * advantages
                    loss_pg = -torch.mean(torch.minimum(pg_unclipped, pg_clipped))

                    kl_seq = (logp_pi_seq - logp_ref_seq)                                   # [N]
                    loss_kl = KL_COEF * kl_seq.mean()

                    loss = loss_pg + loss_kl

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

            global_step += 1
            if global_step % LOG_INTERVAL == 0:
                with torch.no_grad():
                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std().item()
                    clipfrac = ((ratio < (1.0 - CLIP_EPS)) | (ratio > (1.0 + CLIP_EPS))).float().mean().item()

                logger.info(
                    f"[Step {global_step}] "
                    f"loss={loss.item():.4f} | "
                    f"pg={loss_pg.item():.4f} | "
                    f"kl={loss_kl.item():.4f} | "
                    f"mean_reward={rewards.mean().item():.4f} | "
                    f"ratio_mean={ratio_mean:.3f} std={ratio_std:.3f} clipfrac={clipfrac:.3f}"
                )

        ckpt_path = os.path.join(SAVE_DIR, f"grpo_policy_epoch{epoch+1}.pt")
        torch.save(policy.state_dict(), ckpt_path)
        logger.info(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
