"""
train_dpo_policy.py

Assignment 3 - Part 3: Direct Preference Optimization (DPO)

- Loads tokenized HH-RLHF pairwise dataset from disk
- Policy model: trainable
- Reference model: frozen
- Optimizes DPO loss over chosen/rejected pairs
"""

import os
import sys
import atexit
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from dpo_loss import dpo_loss, DPOConfig


# ============================
# CONFIG
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2-medium"
TRAIN_PATH = "hh_rlhf_train_tokenized"
VAL_PATH = "hh_rlhf_val_tokenized"

BATCH_SIZE = 4
LR = 1e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 20

# DataLoader perf knobs
NUM_WORKERS = 4
PIN_MEMORY = True

# Subsampling (recommended for faster iteration)
SUBSAMPLE_SEED = 0
MAX_TRAIN = 10000   
MAX_VAL = 2000

# Multi-GPU policy only (reference stays single GPU to save VRAM)
USE_POLICY_DATAPARALLEL = True

DPO_CFG = DPOConfig(beta=0.1)


# ============================
# TIMESTAMPED LOGGING (stdout + stderr)
# ============================

def _ts_prefix() -> str:
    now = datetime.now()
    ms = now.microsecond // 1000
    return now.strftime("%Y-%m-%d %H:%M:%S") + f",{ms:03d}" + " | "

class TimestampedTee:
    """
    Wraps stdout/stderr and prefixes each completed line with a timestamp.
    Buffers partial lines so timestamps only appear at line boundaries.
    """
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
                s.write(out)
                s.flush()

    def flush(self):
        # Flush any remaining partial line
        if self._buf:
            out = _ts_prefix() + self._buf
            self._buf = ""
            for s in self.streams:
                s.write(out)
                s.flush()
        for s in self.streams:
            s.flush()


def setup_timestamped_logging() -> str:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = MODEL_NAME.replace("/", "_")
    log_path = os.path.join("logs", f"dpo_{safe_model}_bs{BATCH_SIZE}_{ts}.txt")

    log_f = open(log_path, "w", buffering=1)

    orig_out = sys.stdout
    orig_err = sys.stderr

    sys.stdout = TimestampedTee(orig_out, log_f)
    sys.stderr = TimestampedTee(orig_err, log_f)

    atexit.register(log_f.close)
    return log_path


# ============================
# HELPER: SEQUENCE LOG-PROB
# ============================

def sequence_logprob(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]
    logprobs_all = torch.log_softmax(logits, dim=-1)  # [B, T, V]

    target_ids = input_ids[:, 1:].contiguous()        # [B, T-1]
    logprobs_all = logprobs_all[:, :-1, :]            # [B, T-1, V]
    target_mask = attention_mask[:, 1:].float()       # [B, T-1]

    token_logps = logprobs_all.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    token_logps = token_logps * target_mask
    return token_logps.sum(dim=1)                     # [B]


# ============================
# VALIDATION LOOP
# ============================

@torch.no_grad()
def evaluate(policy: nn.Module, ref_policy: nn.Module, val_loader: DataLoader):
    policy.eval()
    ref_policy.eval()

    total_loss = 0.0
    total_pref_acc = 0.0
    total_ref_pref_acc = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

        logp_pi_chosen = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        logp_pi_rejected = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"])

        logp_ref_chosen = sequence_logprob(ref_policy, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        logp_ref_rejected = sequence_logprob(ref_policy, batch["rejected_input_ids"], batch["rejected_attention_mask"])

        loss, info = dpo_loss(
            logp_pi_chosen,
            logp_pi_rejected,
            logp_ref_chosen,
            logp_ref_rejected,
            config=DPO_CFG,
        )

        total_loss += float(loss.item())
        total_pref_acc += float(info["pref_acc"])
        total_ref_pref_acc += float(info["ref_pref_acc"])
        num_batches += 1

    denom = max(1, num_batches)
    return total_loss / denom, total_pref_acc / denom, total_ref_pref_acc / denom


# ============================
# MAIN
# ============================

def main():
    log_path = setup_timestamped_logging()
    print(f"Logging to: {log_path}")

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device count visible: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading tokenized datasets from disk...")
    train_hf = load_from_disk(TRAIN_PATH)
    val_hf = load_from_disk(VAL_PATH)

    print(f"Original Train size: {len(train_hf)}")
    print(f"Original Val size:   {len(val_hf)}")

    # Subsample
    if MAX_TRAIN is not None and MAX_TRAIN > 0 and len(train_hf) > MAX_TRAIN:
        train_hf = train_hf.shuffle(seed=SUBSAMPLE_SEED).select(range(MAX_TRAIN))
    if MAX_VAL is not None and MAX_VAL > 0 and len(val_hf) > MAX_VAL:
        val_hf = val_hf.shuffle(seed=SUBSAMPLE_SEED).select(range(MAX_VAL))

    print(f"Using subset -> Train size: {len(train_hf)} | Val size: {len(val_hf)}")

    # HF -> torch tensors directly
    cols = ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]
    train_hf.set_format(type="torch", columns=cols)
    val_hf.set_format(type="torch", columns=cols)

    pin_memory = bool(PIN_MEMORY and torch.cuda.is_available())
    num_workers = int(NUM_WORKERS)

    train_loader = DataLoader(
        train_hf,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_hf,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    print("Initializing policy and reference models...")
    base_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    base_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # Disable KV cache for training (saves memory, helps throughput)
    base_policy.config.use_cache = False
    base_ref.config.use_cache = False

    # Policy: DataParallel if multiple GPUs
    if USE_POLICY_DATAPARALLEL and torch.cuda.device_count() > 1:
        policy = nn.DataParallel(base_policy)
        print("Policy: using DataParallel")
    else:
        policy = base_policy
        print("Policy: single GPU/CPU")

    # Reference: single GPU (frozen) to save VRAM
    ref_policy = base_ref
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    print("Reference: single GPU (frozen)")

    if isinstance(policy, nn.DataParallel):
        print("Policy param device:", next(policy.module.parameters()).device)
    else:
        print("Policy param device:", next(policy.parameters()).device)
    print("Ref param device:", next(ref_policy.parameters()).device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps) if total_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"===== DPO Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        policy.train()

        running_loss = 0.0
        running_pref_acc = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}

            logp_pi_chosen = sequence_logprob(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            logp_pi_rejected = sequence_logprob(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"])

            with torch.no_grad():
                logp_ref_chosen = sequence_logprob(ref_policy, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                logp_ref_rejected = sequence_logprob(ref_policy, batch["rejected_input_ids"], batch["rejected_attention_mask"])

            loss, info = dpo_loss(
                logp_pi_chosen,
                logp_pi_rejected,
                logp_ref_chosen,
                logp_ref_rejected,
                config=DPO_CFG,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_pref_acc += float(info["pref_acc"])
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                avg_acc = running_pref_acc / LOG_INTERVAL
                print(f"[Step {global_step}] loss = {avg_loss:.4f} | pref_acc = {avg_acc:.4f}")
                running_loss = 0.0
                running_pref_acc = 0.0

        val_loss, val_pref_acc, val_ref_pref_acc = evaluate(policy, ref_policy, val_loader)
        print(
            f"[Epoch {epoch + 1}] "
            f"Val loss = {val_loss:.4f} | Val pref_acc = {val_pref_acc:.4f} | Ref pref_acc = {val_ref_pref_acc:.4f}"
        )

        os.makedirs("models", exist_ok=True)
        state_dict = policy.module.state_dict() if isinstance(policy, nn.DataParallel) else policy.state_dict()
        ckpt_path = f"models/dpo_policy_epoch{epoch+1}.pt"
        torch.save(state_dict, ckpt_path)
        print(f"Saved DPO policy checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
