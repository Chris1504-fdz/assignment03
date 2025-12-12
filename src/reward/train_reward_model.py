"""
train_reward_model.py

Assignment 3 - Part 1.2: Reward Model Training

- Loads tokenized HH-RLHF train/val datasets from disk
- Defines a GPT-2-based reward model
- Trains with pairwise ranking loss:
    L = -log(σ(r_chosen - r_rejected))
- Tracks:
    * loss
    * preference accuracy
    * gradient norms
"""

import math
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
import logging
import sys


# ============================
# LOGGING
# ============================

def setup_logger(log_path: str = "reward_training_log.txt"):
    logger = logging.getLogger("reward_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid double logging if root logger is configured

    # Clear existing handlers if re-running in the same process
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


# ============================
# CONFIG
# ============================

MODEL_NAME = "gpt2-medium"  # must match the tokenizer used in preprocessing
TRAIN_PATH = "hh_rlhf_train_tokenized"
VAL_PATH = "hh_rlhf_val_tokenized"

BATCH_SIZE = 8
LR = 1e-5
NUM_EPOCHS = 5          # more epochs
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# DATASET WRAPPER
# ============================

class PairwiseRewardDataset(torch.utils.data.Dataset):
    """
    Wraps the tokenized dataset.
    Each item returns chosen and rejected sequences.
    """

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]
        return {
            "chosen_input_ids": torch.tensor(item["chosen_input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(item["chosen_attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(item["rejected_input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(item["rejected_attention_mask"], dtype=torch.long),
        }


# ============================
# REWARD MODEL
# ============================

class GPT2RewardModel(nn.Module):
    """
    Reward model: base LM (GPT-2) + scalar head.

    For each sequence, we:
    - get hidden states from the base model
    - pick the last non-padding token representation
    - push through a linear layer to get a scalar reward
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Returns: rewards, shape [batch]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        # Find index of last non-pad token per sequence
        # attention_mask: 1 for valid tokens, 0 for pad
        lengths = attention_mask.sum(dim=1)  # [B]
        last_indices = (lengths - 1).clamp(min=0)  # avoid negative

        # Gather the hidden state at the last index for each sequence
        batch_size, seq_len, hidden_size = hidden_states.size()
        last_indices_expanded = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_size)
        last_hidden = hidden_states.gather(dim=1, index=last_indices_expanded).squeeze(1)  # [B, H]

        rewards = self.value_head(last_hidden).squeeze(-1)  # [B]
        return rewards


# ============================
# TRAINING UTILITIES
# ============================

def pairwise_loss(r_chosen, r_rejected):
    """
    L = -log σ(r_chosen - r_rejected)
    """
    diff = r_chosen - r_rejected
    return -torch.log(torch.sigmoid(diff) + 1e-12).mean()


def compute_accuracy(r_chosen, r_rejected):
    """
    Preference prediction accuracy:
    fraction of pairs where r_chosen > r_rejected.
    """
    diff = r_chosen - r_rejected
    return (diff > 0).float().mean().item()


def compute_grad_norm(model):
    # Handle DataParallel vs normal model
    if isinstance(model, nn.DataParallel):
        params = model.module.parameters()
    else:
        params = model.parameters()

    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            param_norm_sq = p.grad.data.norm(2).item() ** 2
            total_norm_sq += param_norm_sq
    return math.sqrt(total_norm_sq)


# ============================
# MAIN
# ============================

def main():
    logger.info(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading tokenized datasets from disk...")
    train_hf = load_from_disk(TRAIN_PATH)
    val_hf = load_from_disk(VAL_PATH)

    logger.info(f"Train size: {len(train_hf)}")
    logger.info(f"Val size: {len(val_hf)}")

    train_ds = PairwiseRewardDataset(train_hf)
    val_ds = PairwiseRewardDataset(val_hf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Initializing reward model...")
    base_model = GPT2RewardModel(MODEL_NAME).to(DEVICE)
    model = nn.DataParallel(base_model)  # use all visible GPUs

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps) if total_steps > 0 else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # New-style AMP scaler
    scaler = GradScaler("cuda" if DEVICE.type == "cuda" else None)

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            # autocast with new torch.amp API
            with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                r_chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                r_rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

                loss = pairwise_loss(r_chosen, r_rejected)
                acc = compute_accuracy(r_chosen, r_rejected)

            # backward with scaler
            scaler.scale(loss).backward()

            # Unscale before clipping so clip works on real gradients
            scaler.unscale_(optimizer)

            # Compute grad norm after unscaling (before clipping)
            grad_norm = compute_grad_norm(model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Optimizer & scheduler step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            running_acc += acc
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                avg_acc = running_acc / LOG_INTERVAL
                logger.info(
                    f"[Step {global_step}] "
                    f"loss = {avg_loss:.4f} | "
                    f"acc = {avg_acc:.4f} | "
                    f"grad_norm = {grad_norm:.4f}"
                )
                running_loss = 0.0
                running_acc = 0.0

        val_loss, val_acc = evaluate(model, val_loader)
        logger.info(f"[Epoch {epoch + 1}] Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}")

    os.makedirs("reward_model", exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, "reward_model/gpt2_reward_model.pt")
    logger.info("Saved reward model to reward_model/gpt2_reward_model.pt")


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            r_chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            r_rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

            loss = pairwise_loss(r_chosen, r_rejected)
            acc = compute_accuracy(r_chosen, r_rejected)

            total_loss += loss.item()
            total_acc += acc
            total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


if __name__ == "__main__":
    main()
