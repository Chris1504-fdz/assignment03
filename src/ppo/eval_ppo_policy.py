"""
eval_ppo_policy.py

Assignment 3 - Part 2.2: Evaluate PPO policy vs base model.

- Load a small set of HH-RLHF prompts for evaluation
- Load frozen reward model (trained in Part 1)
- Evaluate:
    * Base GPT-2 policy (no PPO)
    * PPO policy loaded from checkpoint
- Compare mean reward under the reward model
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


# ============================
# CONFIG
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POLICY_MODEL_NAME = "gpt2"
REWARD_MODEL_NAME = "gpt2"
REWARD_CHECKPOINT = "reward_model/gpt2_reward_model.pt"
PPO_CKPT_PATH = "models/ppo_policy_epoch1.pt"   # <- your saved PPO checkpoint

MAX_PROMPT_LEN = 128
MAX_GEN_LEN = 64
EVAL_BATCH_SIZE = 8
N_EVAL_PROMPTS = 100
N_PRINT_EXAMPLES = 3


# ============================
# REWARD MODEL (same as train)
# ============================

class GPT2RewardModel(nn.Module):
    """
    Same architecture as in train_reward_model.py:
    base GPT-2 + linear scalar head over last non-pad token.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        lengths = attention_mask.sum(dim=1)        # [B]
        last_indices = (lengths - 1).clamp(min=0)

        batch_size, seq_len, hidden_size = hidden_states.size()
        last_indices_expanded = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_size)
        last_hidden = hidden_states.gather(dim=1, index=last_indices_expanded).squeeze(1)

        rewards = self.value_head(last_hidden).squeeze(-1)  # [B]
        return rewards


# ============================
# HELPERS
# ============================

def sample_eval_prompts(n_prompts: int) -> List[str]:
    """
    Load HH-RLHF and sample a subset of prompts for evaluation.
    """
    print("Loading HH-RLHF eval prompts...")
    # Use test split if available; many people just use 'train' for simplicity,
    # but your earlier logs say "eval prompts", so we try 'test'.
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="test")
    except Exception:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")

    if "prompt" in ds.column_names:
        prompts = ds["prompt"]
    else:
        prompts = ds["chosen"]

    n_prompts = min(n_prompts, len(prompts))
    indices = random.sample(range(len(prompts)), n_prompts)
    eval_prompts = [prompts[i] for i in indices]

    print(f"Using {len(eval_prompts)} prompts for evaluation")
    return eval_prompts


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_reward_model(device: torch.device) -> GPT2RewardModel:
    print("Loading reward model for evaluation...")
    rm = GPT2RewardModel(REWARD_MODEL_NAME)
    state_dict = torch.load(REWARD_CHECKPOINT, map_location="cpu")
    rm.load_state_dict(state_dict)
    rm.to(device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False
    return rm


def generate_and_score(
    model: AutoModelForCausalLM,
    prompts: List[str],
    tokenizer: AutoTokenizer,
    reward_model: GPT2RewardModel,
    device: torch.device,
    max_prompt_len: int,
    max_gen_len: int,
    batch_size: int,
) -> List[float]:
    """
    Generate responses for each prompt, then score full (prompt+response) with reward model.
    Returns list of scalar rewards.
    """
    all_rewards: List[float] = []

    model.eval()
    reward_model.eval()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_len,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            full_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            reward_enc = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=max_prompt_len + max_gen_len,
                return_tensors="pt",
            )
            r_input_ids = reward_enc["input_ids"].to(device)
            r_attention_mask = reward_enc["attention_mask"].to(device)

            rewards = reward_model(r_input_ids, r_attention_mask)  # [B]

        all_rewards.extend(rewards.cpu().tolist())

    return all_rewards


def print_sample_generations(
    model: AutoModelForCausalLM,
    prompts: List[str],
    tokenizer: AutoTokenizer,
    reward_model: GPT2RewardModel,
    device: torch.device,
    max_prompt_len: int,
    max_gen_len: int,
    n_examples: int,
    tag: str,
):
    """
    Print a few sample generations and their rewards.
    """
    print(f"\n==== {tag}: sample generations ====\n")

    indices = list(range(len(prompts)))
    random.shuffle(indices)
    indices = indices[:n_examples]

    for i, idx in enumerate(indices, start=1):
        prompt = prompts[idx]

        enc = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_len,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            reward_enc = tokenizer(
                [full_text],
                padding=True,
                truncation=True,
                max_length=max_prompt_len + max_gen_len,
                return_tensors="pt",
            )
            r_input_ids = reward_enc["input_ids"].to(device)
            r_attention_mask = reward_enc["attention_mask"].to(device)

            reward = reward_model(r_input_ids, r_attention_mask)[0].item()

        print(f"--- Example {i} ---")
        print("[PROMPT]:\n")
        print(prompt.strip(), "\n")
        print(f"[{tag} REWARD]: {reward:.4f}")
        print(f"[{tag} RESPONSE]:\n")
        print(full_text.strip(), "\n")


# ============================
# MAIN
# ============================

def main():
    print(f"Using device: {DEVICE}")

    # 1) Prompts
    eval_prompts = sample_eval_prompts(N_EVAL_PROMPTS)

    # 2) Tokenizer & reward model
    tokenizer = build_tokenizer()
    reward_model = build_reward_model(DEVICE)

    # 3) Evaluate base GPT-2
    print("\nLoading base GPT-2 (no PPO)...")
    base_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME).to(DEVICE)

    print("\nEvaluating base model...")
    base_rewards = generate_and_score(
        model=base_model,
        prompts=eval_prompts,
        tokenizer=tokenizer,
        reward_model=reward_model,
        device=DEVICE,
        max_prompt_len=MAX_PROMPT_LEN,
        max_gen_len=MAX_GEN_LEN,
        batch_size=EVAL_BATCH_SIZE,
    )
    mean_base_reward = sum(base_rewards) / len(base_rewards)
    print(f"Mean reward - base GPT-2: {mean_base_reward:.4f}")

    # Print a few base generations (this should match the style of your earlier logs)
    print_sample_generations(
        model=base_model,
        prompts=eval_prompts,
        tokenizer=tokenizer,
        reward_model=reward_model,
        device=DEVICE,
        max_prompt_len=MAX_PROMPT_LEN,
        max_gen_len=MAX_GEN_LEN,
        n_examples=N_PRINT_EXAMPLES,
        tag="BASE",
    )

    # 4) Free base model to avoid CUDA OOM before loading PPO model
    del base_model
    torch.cuda.empty_cache()

    # 5) Load PPO policy from checkpoint
    print("\nLoading PPO policy from checkpoint...")
    # First create a fresh GPT-2, then load weights on CPU to avoid GPU OOM during load
    ppo_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME)
    state_dict = torch.load(PPO_CKPT_PATH, map_location="cpu")
    ppo_model.load_state_dict(state_dict)
    ppo_model.to(DEVICE)
    ppo_model.eval()

    # 6) Evaluate PPO policy
    print("\nEvaluating PPO policy...")
    ppo_rewards = generate_and_score(
        model=ppo_model,
        prompts=eval_prompts,
        tokenizer=tokenizer,
        reward_model=reward_model,
        device=DEVICE,
        max_prompt_len=MAX_PROMPT_LEN,
        max_gen_len=MAX_GEN_LEN,
        batch_size=EVAL_BATCH_SIZE,
    )
    mean_ppo_reward = sum(ppo_rewards) / len(ppo_rewards)
    print(f"Mean reward - PPO policy: {mean_ppo_reward:.4f}")

    # 7) Sample PPO generations
    print_sample_generations(
        model=ppo_model,
        prompts=eval_prompts,
        tokenizer=tokenizer,
        reward_model=reward_model,
        device=DEVICE,
        max_prompt_len=MAX_PROMPT_LEN,
        max_gen_len=MAX_GEN_LEN,
        n_examples=N_PRINT_EXAMPLES,
        tag="PPO",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
