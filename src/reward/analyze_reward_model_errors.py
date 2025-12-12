"""
analyze_reward_model_errors.py

Part 1.2 - Task B: Error analysis of the reward model.

- Rebuilds the same train/val split as in preprocessing (seed=42, 10% val)
- Loads the trained GPT-2 reward model
- Runs it on the validation set
- Finds examples where the model *disagrees* with the human preference
- Prints 20+ mistaken cases for qualitative analysis
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

MODEL_NAME = "gpt2"
CHECKPOINT_PATH = "reward_model/gpt2_reward_model.pt"
MAX_LENGTH = 512
VAL_FRACTION = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_valid_examples(example):
    chosen = example["chosen"]
    rejected = example["rejected"]
    if chosen is None or rejected is None:
        return False
    if len(chosen.strip()) == 0 or len(rejected.strip()) == 0:
        return False
    return True


class GPT2RewardModel(nn.Module):
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

        lengths = attention_mask.sum(dim=1)  # [B]
        last_indices = (lengths - 1).clamp(min=0)

        batch_size, seq_len, hidden_size = hidden_states.size()
        last_indices_expanded = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_size)
        last_hidden = hidden_states.gather(dim=1, index=last_indices_expanded).squeeze(1)

        rewards = self.value_head(last_hidden).squeeze(-1)  # [B]
        return rewards


def tokenize_batch(texts, tokenizer):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def main():
    print("Loading raw HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.filter(filter_valid_examples)

    print(f"Total valid examples: {len(ds)}")

    # Recreate the same train/val split as preprocessing
    splits = ds.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    val_ds = splits["test"]
    print(f"Validation size (raw text): {len(val_ds)}")

    print("Loading tokenizer and reward model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2RewardModel(MODEL_NAME).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    mistaken_examples = []
    margins = []

    # Go through some or all validation examples
    # You can cap this to, say, first 5000 for speed if you want
    for i in range(len(val_ds)):
        ex = val_ds[i]
        chosen_text = ex["chosen"]
        rejected_text = ex["rejected"]

        # Tokenize both
        with torch.no_grad():
            chosen_ids, chosen_mask = tokenize_batch([chosen_text], tokenizer)
            rejected_ids, rejected_mask = tokenize_batch([rejected_text], tokenizer)

            chosen_ids = chosen_ids.to(DEVICE)
            chosen_mask = chosen_mask.to(DEVICE)
            rejected_ids = rejected_ids.to(DEVICE)
            rejected_mask = rejected_mask.to(DEVICE)

            r_chosen = model(chosen_ids, chosen_mask).item()
            r_rejected = model(rejected_ids, rejected_mask).item()

        # Human label says: chosen > rejected
        # Model is wrong if it assigns lower or equal reward
        if r_chosen <= r_rejected:
            mistaken_examples.append(
                {
                    "index": i,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "r_chosen": r_chosen,
                    "r_rejected": r_rejected,
                    "margin": r_chosen - r_rejected,
                }
            )
            margins.append(r_chosen - r_rejected)

        # Stop once we have, say, 30-40 mistakes
        if len(mistaken_examples) >= 40:
            break

    print(f"\nCollected {len(mistaken_examples)} mistaken examples.")
    if margins:
        print(f"Mean margin (r_chosen - r_rejected) among mistakes: {np.mean(margins):.4f}")

    # Print 20+ mistakes for manual inspection
    print("\n==== SAMPLE MISTAKES (for error analysis) ====\n")

    for j, ex in enumerate(mistaken_examples[:25]):
        print(f"--- Mistake {j+1} (val index {ex['index']}) ---")
        print(f"r_chosen  = {ex['r_chosen']:.4f}")
        print(f"r_rejected= {ex['r_rejected']:.4f}")
        print(f"margin    = {ex['margin']:.4f}")
        print("\n[CHOSEN (human-preferred)]:")
        print(ex["chosen"][:1000])  # truncate to avoid insane length
        print("\n[REJECTED]:")
        print(ex["rejected"][:1000])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
