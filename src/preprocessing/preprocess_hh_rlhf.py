"""
preprocess_hh_rlhf.py

Part 1.1 - Task B: Data preprocessing pipeline for HH-RLHF

Functionality:
- Load Anthropic HH-RLHF dataset
- Filter out invalid pairs (empty or missing)
- Split into train / validation
- For each pair, extract (prompt, chosen_answer, rejected_answer)
- Tokenize (prompt + chosen_answer) and (prompt + rejected_answer)
- Save processed datasets to disk
"""

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# ============================
# CONFIGURATION
# ============================

MODEL_NAME = "gpt2"      # small model, good for reward model backbone later
MAX_LENGTH = 512         # max token length for truncation
VAL_FRACTION = 0.1       # validation split fraction
SEED = 42                # for reproducibility

SAVE_TO_DISK = True      # set to False if you do not want to save
TRAIN_SAVE_PATH = "hh_rlhf_train_tokenized"
VAL_SAVE_PATH = "hh_rlhf_val_tokenized"


# ============================
# PREPROCESSING FUNCTIONS
# ============================

def filter_valid_examples(example):
    """
    Filter function to remove invalid or empty preference pairs.
    Handles edge cases where chosen or rejected is missing or blank.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]

    if chosen is None or rejected is None:
        return False

    if len(chosen.strip()) == 0 or len(rejected.strip()) == 0:
        return False

    return True


def split_prompt_and_answer(dialog: str):
    """
    Split a full HH-RLHF dialog into:
        - prompt (everything up to and including the last 'Assistant:' tag)
        - answer (the text after that tag)

    If we can't find 'Assistant:', we treat the whole dialog as prompt
    and return empty answer.
    """
    if dialog is None:
        return "", ""

    # Try with newline first (typical format: '\n\nHuman: ...\n\nAssistant: ...')
    tag = "\nAssistant:"
    idx = dialog.rfind(tag)
    if idx == -1:
        # Fallback: search for 'Assistant:' without newline
        tag = "Assistant:"
        idx = dialog.rfind(tag)

    if idx == -1:
        # No Assistant: tag at all → treat everything as prompt
        return dialog, ""

    # Include the 'Assistant:' label in the prompt so the model sees who is speaking
    prompt = dialog[: idx + len(tag)]
    answer = dialog[idx + len(tag):].strip()
    return prompt, answer


def preprocess_batch(batch, tokenizer):
    """
    Tokenize chosen and rejected as:
        chosen_input_ids   = tokenizer(prompt + chosen_answer)
        rejected_input_ids = tokenizer(prompt + rejected_answer)

    This makes the reward model score answers conditioned on the same prompt,
    instead of scoring entire transcripts directly.
    """
    chosen_raw = batch["chosen"]
    rejected_raw = batch["rejected"]

    chosen_texts = []
    rejected_texts = []

    for chosen, rejected in zip(chosen_raw, rejected_raw):
        # Split each dialog into (prompt, answer)
        ch_prompt, ch_answer = split_prompt_and_answer(chosen)
        rj_prompt, rj_answer = split_prompt_and_answer(rejected)

        # In HH-RLHF, prompts should be the same for chosen/rejected.
        # If not, we can fall back to using the chosen prompt only.
        if ch_prompt.strip() != rj_prompt.strip():
            prompt = ch_prompt  # fallback: use chosen prompt
        else:
            prompt = ch_prompt

        # Build full texts seen by reward model
        # You can tweak spacing/newlines here if you like.
        chosen_text = (prompt + " " + ch_answer).strip()
        rejected_text = (prompt + " " + rj_answer).strip()

        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    chosen_enc = tokenizer(
        chosen_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    rejected_enc = tokenizer(
        rejected_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }


def main():
    # ----------------------------
    # 1. Load dataset
    # ----------------------------
    print("Loading Anthropic HH-RLHF dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"Raw dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # ----------------------------
    # 2. Filter invalid / empty pairs
    # ----------------------------
    print("\nFiltering invalid or empty chosen/rejected pairs...")
    ds = ds.filter(filter_valid_examples)
    print(f"Dataset size after filtering: {len(ds)}")

    # ----------------------------
    # 3. Train / validation split
    # ----------------------------
    print("\nCreating train/validation split...")
    ds_splits: DatasetDict = ds.train_test_split(
        test_size=VAL_FRACTION,
        seed=SEED,
    )

    train_ds = ds_splits["train"]
    val_ds = ds_splits["test"]

    print(f"Train size: {len(train_ds)}")
    print(f"Validation size: {len(val_ds)}")

    # ----------------------------
    # 4. Load tokenizer
    # ----------------------------
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default, so map pad to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Using MAX_LENGTH = {MAX_LENGTH}")

    # ----------------------------
    # 5. Apply tokenization
    # ----------------------------
    print("\nTokenizing train split...")
    cols_to_remove = train_ds.column_names  # we will replace with tokenized columns

    train_ds_tokenized = train_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )

    print("Tokenizing validation split...")
    val_ds_tokenized = val_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )

    print("\nTokenized train columns:", train_ds_tokenized.column_names)
    print("Tokenized val columns:", val_ds_tokenized.column_names)

    # Optional: trivial sanity check count
    train_lengths = [int(np.array(x).sum() > 0) for x in train_ds_tokenized["chosen_input_ids"]]
    print(f"\nTokenization complete. Number of train examples: {len(train_lengths)}")

    # ----------------------------
    # 6. Save to disk (optional)
    # ----------------------------
    if SAVE_TO_DISK:
        print(f"\nSaving tokenized train dataset to: {TRAIN_SAVE_PATH}")
        train_ds_tokenized.save_to_disk(TRAIN_SAVE_PATH)

        print(f"Saving tokenized validation dataset to: {VAL_SAVE_PATH}")
        val_ds_tokenized.save_to_disk(VAL_SAVE_PATH)

    print("\n=== Preprocessing (Task B) complete ===")


if __name__ == "__main__":
    main()
