"""
eval_all_policies.py (verbose)

Evaluates:
  - BASE (pretrained)
  - PPO checkpoint (latest)
  - GRPO checkpoint (latest)
  - DPO checkpoint (latest, optionally different backbone)

Outputs (saved to eval_outputs/<run_id>/):
  - summary_table.md + summary_table.csv
  - per_prompt_outputs.jsonl
  - samples_qualitative.txt
  - pairwise_human_judge.jsonl
"""

import os
import re
import glob
import json
import random
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


# ----------------- config -----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_BACKBONE = "gpt2-medium"
DPO_BACKBONE = "gpt2-medium"
REWARD_CHECKPOINT = "reward_model/gpt2_reward_model.pt"

PPO_PATTERN = "models/ppo_policy*.pt"
GRPO_PATTERN = "models/grpo_policy_epoch*.pt"
DPO_PATTERN = "models/dpo_policy_epoch*.pt"

MAX_PROMPT_LEN = 128
MAX_GEN_LEN = 64

EVAL_SPLIT = "test"          # fallback to 'train' if split not found
NUM_EVAL_PROMPTS = 200       # metrics evaluation
EVAL_BATCH_SIZE = 12

NUM_JUDGE_PROMPTS = 200      # for human judging file

TOP_P = 0.9
TEMPERATURE = 1.0
SEED = 42

NUM_SHOW_EXAMPLES = 20
OUT_ROOT = "eval_outputs"

COMPUTE_REWARD_PROXY_WINRATE = True

VERBOSE = True
PRINT_EVERY_BATCH = 5  # during generate_and_score


# ----------------- small helpers -----------------

def vprint(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def dump_jsonl(path: str, rows: List[dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vprint(f"[write] jsonl -> {path}  (rows={len(rows)})")

def write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    vprint(f"[write] text -> {path}  (chars={len(text)})")

def write_csv(path: str, header: List[str], rows: List[List]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    vprint(f"[write] csv -> {path}  (rows={len(rows)})")

def list_checkpoints(pattern: str) -> List[str]:
    paths = glob.glob(pattern)
    if not paths:
        return []

    def extract_epoch(p: str) -> int:
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths.sort(key=extract_epoch)
    return paths

def get_latest_checkpoint(pattern: str) -> Optional[str]:
    ckpts = list_checkpoints(pattern)
    return ckpts[-1] if ckpts else None


# ----------------- prompt cleaning -----------------

def extract_first_human_turn(raw_prompt: str) -> str:
    if raw_prompt is None:
        return ""
    parts = re.split(r"\n\s*Assistant\s*:\s*", raw_prompt, maxsplit=1)
    first_chunk = parts[0].strip()
    first_chunk = re.sub(r"^\s*Human\s*:\s*", "", first_chunk).strip()
    first_chunk = re.sub(r"\s+", " ", first_chunk).strip()
    return first_chunk

def display_first_line(text: str) -> str:
    return (text or "").strip()

def format_hh_prompt(question: str) -> str:
    question = question.strip()
    return f"Human: {question}\n\nAssistant:"

def truncate_at_next_human(text: str) -> str:
    idx = text.find("Human:")
    if idx == -1:
        return text.strip()
    return text[:idx].strip()


# ----------------- reward model -----------------

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
        last_indices = (lengths - 1).clamp(min=0)  # [B]

        B, T, H = hidden_states.size()
        idx = last_indices.view(-1, 1, 1).expand(-1, 1, H)
        last_hidden = hidden_states.gather(dim=1, index=idx).squeeze(1)
        rewards = self.value_head(last_hidden).squeeze(-1)
        return rewards

def infer_reward_backbone_from_state_dict(state: dict) -> str:
    if "value_head.weight" not in state:
        raise ValueError("Checkpoint missing value_head.weight; cannot infer reward backbone.")
    hidden = int(state["value_head.weight"].shape[1])
    mapping = {
        768: "gpt2",
        1024: "gpt2-medium",
        1280: "gpt2-large",
        1600: "gpt2-xl",
    }
    if hidden not in mapping:
        raise ValueError(f"Unknown reward hidden size {hidden}; update mapping.")
    return mapping[hidden]


# ----------------- token logprobs + KL estimate -----------------

@torch.no_grad()
def token_logprobs_for_sequences(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]
    logp = torch.log_softmax(logits, dim=-1)

    targets = input_ids[:, 1:]
    logp = logp[:, :-1, :]
    token_logp = logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_logp

def response_token_mask_from_prompt(attn_prompt: torch.Tensor, attn_full: torch.Tensor) -> torch.Tensor:
    B, T_full = attn_full.size()
    prompt_lens = attn_prompt.sum(dim=1).long()
    mask = torch.zeros((B, T_full - 1), device=attn_full.device, dtype=torch.float32)

    for b in range(B):
        pl = int(prompt_lens[b].item())
        start_t = max(pl - 1, 0)
        valid_t = attn_full[b, 1:].float()
        m = torch.zeros_like(valid_t)
        m[start_t:] = 1.0
        mask[b] = m * valid_t

    return mask


# ----------------- data loading -----------------

def load_eval_questions(n_prompts: int) -> List[str]:
    split_used = None
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split=EVAL_SPLIT)
        split_used = EVAL_SPLIT
    except Exception:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
        split_used = "train"

    raw_prompts = ds["prompt"] if "prompt" in ds.column_names else ds["chosen"]

    if len(raw_prompts) > n_prompts:
        idx = random.sample(range(len(raw_prompts)), n_prompts)
        raw_prompts = [raw_prompts[i] for i in idx]

    questions = [extract_first_human_turn(p) for p in raw_prompts]
    vprint(f"[data] Loaded {len(questions)} prompts from split='{split_used}'")
    return questions


# ----------------- generation + scoring -----------------

@dataclass
class ModelRun:
    name: str
    backbone: str
    checkpoint: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    ref_model: AutoModelForCausalLM  # for KL

@torch.no_grad()
def generate_and_score(
    run: ModelRun,
    reward_model: GPT2RewardModel,
    reward_tokenizer: AutoTokenizer,
    questions: List[str],
) -> List[dict]:
    run.model.eval()
    run.ref_model.eval()
    reward_model.eval()

    vprint(f"\n[eval] Model={run.name} | backbone={run.backbone} | ckpt={run.checkpoint}")
    vprint(f"[eval] Prompts={len(questions)} | batch_size={EVAL_BATCH_SIZE} | max_prompt_len={MAX_PROMPT_LEN} | max_gen_len={MAX_GEN_LEN}")

    rows: List[dict] = []
    n_batches = (len(questions) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

    for bidx, start in enumerate(range(0, len(questions), EVAL_BATCH_SIZE), start=1):
        batch_q = questions[start:start + EVAL_BATCH_SIZE]
        batch_prompts = [format_hh_prompt(q) for q in batch_q]

        enc = run.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn_prompt = enc["attention_mask"].to(DEVICE)

        gen_ids = run.model.generate(
            input_ids=input_ids,
            attention_mask=attn_prompt,
            max_new_tokens=MAX_GEN_LEN,
            do_sample=True,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            pad_token_id=run.tokenizer.eos_token_id,
        )

        attn_full = (gen_ids != run.tokenizer.pad_token_id).long()

        # KL on response tokens only
        pi_tok_logp = token_logprobs_for_sequences(run.model, gen_ids, attn_full)
        ref_tok_logp = token_logprobs_for_sequences(run.ref_model, gen_ids, attn_full)
        resp_mask = response_token_mask_from_prompt(attn_prompt, attn_full)

        denom = resp_mask.sum(dim=1).clamp(min=1.0)
        kl_seq = ((pi_tok_logp - ref_tok_logp) * resp_mask).sum(dim=1) / denom

        # decode full text for reward scoring
        full_texts = run.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        r_enc = reward_tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN + MAX_GEN_LEN,
            return_tensors="pt",
        )
        r_input_ids = r_enc["input_ids"].to(DEVICE)
        r_attn = r_enc["attention_mask"].to(DEVICE)
        rewards = reward_model(r_input_ids, r_attn)

        prompt_lens = attn_prompt.sum(dim=1).tolist()
        total_lens = attn_full.sum(dim=1).tolist()

        for i in range(len(batch_q)):
            pl = int(prompt_lens[i])
            tl = int(total_lens[i])
            resp_len = max(tl - pl, 0)

            new_tokens = gen_ids[i, pl:pl + resp_len]
            completion = run.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completion = truncate_at_next_human(completion)

            rows.append({
                "model": run.name,
                "backbone": run.backbone,
                "checkpoint": run.checkpoint,
                "question": batch_q[i],
                "question_display": display_first_line(batch_q[i]),
                "completion": completion,
                "reward": float(rewards[i].item()),
                "kl": float(kl_seq[i].item()),
                "prompt_len_tokens": pl,
                "response_len_tokens": resp_len,
                "total_len_tokens": tl,
            })

        if (bidx % PRINT_EVERY_BATCH == 0) or (bidx == 1) or (bidx == n_batches):
            vprint(f"[eval] {run.name}: batch {bidx}/{n_batches} done (rows={len(rows)})")

    vprint(f"[eval] Done {run.name}: collected rows={len(rows)}")
    return rows


def summarize(rows: List[dict]) -> Dict[str, float]:
    r = torch.tensor([x["reward"] for x in rows], dtype=torch.float32)
    k = torch.tensor([x["kl"] for x in rows], dtype=torch.float32)
    l = torch.tensor([x["response_len_tokens"] for x in rows], dtype=torch.float32)

    return {
        "mean_reward": float(r.mean().item()),
        "std_reward": float(r.std(unbiased=False).item()),
        "mean_kl": float(k.mean().item()),
        "std_kl": float(k.std(unbiased=False).item()),
        "mean_length": float(l.mean().item()),
        "std_length": float(l.std(unbiased=False).item()),
    }

def reward_proxy_winrate(model_rows: List[dict], base_rows: List[dict]) -> float:
    wins = 0
    for a, b in zip(model_rows, base_rows):
        if a["reward"] > b["reward"]:
            wins += 1
    return wins / max(1, len(base_rows))


# ----------------- main -----------------

def main():
    set_seed(SEED)

    vprint("=" * 80)
    vprint(f"[start] eval_all_policies.py | device={DEVICE}")
    vprint(f"[start] seed={SEED} | top_p={TOP_P} | temp={TEMPERATURE}")
    vprint(f"[start] base_backbone={BASE_BACKBONE} | dpo_backbone={DPO_BACKBONE}")
    vprint(f"[start] reward_ckpt={REWARD_CHECKPOINT}")
    vprint("=" * 80)

    run_id = now_id()
    out_dir = os.path.join(OUT_ROOT, run_id)
    ensure_dir(out_dir)
    vprint(f"[io] output_dir={out_dir}")

    # Checkpoints discovery
    vprint("\n[ckpt] Searching checkpoints:")
    ppo_ckpt = get_latest_checkpoint(PPO_PATTERN)
    grpo_ckpt = get_latest_checkpoint(GRPO_PATTERN)
    dpo_ckpt = get_latest_checkpoint(DPO_PATTERN)
    vprint(f"[ckpt] PPO pattern='{PPO_PATTERN}' -> {ppo_ckpt or 'NONE'}")
    vprint(f"[ckpt] GRPO pattern='{GRPO_PATTERN}' -> {grpo_ckpt or 'NONE'}")
    vprint(f"[ckpt] DPO pattern='{DPO_PATTERN}' -> {dpo_ckpt or 'NONE'}")

    # Load evaluation prompts
    vprint("\n[data] Loading evaluation prompts...")
    eval_questions = load_eval_questions(NUM_EVAL_PROMPTS)

    vprint("[data] Loading judge prompts (for human labeling file)...")
    judge_questions = load_eval_questions(NUM_JUDGE_PROMPTS)

    # Tokenizers
    vprint("\n[tokenizer] Loading tokenizers...")
    base_tok = AutoTokenizer.from_pretrained(BASE_BACKBONE)
    base_tok.padding_side = "left"
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    dpo_tok = AutoTokenizer.from_pretrained(DPO_BACKBONE)
    dpo_tok.padding_side = "left"
    if dpo_tok.pad_token is None:
        dpo_tok.pad_token = dpo_tok.eos_token

    # Reward model: auto-infer backbone
    vprint("\n[reward] Loading reward checkpoint and inferring backbone...")
    state = torch.load(REWARD_CHECKPOINT, map_location="cpu")
    reward_backbone = infer_reward_backbone_from_state_dict(state)
    vprint(f"[reward] inferred backbone='{reward_backbone}'")

    reward_tok = AutoTokenizer.from_pretrained(reward_backbone)
    reward_tok.padding_side = "left"
    if reward_tok.pad_token is None:
        reward_tok.pad_token = reward_tok.eos_token

    reward_model = GPT2RewardModel(reward_backbone).to(DEVICE)
    reward_model.load_state_dict(torch.load(REWARD_CHECKPOINT, map_location=DEVICE))
    reward_model.eval()
    vprint("[reward] reward model loaded")

    # Load base + refs
    vprint("\n[models] Loading base / refs...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_BACKBONE).to(DEVICE)
    base_ref = AutoModelForCausalLM.from_pretrained(BASE_BACKBONE).to(DEVICE)
    dpo_ref = AutoModelForCausalLM.from_pretrained(DPO_BACKBONE).to(DEVICE)

    # Runs list
    runs: List[ModelRun] = []
    runs.append(ModelRun(
        name="BASE",
        backbone=BASE_BACKBONE,
        checkpoint="(pretrained)",
        model=base_model,
        tokenizer=base_tok,
        ref_model=base_model,  # KL=0 baseline
    ))

    if ppo_ckpt:
        vprint("[models] Loading PPO checkpoint...")
        ppo_model = AutoModelForCausalLM.from_pretrained(BASE_BACKBONE).to(DEVICE)
        ppo_model.load_state_dict(torch.load(ppo_ckpt, map_location=DEVICE))
        runs.append(ModelRun(
            name="PPO",
            backbone=BASE_BACKBONE,
            checkpoint=os.path.basename(ppo_ckpt),
            model=ppo_model,
            tokenizer=base_tok,
            ref_model=base_ref,
        ))

    if grpo_ckpt:
        vprint("[models] Loading GRPO checkpoint...")
        grpo_model = AutoModelForCausalLM.from_pretrained(BASE_BACKBONE).to(DEVICE)
        grpo_model.load_state_dict(torch.load(grpo_ckpt, map_location=DEVICE))
        runs.append(ModelRun(
            name="GRPO",
            backbone=BASE_BACKBONE,
            checkpoint=os.path.basename(grpo_ckpt),
            model=grpo_model,
            tokenizer=base_tok,
            ref_model=base_ref,
        ))

    if dpo_ckpt:
        vprint("[models] Loading DPO checkpoint...")
        dpo_model = AutoModelForCausalLM.from_pretrained(DPO_BACKBONE).to(DEVICE)
        dpo_model.load_state_dict(torch.load(dpo_ckpt, map_location=DEVICE))
        runs.append(ModelRun(
            name="DPO",
            backbone=DPO_BACKBONE,
            checkpoint=os.path.basename(dpo_ckpt),
            model=dpo_model,
            tokenizer=dpo_tok,
            ref_model=dpo_ref,
        ))

    vprint("\n[plan] Models to evaluate:")
    for r in runs:
        vprint(f"  - {r.name:4s} | backbone={r.backbone:10s} | ckpt={r.checkpoint}")

    # Evaluate metrics on eval_questions
    vprint("\n[run] Evaluating metrics on eval_questions...")
    all_rows_by_model: Dict[str, List[dict]] = {}
    for run in runs:
        rows = generate_and_score(run, reward_model, reward_tok, eval_questions)
        all_rows_by_model[run.name] = rows

    # Save per-prompt outputs
    vprint("\n[save] Writing per-prompt outputs...")
    per_prompt_path = os.path.join(out_dir, "per_prompt_outputs.jsonl")
    merged_rows: List[dict] = []
    for _, rows in all_rows_by_model.items():
        merged_rows.extend(rows)
    dump_jsonl(per_prompt_path, merged_rows)

    # Summary table
    vprint("\n[summary] Computing summary table...")
    summary_rows = []
    base_rows = all_rows_by_model["BASE"]

    for run in runs:
        rows = all_rows_by_model[run.name]
        s = summarize(rows)

        win = None
        if COMPUTE_REWARD_PROXY_WINRATE:
            win = 0.5 if run.name == "BASE" else reward_proxy_winrate(rows, base_rows)

        summary_rows.append({
            "Model": run.name,
            "Checkpoint": run.checkpoint,
            "Backbone": run.backbone,
            "Mean reward": s["mean_reward"],
            "Std reward": s["std_reward"],
            "Mean KL": s["mean_kl"] if run.name != "BASE" else 0.0,
            "Std KL": s["std_kl"] if run.name != "BASE" else 0.0,
            "Mean length (response tokens)": s["mean_length"],
            "WinRate vs BASE (reward-proxy)": win,
        })

    md_lines = []
    md_lines.append("| Model | Checkpoint | Backbone | Mean reward | Std reward | Mean KL | Std KL | Mean length | WinRate vs BASE (reward-proxy) |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        md_lines.append(
            f"| {r['Model']} | {r['Checkpoint']} | {r['Backbone']} | "
            f"{r['Mean reward']:.4f} | {r['Std reward']:.4f} | "
            f"{r['Mean KL']:.4f} | {r['Std KL']:.4f} | "
            f"{r['Mean length (response tokens)']:.1f} | "
            f"{(r['WinRate vs BASE (reward-proxy)'] if r['WinRate vs BASE (reward-proxy)'] is not None else 'NA'):.3f} |"
        )

    summary_md = "\n".join(md_lines) + "\n"
    write_text(os.path.join(out_dir, "summary_table.md"), summary_md)

    csv_header = [
        "Model", "Checkpoint", "Backbone",
        "Mean reward", "Std reward", "Mean KL", "Std KL",
        "Mean length (response tokens)", "WinRate vs BASE (reward-proxy)"
    ]
    csv_rows = []
    for r in summary_rows:
        csv_rows.append([
            r["Model"], r["Checkpoint"], r["Backbone"],
            f"{r['Mean reward']:.6f}", f"{r['Std reward']:.6f}",
            f"{r['Mean KL']:.6f}", f"{r['Std KL']:.6f}",
            f"{r['Mean length (response tokens)']:.3f}",
            "" if r["WinRate vs BASE (reward-proxy)"] is None else f"{r['WinRate vs BASE (reward-proxy)']:.6f}"
        ])
    write_csv(os.path.join(out_dir, "summary_table.csv"), csv_header, csv_rows)

    # Qualitative examples file (do not print to terminal)
    vprint("\n[qual] Writing qualitative samples...")
    n_show = min(NUM_SHOW_EXAMPLES, len(eval_questions))
    sample_questions = eval_questions[:n_show]

    qualitative_lines = []
    for i in range(n_show):
        q_disp = display_first_line(sample_questions[i])
        qualitative_lines.append("=" * 90)
        qualitative_lines.append(f"Example {i+1}")
        qualitative_lines.append("-" * 90)
        qualitative_lines.append("PROMPT:")
        qualitative_lines.append(q_disp)
        qualitative_lines.append("")
        for run in runs:
            row = all_rows_by_model[run.name][i]
            qualitative_lines.append(f"[{run.name}]")
            qualitative_lines.append(row["completion"] if row["completion"] else "(empty)")
            qualitative_lines.append("")
    write_text(os.path.join(out_dir, "samples_qualitative.txt"), "\n".join(qualitative_lines) + "\n")

    # Pairwise human judging file
    vprint("\n[judge] Regenerating completions for human judging set...")
    judge_rows_by_model: Dict[str, List[dict]] = {}
    for run in runs:
        judge_rows_by_model[run.name] = generate_and_score(run, reward_model, reward_tok, judge_questions)

    pairwise = []
    for i, q in enumerate(judge_questions):
        base_comp = judge_rows_by_model["BASE"][i]["completion"]
        for run in runs:
            if run.name == "BASE":
                continue
            model_comp = judge_rows_by_model[run.name][i]["completion"]
            pairwise.append({
                "prompt": display_first_line(q),
                "base_model": "BASE",
                "model": run.name,
                "base_completion": base_comp,
                "model_completion": model_comp,
                "winner": None,
            })
    dump_jsonl(os.path.join(out_dir, "pairwise_human_judge.jsonl"), pairwise)

    # Final terminal summary (keep it short but includes the table)
    vprint("\n" + "=" * 80)
    vprint(f"[done] Saved evaluation artifacts to: {out_dir}")
    vprint(f"  - summary_table.md / summary_table.csv")
    vprint(f"  - per_prompt_outputs.jsonl")
    vprint(f"  - samples_qualitative.txt (NUM_SHOW_EXAMPLES={NUM_SHOW_EXAMPLES})")
    vprint(f"  - pairwise_human_judge.jsonl (NUM_JUDGE_PROMPTS={NUM_JUDGE_PROMPTS})")
    vprint("=" * 80)
    print("\nMarkdown table:\n")
    print(summary_md)


if __name__ == "__main__":
    main()
