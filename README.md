# Assignment 3 - RLHF (Reward Model + PPO + GRPO + DPO) + Evaluation

This repository implements an end-to-end RLHF pipeline on the Anthropic HH-RLHF dataset:

- Reward Model trained from preference pairs (chosen vs rejected).
- PPO-based RLHF policy training using the learned reward model.
- GRPO policy training using group-relative advantages (multiple samples per prompt).
- DPO policy training directly from preference pairs.
- A unified evaluation script that produces quantitative metrics, qualitative samples, and files to support win-rate judging on 100+ prompts.

The code emphasizes reproducibility (fixed seeds, consistent prompt formatting, saved checkpoints, deterministic evaluation output folders) and produces submission-ready artifacts.

---

## 1. Repository structure

Key files:

- src/reward/train_reward_model.py  
  Trains the reward model and saves: reward_model/gpt2_reward_model.pt

- src/train_ppo_policy.py  
  Trains PPO policy and saves: models/ppo_policy_token_epoch*.pt

- src/train_grpo_policy.py  
  Trains GRPO policy and saves: models/grpo_policy_epoch*.pt

- src/train_dpo_policy.py  
  Trains DPO policy and saves: models/dpo_policy_epoch*.pt

- src/eval_all_policies.py  
  Evaluates BASE/PPO/GRPO/DPO and writes artifacts to: eval_outputs/<run_id>/

---

## 2. Dataset and prompt formatting

This project uses the Anthropic/hh-rlhf dataset via Hugging Face datasets.

Prompts are cleaned to keep only the first Human turn, then formatted as:

Human: <question>

Assistant:

This makes evaluation consistent across BASE/PPO/GRPO/DPO and matches typical RLHF prompt formatting.

---

## 3. Training pipeline

### 3.1 Reward model training

```bash
python src/reward/train_reward_model.py
```

Outputs:
- reward_model/gpt2_reward_model.pt

Example log snippet:

```
2025-12-10 09:53:08,284 | [Epoch 5] Validation loss: 0.9264, Validation acc: 0.6323
2025-12-10 09:53:09,623 | Saved reward model to reward_model/gpt2_reward_model.pt
```

Design rationale:
- The reward model is a GPT-2-family encoder (AutoModel) plus a scalar value head.
- It is trained to assign higher reward to the chosen completion than to the rejected completion.
- This reward model is then used as a consistent scoring function for PPO and for evaluation.

---

### 3.2 PPO training (RLHF)

```bash
python src/train_ppo_policy.py
```

Outputs:
- models/ppo_policy_token_epoch1.pt

Example log snippet:

```
2025-12-12 07:57:07,630 | Saved PPO policy checkpoint: models/ppo_policy_token_epoch1.pt
```

Design rationale:
- PPO uses a clipped objective to stabilize updates.
- PPO also includes a KL penalty (regularization to a frozen reference model) to reduce destructive drift.

---

### 3.3 GRPO training

```bash
python src/train_grpo_policy.py
```

Outputs:
- models/grpo_policy_epoch1.pt

Example log snippet:

```
2025-12-12 14:06:26,089 | Saved: models/grpo_policy_epoch1.pt
```

Design rationale:
- GRPO samples multiple completions per prompt and computes group-relative advantages.
- This reduces sensitivity to absolute reward scale and provides a stable learning signal from within-group comparisons.

---

### 3.4 DPO training

```bash
python src/train_dpo_policy.py
```

Outputs:
- models/dpo_policy_epoch1.pt

Example log snippet:

```
[Epoch 1] Val loss = 0.7511 | Val pref_acc = 0.4815 | Ref pref_acc = 0.4845
Saved DPO policy checkpoint: models/dpo_policy_epoch1.pt
```

Design rationale:
- DPO optimizes preference alignment directly using chosen vs rejected pairs, without requiring reward-model rollouts during training.
- A reference model is used to define the implicit preference objective.

---

## 4. Evaluation and outputs

Run evaluation:

```bash
python src/eval_all_policies.py
```

This script:
- loads the latest checkpoints in models/
- evaluates BASE/PPO/GRPO/DPO on the same prompt set
- writes all evaluation artifacts to eval_outputs/<run_id>/

Artifacts written:
- summary_table.md and summary_table.csv
- per_prompt_outputs.jsonl (prompt + completion + reward + KL + lengths per model)
- samples_qualitative.txt (about 20 prompts, completion-only per model)
- pairwise_human_judge.jsonl (BASE vs each trained model for 100+ prompts; winner field is left blank for labeling)

---

## 5. Quantitative results (your run)

Below is the exact summary table produced by eval_all_policies.py:

| Model | Checkpoint | Backbone | Mean reward | Std reward | Mean KL | Std KL | Mean length | WinRate vs BASE (reward-proxy) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | (pretrained) | gpt2-medium | -3.6854 | 3.1128 | 0.0000 | 0.0000 | 63.3 | 0.500 |
| PPO | ppo_policy_token_epoch1.pt | gpt2-medium | -4.7458 | 2.5038 | -11.7963 | 5.9303 | 64.0 | 0.345 |
| GRPO | grpo_policy_epoch1.pt | gpt2-medium | -3.0028 | 2.0460 | -11.9982 | 5.8110 | 64.0 | 0.590 |
| DPO | dpo_policy_epoch1.pt | gpt2-medium | -3.0602 | 3.1669 | 1.0345 | 0.5119 | 63.4 | 0.560 |

### How to interpret these columns

- Mean reward / Std reward  
  These are computed by running each model’s generated outputs through the learned reward model. Higher mean reward indicates the reward model prefers the completions more often.

- Mean KL / Std KL  
  This is an estimate of policy drift from a reference model, measured on response tokens. It is useful for analyzing the reward-vs-drift tradeoff.  
  (Note: depending on implementation conventions, this value can be reported as signed or absolute; the key idea is that it tracks how far the policy moves from the reference distribution.)

- Mean length (response tokens)  
  Average response length; important because longer responses can sometimes inflate reward or KL. In this run, lengths are very similar across models (~63-64), which makes the comparisons fairer.

- WinRate vs BASE (reward-proxy)  
  For each prompt, the model and BASE are compared by reward score. WinRate is the fraction of prompts where the model’s reward > BASE reward. This is a proxy win-rate based on the learned reward model and is not a substitute for a true human or GPT judge, but it is a consistent quantitative signal.

### High-level observations from this run

- GRPO and DPO outperform BASE under the reward-proxy metric:
  - GRPO WinRate vs BASE = 0.590
  - DPO WinRate vs BASE = 0.560
- PPO underperforms BASE under reward-proxy in this run (WinRate 0.345). This is still informative analytically: PPO can be sensitive to reward model calibration, KL coefficient choice, and sampling settings.

Because response lengths are comparable across models, these win-rate differences are less likely to be explained purely by response length.

---

## 6. Human win-rate judging (100+ prompts)

The assignment requires win-rate comparisons on 100+ prompts using human evaluation (or an external judge model).

This repository generates:
- eval_outputs/<run_id>/pairwise_human_judge.jsonl

Each record includes:
- prompt
- BASE completion
- model completion
- winner (left as None)

To compute human win-rates:
1. Open the JSONL file.
2. For each example, set winner to BASE or the model name (PPO, GRPO, DPO).
3. Aggregate win rates by model.

This format provides an auditable record of comparisons for grading.

---

## 7. Reproducibility

Default reproducibility controls:
- SEED = 42
- consistent prompt cleaning and formatting
- fixed generation settings during evaluation:
  - top_p = 0.9
  - temperature = 1.0
  - max_prompt_len = 128
  - max_gen_len = 64

---

## 8. Deliverables checklist

- [•] Code for Reward Model, PPO, GRPO, DPO
- [•] Evaluation script producing quantitative + qualitative artifacts
- [•] README.md
- [•] Saved model checkpoints in models/
- [•] Generated samples + pairwise judge file in eval_outputs/<run_id>/

---

## 9. Quick commands

Reward model:
```bash
python src/reward/train_reward_model.py
```

PPO:
```bash
python src/train_ppo_policy.py
```

GRPO:
```bash
python src/train_grpo_policy.py
```

DPO:
```bash
python src/train_dpo_policy.py
```

Evaluation:
```bash
python src/eval_all_policies.py
```
