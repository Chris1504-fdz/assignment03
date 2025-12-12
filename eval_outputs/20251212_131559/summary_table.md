| Model | Checkpoint | Backbone | Mean reward | Std reward | Mean KL | Std KL | Mean length | WinRate vs BASE (reward-proxy) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | (pretrained) | gpt2-medium | -3.5738 | 2.9071 | 0.0000 | 0.0000 | 62.9 | 0.500 |
| DPO | dpo_policy_epoch1.pt | gpt2-medium | -2.9683 | 3.6386 | 0.9591 | 0.5023 | 63.3 | 0.530 |
