---
language: en
license: apache-2.0
tags:
  - reinforcement-learning
  - grpo
  - life-planning
  - openenv
  - qwen2.5
  - lora
  - curriculum-learning
  - trl
  - unsloth
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
  - generated
pipeline_tag: text-generation
---

<div align="center">

# 🪐 LifeStack-GRPO (v1)

### Qwen2.5-1.5B fine-tuned with 5-stage GRPO curriculum for life crisis resolution

**Built for the Meta × HuggingFace PyTorch OpenEnv Hackathon 2026**
**Team BholeChature — Scaler School of Technology, Bangalore**

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-black?style=flat-square&logo=github)](https://github.com/oki-dokii/Meta-R2)
[![HF Space](https://img.shields.io/badge/HF_Space-Live_Demo-yellow?style=flat-square)](https://huggingface.co/spaces/jdsb06/meta-r2)
[![v4 Model](https://img.shields.io/badge/Latest-lifestack--grpo--v4-green?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo-v4)

</div>

---

> **Note:** This is v1 — the single-step curriculum checkpoint. For the best model, see [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) (episodic GRPO, peak reward 0.856).

---

## What is This?

This is a **LoRA adapter** for `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned using **GRPO** via TRL + Unsloth on the **LifeStack** environment — an OpenEnv-compatible RL environment that models life as a 40-edge dependency graph across 8 domains.

**v1 is the single-step checkpoint.** It was trained with a 5-stage difficulty curriculum and is the most *consistent* checkpoint — 45/50 episodes produced non-failing outputs. The later v4 model adds multi-step episodic planning capability.

---

## Model Details

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter type | LoRA (r=16, alpha=16) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 / 1,562,179,072 (**1.18%**) |
| Training hardware | Tesla T4 (14.7 GB VRAM) |
| Precision | FP16 |
| Framework | TRL + Unsloth 2026.4.8 + Transformers 5.5.0 |

---

## Training: 5-Stage GRPO Curriculum

| Stage | Reward Functions | LR | Steps |
|---|---|---|---|
| 1 | `format_fn` + `clean_eos_fn` (warm-up only) | 8e-6 | 25 |
| 2 | All 9 reward signals | 5e-6 | 25 |
| 3 | All 9 reward signals | 3e-6 | 25 |
| 4 | All 9 reward signals | 2e-6 | 25 |
| 5 | All 9 reward signals | 1e-6 | 25 |

**Total: 125 optimizer steps** | **~45 minutes on Tesla T4**

### Training Metrics (from `train_run_v1.log`)

| Step | Format mean | Reward mean | KL |
|---|---|---|---|
| 5 (stage 4 start) | 0.075 | −0.568 | 7.3e-6 |
| 10 | 0.184 | −0.446 | 1.2e-5 |
| 15 | 0.145 | −0.479 | 1.5e-5 |
| 20 | 0.160 | −0.463 | 2.0e-5 |
| 25 (stage 4 end) | −0.036 | −0.652 | 2.8e-5 |
| 30 (stage 5 start) | 0.043 | −0.612 | 3.4e-5 |
| 50 (stage 5 end) | **0.195** | **−0.396** | 8.2e-5 |

**Post-training evaluation:** −0.10 avg reward over 50 episodes across all 8 domains.

### 9-Signal Reward Orchestrator

1. **`reward_format_fn`** — JSON validity + required keys + valid action_type
2. **`reward_clean_eos_fn`** — Penalises trailing text after `}`
3. **`reward_plausibility_fn`** — Blocks zero-cost massive metric claims
4. **`reward_task_success_fn`** — Environment simulation: did the action resolve the task?
5. **`reward_milestone_fn`** — Did the action unlock a task milestone?
6. **`reward_replan_fn`** — Did the agent adapt after exogenous events?
7. **`reward_reasoning_fn`** — Does the reasoning text logically justify the action type?
8. **`reward_human_feedback_fn`** — ChromaDB memory: alignment with past successful trajectories
9. **`reward_longterm_fn`** — 7-day γ=0.9 discounted rollout

---

## Known Limitation: `clipped_ratio = 1.0`

The model fills its full 96-token budget on every generation — never emits a natural EOS during training. Mitigated at inference via `_JsonCompleteStopping` and first-object extraction.

---

## How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("jdsb06/lifestack-grpo")
model = PeftModel.from_pretrained(base, "jdsb06/lifestack-grpo")
model.eval()
```

---

## Citation

```bibtex
@misc{lifestack2026,
  title        = {LifeStack: Training AI to Handle Life's Cascading Crises},
  author       = {Team BholeChature, Scaler School of Technology},
  year         = {2026},
  howpublished = {Meta × HuggingFace PyTorch OpenEnv Hackathon 2026},
  url          = {https://github.com/oki-dokii/Meta-R2}
}
```
