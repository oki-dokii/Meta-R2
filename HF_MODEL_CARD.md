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
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
  - generated
pipeline_tag: text-generation
---

<div align="center">

# 🪐 LifeStack-GRPO

### Qwen2.5-1.5B fine-tuned with GRPO to resolve multi-domain life crises

**Built for the Meta × HuggingFace PyTorch OpenEnv Hackathon 2026**
**Team BholeChature — Scaler School of Technology, Bangalore**

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-black?style=flat-square&logo=github)](https://github.com/oki-dokii/Meta-R2)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue?style=flat-square)](https://github.com/facebookresearch/openenv)

</div>

---

## What is this?

This is a **LoRA adapter** for `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned using **Group Relative Policy Optimization (GRPO)** via TRL + Unsloth on the **LifeStack** environment — an OpenEnv-compatible RL environment that models life as a 40-edge dependency graph across 6 domains.

Given a structured life crisis prompt (flight cancelled, boss moved a deadline, card declined — simultaneously), the model outputs a **single JSON action plan** specifying what to do, where to focus, and why.

**Example output:**
```json
{
  "action_type": "negotiate",
  "target_domain": "career",
  "metric_changes": {"career.workload": -10.0, "mental_wellbeing.stress_level": -5.0},
  "resource_cost": {"time": 0.5, "money": 0.0, "energy": 5.0},
  "reasoning": "Boss extension request reduces cascade before financial fix"
}
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter type | LoRA (r=16, alpha=16) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 / 1,562,179,072 (**1.18%**) |
| Adapter size | ~74 MB (`adapter_model.safetensors`) |
| Training hardware | Tesla T4 (14.7 GB VRAM) |
| Precision | FP16 (T4 / compute 7.5) |
| Framework | TRL + Unsloth 2026.4.8 + Transformers 5.5.0 |

---

## Training: 5-Stage GRPO Curriculum

Training used a **difficulty curriculum** — stages start with easy single-domain conflicts (difficulty 1) and progress to hard multi-domain cascades (difficulty 5).

### Stage Architecture

| Stage | Reward Functions | LR | Steps |
|---|---|---|---|
| 1 | `format_fn` + `clean_eos_fn` (warm-up) | 8e-6 | 25 |
| 2 | All 9 reward signals | 5e-6 | 25 |
| 3 | All 9 reward signals | 3e-6 | 25 |
| 4 | All 9 reward signals | 2e-6 | 25 |
| 5 | All 9 reward signals | 1e-6 | 25 |

**Total: 125 optimizer steps** | **~45 minutes on Tesla T4**

### 9-Signal Reward Orchestrator

The reward system uses **9 non-overlapping signals** to prevent reward hacking:

1. **`reward_format_fn`** — JSON validity + required keys + valid action type (+1.0 / +0.5 / −0.5)
2. **`reward_clean_eos_fn`** — Penalises trailing text after `}` (+0.20 / −0.10)
3. **`reward_plausibility_fn`** — Blocks zero-cost massive metric claims (−0.30)
4. **`reward_task_success_fn`** — Environment simulation: did the action resolve the task?
5. **`reward_milestone_fn`** — Did the action unlock a task milestone?
6. **`reward_replan_fn`** — Did the agent adapt after exogenous events?
7. **`reward_reasoning_fn`** — Does the reasoning text logically justify the action type?
8. **`reward_human_feedback_fn`** — ChromaDB memory: alignment with past successful trajectories
9. **`reward_longterm_fn`** — 7-day γ=0.9 discounted rollout (what happens over the next week?)

### Training Metrics

| Step (cumulative) | Format mean | Reward mean | KL |
|---|---|---|---|
| 25 (end stage 4 step 5) | 0.075 | −0.568 | 7.3e-6 |
| 75 (mid stage 4) | 0.145 | −0.446 | 1.5e-5 |
| 125 (end stage 4) | −0.036 | −0.652 | 2.8e-5 |
| 150 (mid stage 5) | 0.117 | −0.481 | 3.8e-5 |
| 225 (end stage 5) | 0.195 | −0.396 | 8.2e-5 |

**Post-training evaluation:** −0.10 avg reward over 50 episodes across all 8 domains.

### Known Limitation: `clipped_ratio = 1.0`

The model fills its full 96-token completion budget on every generation — it never emits a natural EOS token during training. This is a known GRPO challenge when the model has a deeply ingrained tendency from earlier stages to produce long outputs. At **inference time** this is fully mitigated: `_JsonCompleteStopping` halts generation when the first `{...}` closes, and `_load_first_json_object` extracts only the first valid JSON. The user always receives a clean, compact action plan.

---

## How to Load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
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

Or with Unsloth (2× faster):
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="jdsb06/lifestack-grpo",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

---

## How to Run Inference

```python
prompt = """You are LifeStack. Return ONLY compact JSON.
Task: Survive Airport Cancellation
Key metrics:
- career.workload: 85.0
- finances.liquidity: 30.0
- mental_wellbeing.stress_level: 75.0
Budget: time=10.0, money=200.0, energy=40.0
Required keys: action_type, target_domain, metric_changes, resource_cost, reasoning.
Keep reasoning under 25 words. No markdown."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=96, temperature=0.3, do_sample=True)

completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(completion)
```

Or clone the repo and run the full Gradio demo:
```bash
git clone https://github.com/oki-dokii/Meta-R2
cd Meta-R2
pip install -r requirements.txt
python scripts/gradio_demo.py --model-dir ./lifestack_model
```

---

## The LifeStack Environment

This model was trained inside **LifeStack**, an OpenEnv-compatible environment that models life as a 40-edge directed dependency graph:

- **6 domains**: Career, Finances, Relationships, Physical Health, Mental Wellbeing, Time
- **23 sub-metrics** with baseline values and cascade dampening (0.6 per hop, from Starcke & Brand 2012)
- **8 task domains**: career, finances, relationships, physical_health, mental_wellbeing, time, transport_crisis (5 modes), code_merge_crisis
- **Exogenous events**: probabilistic mid-episode disruptions (ticket surge, lounge closure, etc.)
- **Route system**: multiple viable paths with preconditions and mutual exclusion
- **7-day rollout**: `env.rollout(n_steps=7, gamma=0.9)` for long-horizon reward signal

```python
from core.lifestack_env import LifeStackEnv, LifeStackAction

env = LifeStackEnv()
obs = env.reset(task=task)
action = LifeStackAction(
    action_type="negotiate",
    target="career",
    metric_changes={"career.workload": -15.0},
    resource_cost={"time": 0.5, "money": 0.0, "energy": 5.0},
    actions_taken=1
)
obs = env.step(action)
print(obs.reward)  # float in [-1, 1]
```

---

## Research Grounding

| Principle | Source | Implementation |
|---|---|---|
| Stress cascade dampening (0.6/hop) | Starcke & Brand (2012) | `DependencyGraph.cascade()` |
| Scarcity bandwidth tax | Mullainathan & Shafir (2013) | Budget depletion blocks actions |
| Multi-objective reward | Roijers et al. (2013) | 9 non-overlapping signals |
| Retrieval-augmented moderation | RAM / RAG | ChromaDB `LifeStackMemory` |

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
