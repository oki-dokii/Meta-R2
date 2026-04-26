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

# 🪐 LifeStack-GRPO-v4

### Qwen2.5-1.5B fine-tuned with episodic GRPO to resolve multi-domain life crises

**Built for the Meta × HuggingFace PyTorch OpenEnv Hackathon 2026**
**Team BholeChature — Scaler School of Technology, Bangalore**

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-black?style=flat-square&logo=github)](https://github.com/oki-dokii/Meta-R2)
[![HF Space](https://img.shields.io/badge/HF_Space-Live_Demo-yellow?style=flat-square)](https://huggingface.co/spaces/jdsb06/meta-r2)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)
[![Colab](https://img.shields.io/badge/Colab-Re--run_Training-orange?style=flat-square&logo=google-colab)](https://github.com/oki-dokii/Meta-R2/blob/main/notebooks/Colab_GRPO_Training.ipynb)

</div>

---

## The Friday 6:00 PM Problem

It's Friday evening. Your flight home was just cancelled. You open your banking app to rebook — your card is declined due to a "security flag." Simultaneously, a Slack notification: your boss moved Monday's 9:00 AM deadline to **Sunday afternoon**. You have $200 in cash, five hours of usable energy, and four different people expecting you in different places.

You turn to your AI assistant. It finds a cheaper flight — a 12-hour layover that kills your weekend. You ask it to message your boss, but the tone sounds defensive, triggering a "clarification" meeting that eats more of your time. Every "solution" applied in isolation creates a new wound elsewhere.

This is a **Life Problem** — cascading, interconnected, resource-constrained. No existing AI environment was built to handle it. **We built LifeStack.**

---

## What is This Model?

This is **v4** of our LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct`, fine-tuned using **episodic GRPO** (Group Relative Policy Optimization) via TRL 0.15.1 + Unsloth 2026.4.8 on the **LifeStack** environment — an OpenEnv-compatible RL environment with **23 interdependent metrics across 6 life-metric domains**, connected by a **32-edge directed dependency graph**. Training tasks are sampled across 8 task domains (the 6 life domains + `transport_crisis` + `code_merge_crisis`).

Given a structured life crisis, the model outputs a **single JSON action plan**:

```json
{
  "action_type": "negotiate",
  "target_domain": "career",
  "metric_changes": {"career.workload": -10.0, "mental_wellbeing.stress_level": -5.0},
  "resource_cost": {"time": 0.5, "money": 0.0, "energy": 5.0},
  "reasoning": "Request deadline extension before financial situation worsens"
}
```

**v4 is the active production model** powering the [live HF Space demo](https://huggingface.co/spaces/jdsb06/meta-r2).

---

## Training Evidence

### Reward Curve (v4 — real run from `train_run_v4.log`)
![Reward Curve](plots/reward_curve.png)

### Loss Curve
![Loss Curve](plots/loss_curve.png)

### Per-Function Reward Components
![Reward Components](plots/reward_components.png)

### 4-Panel Training Summary
![Training Summary](plots/training_summary.png)

**Raw training log:** [`train_run_v4.log`](https://huggingface.co/jdsb06/lifestack-grpo-v4/blob/main/train_run_v4.log) (109 kB — full Unsloth/TRL output)

---

## v4 Key Training Metrics (Stage 3 final)

| Metric | Value |
|--------|-------|
| Peak GRPO reward | **0.856** |
| Format reward (peak) | **+0.660** |
| Episode return (avg) | **+0.140** |
| Natural EOS terminations | **First appeared** (model self-terminating after JSON) |
| Training time | ~2 hrs (3 stages × ~39 min on A100 80GB) |
| Training hardware | A100 80GB |
| Framework | TRL 0.15.1 + Unsloth 2026.4.8 + Transformers 5.5.0 |

---

## The Full Training Story

We did not get here in one clean run. Here is the honest story of every iteration.

| Run | Key Fix | Reward |
|-----|---------|--------|
| Base (no LoRA) | — | −0.07 eval |
| Run 1 | — (broken JSON parser) | −0.47 eval — **571% worse** than base |
| Run 2 | Shorter completions | −0.41 eval |
| **Run 3** | **Greedy regex `re.search(r'\{.*\}', text, re.DOTALL)`** | **−0.010** — first real learning |
| Run 4 (v1) | 5-stage curriculum | −0.100 eval — consistency ↑ |
| v3 | Episodic GRPO horizon=3 | +0.140 ep. return — new multi-step capability |
| **v4** | **Dropped dead-weight reward, doubled episode return weight** | **0.856 peak** |

**The single fix that unlocked learning (Run 3):**

```python
# Before — always failed on trailing text:
data = json.loads(completion)

# After — extract first complete JSON object (greedy = critical for nested objects):
match = re.search(r'\{.*\}', completion, re.DOTALL)
data = json.loads(match.group())
```

Why greedy (`.*`) and not non-greedy (`.*?`)? Non-greedy stops at the *first* `}` — breaking on any nested object like `"resource_cost": {"time": 1}`. Greedy takes the outermost `{...}`.

---

## Model Architecture

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter type | LoRA (r=16, alpha=16) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 / 1,562,179,072 (**1.18%**) |
| Adapter size | ~74 MB |
| Training hardware | A100 80GB |
| Precision | BF16 |
| Framework | TRL 0.15.1 + Unsloth 2026.4.8 + Transformers 5.5.0 |

---

## Training: 3-Stage Episodic GRPO Curriculum

v4 uses **episodic training** — the model sees a 3-step sequence, observes the environment state change after each action, and is scored on **discounted trajectory return**. This is fundamentally harder than single-step scoring.

| Stage | Difficulty | Episodes | Episode Horizon | Avg Reward |
|-------|-----------|---------|----------------|------------|
| 1 | 1 (easy) | 60 | 3 | — (not captured in log) |
| 2 | 2 (medium) | 60 | 3 | **0.687** |
| 3 | 3 (hard) | 60 | 3 | **0.717** (peak: **0.856**) |

### Reward Function Stack (v4)

| Function | Weight | Role |
|---|---|---|
| `reward_episode_format_fn` | 1.0 | JSON validity + required keys + valid action_type |
| `reward_clean_eos_fn` | 0.5 | Penalises trailing text after `}` |
| `reward_episode_plausibility_fn` | 0.5 | Blocks zero-cost massive metric claims |
| `reward_episode_return_fn` | 2.0 | 3-step discounted environment trajectory reward |

**Removed in v4:** `reward_compact_fn` — empirically constant −0.5 across all v3 steps, std=0 → zero gradient contribution. Removing it was as important as any hyperparameter tweak.

---

## The LifeStack Environment

Trained inside **LifeStack**, an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment:

- **8 task domains**: career, finances, relationships, physical_health, mental_wellbeing, time, transport_crisis, code_merge_crisis
- **32-edge directed dependency graph**: stress propagates across connected metrics with 0.6× dampening per hop (Starcke & Brand, 2012)
- **23 sub-metrics** with baselines and cascade dampening
- **5 personality profiles** (Big Five / OCEAN) — same action scores differently for an anxious introvert vs a confident executive
- **Exogenous events**: probabilistic mid-episode disruptions
- **ResourceBudget**: time_hours, money_dollars, energy_units — scarcity degrades action effectiveness (Mullainathan & Shafir, 2013)

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
print(obs.reward)  # float reward from environment simulation
```

---

## How to Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("jdsb06/lifestack-grpo-v4")
model = PeftModel.from_pretrained(base, "jdsb06/lifestack-grpo-v4")
model.eval()
```

Or with Unsloth (2× faster):
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="jdsb06/lifestack-grpo-v4",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

---

## Research Grounding

| Principle | Source | Implementation |
|---|---|---|
| Stress cascade dampening (0.6/hop) | Starcke & Brand (2012) | `DependencyGraph.cascade()` |
| Scarcity bandwidth tax | Mullainathan & Shafir (2013) | Budget depletion blocks actions |
| Multi-objective reward weighting | Roijers et al. (2013) | 4 non-overlapping reward signals |
| Retrieval-augmented moderation | RAM / RAG | ChromaDB `LifeStackMemory` |

---

## Model Versions

| Model | Description | Best for |
|-------|-------------|---------|
| [jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) | v1 — 5-stage single-step curriculum | Consistency (45/50 non-failing) |
| [jdsb06/lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3) | v3 — episodic, EOS-aware | Multi-step baseline |
| **[jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4)** | **v4 — episodic curriculum, peak 0.856** | **Production — active demo** |

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
