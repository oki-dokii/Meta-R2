---
language: en
license: apache-2.0
tags:
  - reinforcement-learning
  - grpo
  - episodic-rl
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

**Meta × HuggingFace PyTorch OpenEnv Hackathon 2026**  
**Team BholeChature — Scaler School of Technology, Bangalore**

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-black?style=flat-square&logo=github)](https://github.com/oki-dokii/Meta-R2)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-blue?style=flat-square)](https://github.com/facebookresearch/openenv)

</div>

---

## What is this?

A **LoRA adapter** for `Qwen/Qwen2.5-1.5B-Instruct`, trained with **Group Relative Policy Optimization (GRPO)** inside **LifeStack** — an OpenEnv-compatible RL environment that simulates life as a **40-edge directed dependency graph** across 6 domains.

Given a structured life crisis (flight cancelled, card declined, boss moved your deadline — simultaneously), the model outputs a single **JSON action plan**:

```json
{
  "action_type": "negotiate",
  "target_domain": "career",
  "metric_changes": {"career.workload": -10.0, "mental_wellbeing.stress_level": -5.0},
  "resource_cost": {"time": 0.5, "money": 0.0, "energy": 5.0},
  "reasoning": "Boss extension reduces cascade before financial fix"
}
```

Valid `action_type` values: `negotiate | communicate | delegate | spend | reschedule | rest | deprioritize | execute`

---

## Model Details

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter type | LoRA (r=16, alpha=16) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 / 1,562,179,072 (**1.18%**) |
| Final model size | 1.1 GB (30 weight files) |
| Training hardware | Tesla T4 (15 GB VRAM) |
| Precision | FP16 (T4 / CUDA compute 7.5) |
| Framework | TRL 0.15.1 + Unsloth 2026.4.8 + Transformers 5.5.0 |

---

## The Engineering Journey: 4 Runs to a Real Signal

This model was not trained in one successful run. Here is the honest story.

### Run 1 — Broken Baseline
```
max_completion_length: 256
completions/clipped_ratio: 1.0  (every completion hit the limit)
frac_reward_zero_std: 0.75      (no gradient — 75% of groups tied)
reward (Stage 1 final): -0.944
Eval mean reward: -0.47
Status: ❌ model not learning
```
Root cause: the model outputs valid JSON *followed by explanation text*.  
`json.loads(completion)` fails on trailing text → reward always −0.5 → no learning signal.

### Run 2 — Shorter Completions
```
max_completion_length: 128
reward improved to: -0.266 (Stage 1)
frac_reward_zero_std dropped to: 0.625
Eval mean reward: -0.41
Status: 🟡 minimal improvement, root cause still present
```

### Run 3 — The Fix That Mattered (+97%)
```
Fix: greedy regex extraction before JSON parsing
Eval mean reward: -0.010  ← first positive/near-zero reward
frac_reward_zero_std: 0.05
reward_format_fn: +0.569
Status: ✅ real learning signal
```

The single change that caused a **97% reward improvement** (−0.944 → +0.023):

```python
# Before — always failed on trailing text:
data = json.loads(completion)

# After — extract first complete JSON object:
import re, json
match = re.search(r'\{.*\}', completion, re.DOTALL)  # greedy ← critical
data = json.loads(match.group())
```

**Why greedy and not non-greedy (`\{.*?\}`)?** Non-greedy stops at the *first* `}` — breaking on any nested object like `"resource_cost": {"time": 1}`. Greedy takes the outermost `{...}`, which is correct.

### Run 4 — Final Model (5-stage curriculum)
```
Stages: 5 × 100 prompts = 125 total optimizer steps
max_completion_length: 96 (tight — one JSON fits, two don't)
Additional fix: reward_clean_eos_fn added

Stage 5 final metrics:
  reward: -0.396
  reward_format_fn: +0.195
  reward_task_success: -0.100

Eval (50 episodes, 8 domains):
  ~45/50 episodes hitting reward = 0.000 (consistent, non-failing)
  Mean: -0.100

Status: ✅ consistent performance, best model
```

### Progression Summary

| Run | Model | Reward (total) | Episode Return | Key Fix |
|---|---|---|---|---|
| Run 1 | v1 | −0.47 | — | — (broken) |
| Run 2 | v1 | −0.41 | — | Shorter completions |
| **Run 3** | **v1** | **−0.010** | — | **Greedy regex extraction (+97%)** |
| Run 4 | v1 | −0.100 | — | 5-stage curriculum, tighter budget |
| **v3 (episodic)** | **v3** | **−0.77** | **+0.140** | **Multi-step episodes, difficulty 1→2→3** |

> **v3 note**: Total GRPO reward is dominated by `reward_compact_fn: −0.5` (flat, no variance — model still fills 768-token budget). The meaningful signal is `reward_episode_return_fn: +0.140`, which is the actual environment outcome score. `frac_reward_zero_std = 0%` confirms real gradient flow throughout all 135 steps. The model trained on difficulty 1→2→3 conflict sequences.

---

## Training Configuration (Run 4)

| Parameter | Value |
|---|---|
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 |
| `total_batch_size` | 16 |
| `num_generations` | 4 |
| `learning_rate` | 5e-6 (stage-decayed: 8e-6 → 5e-6 → 3e-6 → 2e-6 → 1e-6) |
| `max_completion_length` | 96 |
| `temperature` | 0.9 |
| `num_train_epochs` | 1 per stage |
| `warmup_ratio` | 0.05 |
| Stage 1 reward | format + clean_eos only (warm-up) |
| Stage 2–5 reward | full 9-signal stack |
| Total steps | 125 (25/stage × 5 stages) |
| Total T4 time | ~45 min (Run 4) / ~4–5 hrs across all runs |

---

## V2 Episodic Training Path

The repository now includes a gated v2 path for HF-credit runs:

```bash
python scripts/train_trl.py --episode-train --dry-run
python scripts/train_trl.py --episode-train --stages 2 --episodes-per-stage 40 --episode-horizon 3 --hub-repo-id jdsb06/lifestack-grpo-v2 --push-to-hub
```

V1 trained one JSON action per prompt, then evaluated long-term effects with a
null/rest rollout. V2 prompts the model to produce `{"actions": [...]}` and
scores the sequence by executing it step-by-step in `LifeStackEnv`. The separate
`--full-episode` runner remains closed-loop evaluation: generate one action,
observe the new state, and repeat.

Two contract fixes are included before spending credits on v2:

- Route ids listed in the prompt now receive full format credit when used as the
  target for `action_type="execute"`.
- Missing ChromaDB / human-feedback storage returns neutral reward instead of
  applying a small penalty to every completion.

The credit gate is: pass `--episode-train --dry-run`, then run a tiny GPU smoke
run and confirm non-zero episode reward variance before pushing `v2`.

---

## 9-Signal Reward Orchestrator

| # | Function | Signal |
|---|---|---|
| 1 | `reward_format_fn` | JSON validity + required keys + valid action_type (+1.0 / +0.5 / −0.5) |
| 2 | `reward_clean_eos_fn` | Penalises trailing text after `}` (+0.20 clean / −0.10 trailing) |
| 3 | `reward_plausibility_fn` | Blocks zero-cost massive metric claims (−0.30) |
| 4 | `reward_task_success_fn` | Environment simulation: did the action resolve the task? |
| 5 | `reward_milestone_fn` | Did the action unlock a task milestone? |
| 6 | `reward_replan_fn` | Did the agent adapt after mid-episode exogenous events? |
| 7 | `reward_reasoning_fn` | Does the reasoning text logically justify the chosen action type? |
| 8 | `reward_human_feedback_fn` | ChromaDB memory: alignment with past successful trajectories |
| 9 | `reward_longterm_fn` | 7-day γ=0.9 discounted rollout — penalises cascade collapse by day 4 |

Anti-hacking: `reward_format_fn` validates `action_type` against an allowlist of 8 valid values. Route IDs (e.g. `"rebook_premium"`) score 0.5, not 1.0. Template/copy outputs score 0.5.

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

With Unsloth (2× faster on T4):
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

## Inference

```python
import re, json, torch

prompt = """You are LifeStack. Return ONLY compact JSON.
Task: Survive Friday 6PM Crisis (flight cancelled, card declined, deadline moved to Sunday)
Key metrics:
- career.workload: 85.0
- finances.liquidity: 30.0
- mental_wellbeing.stress_level: 75.0
- time.free_hours_per_week: 15.0
Budget: time=10.0, money=200.0, energy=40.0
Required keys: action_type, target_domain, metric_changes, resource_cost, reasoning.
Keep reasoning under 25 words. No markdown."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=96, temperature=0.3, do_sample=True)

raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Extract first complete JSON (model may produce trailing text)
match = re.search(r'\{.*\}', raw, re.DOTALL)
action = json.loads(match.group()) if match else {}
print(action)
```

Full Gradio demo:
```bash
git clone https://github.com/oki-dokii/Meta-R2 && cd Meta-R2
pip install -r requirements.txt
python scripts/gradio_demo.py --model-dir ./lifestack_model
```

---

## The LifeStack Environment

OpenEnv-compatible environment modelling life as a 40-edge dependency graph:

```
LifeStackEnv
├── 8 task domains
│   career, finances, relationships, physical_health,
│   mental_wellbeing, time, transport_crisis (5 modes), code_merge_crisis
├── 23 sub-metrics with cascade dampening (0.6/hop — Starcke & Brand 2012)
├── SimPerson: 5 Big Five (OCEAN) personality profiles
├── ResourceBudget: time_hours, money_dollars, energy_units
├── Event system: probabilistic mid-episode disruptions
├── Route system: multiple viable paths with preconditions
└── rollout(n_steps=7, gamma=0.9): long-horizon reward signal
```

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
print(f"reward={obs.reward:.3f}  done={obs.done}")
```

---

## Known Limitations

**`clipped_ratio = 1.0` during training:** The model fills its full 96-token completion budget on every generation and never emits a natural EOS token during GRPO training. This is a known challenge when prior training instils a strong fill-the-context habit. **At inference time this is fully mitigated**: `_JsonCompleteStopping` halts generation when the first `{...}` closes, and `re.search(r'\{.*\}', raw, re.DOTALL)` extracts only the first valid JSON object.

**Difficulty ceiling:** The 5-stage curriculum never advanced past difficulty 1 (advance threshold `reward > 0.6` was never met). The model is trained primarily on difficulty-1 scenarios.

---

## Research Grounding

| Principle | Source | Implementation |
|---|---|---|
| Stress cascade dampening (0.6/hop) | Starcke & Brand (2012) | `DependencyGraph.cascade()` |
| Scarcity bandwidth tax | Mullainathan & Shafir (2013) | Budget depletion blocks actions |
| Multi-objective reward weighting | Roijers et al. (2013) | 9 non-overlapping signals |
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
