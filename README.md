---
title: LifeStack
emoji: 🪐
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
---

# LifeStack — Training AI to Handle Life's Cascading Crises

**Meta × HuggingFace PyTorch OpenEnv Hackathon 2026**
**Team BholeChature — Scaler School of Technology, Bangalore**

[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-black?style=flat-square&logo=github)](https://github.com/oki-dokii/Meta-R2)
[![HF Space](https://img.shields.io/badge/HF_Space-Live_Demo-yellow?style=flat-square)](https://huggingface.co/spaces/jdsb06/meta-r2)
[![v1 Model](https://img.shields.io/badge/Model-lifestack--grpo-blue?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo)
[![v3 Model](https://img.shields.io/badge/Model-lifestack--grpo--v3-violet?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo-v3)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-orange?style=flat-square)](https://github.com/facebookresearch/openenv)

---

## What is LifeStack?

LifeStack is an **OpenEnv-compatible reinforcement learning environment** that models human life as a **40-edge directed dependency graph** across 8 domains. A fine-tuned language model acts as an agent inside this environment, proposing structured JSON action plans for multi-domain crises.

**The core idea:** real life crises don't happen in isolation. A cancelled flight cascades into a missed deadline, which cascades into financial stress, which cascades into relationship strain. LifeStack simulates these cascades and trains a model to navigate them.

Given a crisis like *"flight cancelled, card declined, boss moved your deadline to Sunday"*, the model outputs:

```json
{
  "action_type": "negotiate",
  "target_domain": "career",
  "metric_changes": {"career.workload": -10.0, "mental_wellbeing.stress_level": -5.0},
  "resource_cost": {"time": 0.5, "money": 0.0, "energy": 5.0},
  "reasoning": "Request a deadline extension before the financial situation worsens"
}
```

Valid `action_type` values: `negotiate | communicate | delegate | spend | reschedule | rest | deprioritize | execute`

---

## The Environment

LifeStack wraps a simulation of life as a graph:

```
LifeStackEnv (OpenEnv-compatible)
├── 8 task domains
│   career, finances, relationships, physical_health,
│   mental_wellbeing, time, transport_crisis, code_merge_crisis
├── 23 sub-metrics with cascade dampening (0.6/hop — Starcke & Brand 2012)
├── SimPerson: 5 personality profiles (Big Five / OCEAN)
├── ResourceBudget: time_hours, money_dollars, energy_units
├── Event system: probabilistic mid-episode disruptions
├── Route system: multiple viable action paths with preconditions
└── rollout(n_steps=7, gamma=0.9): long-horizon reward signal
```

Every action the model takes is simulated through `DependencyGraph.cascade()`, which propagates changes through connected metrics with 0.6× dampening per hop. A model that spends money to solve a financial crisis correctly sees `finances.liquidity` drop, which then propagates through `mental_wellbeing.stress_level`, which propagates into `physical_health.energy`.

---

## The Model

**Base:** `Qwen/Qwen2.5-1.5B-Instruct`
**Method:** GRPO (Group Relative Policy Optimization) via TRL 0.15.1 + Unsloth
**Adapter:** LoRA (r=16, α=16) on all projection layers
**Trainable params:** 18.5M / 1.56B (1.18%)
**Hardware:** Tesla T4 (15 GB VRAM)

---

## The Full Training Story

This model was not trained in one clean run. Here is the honest story of every run, what broke, what we fixed, and what we learned.

### Establishing a Baseline First

Before any fine-tuning, we evaluated the raw `Qwen/Qwen2.5-1.5B-Instruct` base model (no LoRA) on the same 50-episode deterministic eval loop across all 8 domains:

| Metric | Base Model (no LoRA) |
|---|---|
| Mean reward | **−0.07** |
| Episodes | 50 (deterministic seeds) |

**Per-domain breakdown:**

| Domain | n | Mean Reward |
|---|---|---|
| career | 7 | −0.143 |
| physical_health | 6 | −0.167 |
| mental_wellbeing | 6 | **−0.250** |
| finances | 7 | 0.000 |
| relationships | 6 | 0.000 |
| time | 6 | 0.000 |
| transport_crisis | 6 | 0.000 |
| code_merge_crisis | 6 | 0.000 |

The 0.000 domains are not "good" — they mean the base model produced low-impact or neutral actions that changed nothing. The three negative domains (mental_wellbeing, physical_health, career) are where the base model actively recommended wrong actions that made metrics worse.

---

### Run 1 — Broken Baseline

```
Setup:        Single-step GRPO, max_completion_length=256
clipped_ratio: 1.0   (every completion hit the token limit)
frac_zero_std: 0.75  (no gradient — 75% of groups tied on reward)
Reward Stage 1 final: −0.944
Eval mean reward:     −0.47
Status: FAILED — model not learning
```

**Root cause:** Qwen2.5-1.5B outputs valid JSON *followed by explanation text*. `json.loads(full_completion)` fails on trailing text, so reward was always −0.5. Every completion in a group got the same score → zero gradient → no learning.

At −0.47, the trained model was **571% worse** than the untrained baseline (−0.07).

---

### Run 2 — Shorter Completions

```
Setup:        max_completion_length=128
Reward Stage 1: −0.266
Eval mean reward: −0.41
Status: MINIMAL IMPROVEMENT — root cause still present
```

Reducing completion length helped marginally but the fundamental parsing problem remained. Still **486% worse** than baseline.

---

### Run 3 — The Fix That Mattered (+85.7% over baseline)

```
Fix:          Greedy regex extraction before JSON parsing
Eval mean reward: −0.010
frac_zero_std: 0.05
reward_format_fn: +0.569
Status: FIRST REAL LEARNING SIGNAL
```

The single change that caused a **97% improvement within training runs** (−0.944 → +0.023 GRPO reward):

```python
# Before — always failed on trailing text:
data = json.loads(completion)

# After — extract first complete JSON object:
import re, json
match = re.search(r'\{.*\}', completion, re.DOTALL)  # greedy ← critical
data = json.loads(match.group())
```

**Why greedy and not non-greedy (`\{.*?\}`)?** Non-greedy stops at the *first* `}` — breaking on any nested object like `"resource_cost": {"time": 1}`. Greedy takes the outermost `{...}`.

**Comparison to baseline:**
- Baseline: −0.07
- Run 3: −0.010
- **Improvement: +85.7% over the untrained base model**

---

### Run 4 — Curriculum Training (v1 final checkpoint)

```
Setup:        5-stage curriculum, 100 prompts/stage, max_completion_length=96
              Stage 1: format + clean_eos only (warm-up)
              Stages 2–5: full 9-signal reward stack
              Learning rate: 8e-6 → 5e-6 → 3e-6 → 2e-6 → 1e-6
Stage 5 metrics:
  reward: −0.396
  reward_format_fn: +0.195
  reward_task_success: −0.100
Eval (50 episodes, 8 domains):
  ~45/50 episodes at reward = 0.000 (consistent, non-failing)
  Mean: −0.100
Status: CONSISTENT but PLATEAU
```

**Comparison to baseline:**
- Baseline: −0.07
- Run 4: −0.100
- **Result: −42.9% regression** vs the untrained base model

The 5-stage curriculum plateaued. The difficulty advance threshold (`reward > 0.6`) was never met, so training stayed at difficulty 1 for all stages. The resulting model was *more consistent* (45/50 non-failing vs ~25/50 for Run 3) but scored slightly lower on average due to curriculum overfitting to format signals.

**Run 4 is saved as `jdsb06/lifestack-grpo` (v1).** It is the best single-step checkpoint for consistency, not peak reward.

---

### v3 — Episodic Multi-Step Training (New Capability)

```
Setup:        Episodic GRPO, horizon=3, 3-stage curriculum difficulty 1→2→3
              135 total optimizer steps
              Reward weights: format 1.0 | eos 1.5 | plausibility 0.75 | return 1.0 | compact −0.5

GRPO training metrics:
  Total reward (logged): −0.77
  reward_compact_fn:     −0.50  (std=0 — zero gradient, dead weight)
  reward_episode_return: +0.140 (the real signal)
  reward_format_fn:      +0.629
  frac_zero_std:         0.00   (100% of groups have reward variance — real gradient)
  Reward std:            0.511
Status: CURRICULUM WORKS, reward_compact_fn is dead weight
```

**Why the total reward of −0.77 is misleading:**

`reward_compact_fn` was −0.50 with standard deviation = 0 across all 135 steps. When every completion in a GRPO group gets the same score, that signal contributes **zero gradient**. It is pure dead weight that drags the logged mean without affecting the model at all. The meaningful signal is `reward_episode_return_fn: +0.140`.

`frac_zero_std = 0%` confirms that across all training steps, every group had real reward variance → real gradient flow. The model was learning throughout.

**What v3 learned that the base model cannot do:**

v3 was trained on 3-action episode sequences. The model observes the result of action 1 before choosing action 2, and the result of action 2 before choosing action 3. This is sequential decision-making under cascading consequences — a fundamentally harder task than single-action scoring.

| | Base Model | v1 Run 4 | v3 |
|---|---|---|---|
| Single-step reward | −0.07 | −0.10 | not eval'd single-step |
| Episode return (3-step) | no capability | no capability | **+0.140** |
| Format score | unknown | +0.195 | **+0.629** |
| Zero-grad groups | — | — | **0%** |

**The multi-step comparison is not apples-to-apples with the baseline.** The base model was evaluated on single-step actions. v3's episode return measures a 3-step discounted trajectory (γ=0.9). These are different tasks. The correct framing is:

> The base model has zero multi-step planning capability. v3 demonstrates +0.140 discounted episode return across 3-step sequences — a new capability that did not exist before GRPO training.

**v4 is saved as `jdsb06/lifestack-grpo-v4`.** It is the active production model in the HF Space.

---

## Metric Progression Summary

| Run | vs Baseline | Eval Reward | Key Fix |
|---|---|---|---|
| Base model | — | −0.07 | Untrained Qwen2.5-1.5B-Instruct |
| Run 1 (broken) | −571% | −0.47 | — (JSON parsing broken) |
| Run 2 | −486% | −0.41 | Shorter completions |
| **Run 3** | **+85.7%** | **−0.010** | **Greedy regex extraction** |
| Run 4 (v1) | −42.9% | −0.100 | 5-stage curriculum (consistency ↑, plateau) |
| **v3 episodic** | **new capability** | **+0.140 ep. return** | **Multi-step episodes, 0% zero-grad groups** |

**The clean claims for this project:**
1. Greedy regex extraction unlocked learning: **+85.7% improvement** over the untrained base model (Run 3)
2. Episodic GRPO gave the model multi-step planning capability: **+0.140 discounted episode return** across 3-step sequences — a capability the base model does not have
3. `frac_zero_std = 0%` throughout v3 training confirms the gradient was real at every step

---

## The 9-Signal Reward Stack

| # | Signal | What it measures |
|---|---|---|
| 1 | `reward_format_fn` | JSON validity + required keys + valid action_type (+1.0 / +0.5 / −0.5) |
| 2 | `reward_clean_eos_fn` | Penalises trailing text after `}` (+0.20 clean / −0.10 trailing) |
| 3 | `reward_plausibility_fn` | Blocks zero-cost massive metric claims (−0.30) |
| 4 | `reward_task_success_fn` | Environment simulation: did the action resolve the conflict? |
| 5 | `reward_milestone_fn` | Did the action unlock a task milestone? |
| 6 | `reward_replan_fn` | Did the agent adapt after a mid-episode exogenous event? |
| 7 | `reward_reasoning_fn` | Does the reasoning logically justify the action type? |
| 8 | `reward_human_feedback_fn` | ChromaDB memory: alignment with past successful trajectories |
| 9 | `reward_longterm_fn` | 7-day γ=0.9 discounted rollout — penalises cascade collapse by day 4 |

**Removed in v3/v4:** `reward_compact_fn` — empirically constant −0.5, std=0 across all steps, zero gradient contribution.

---

## Engineering Bugs Fixed

1. **JSON + trailing prose**: model output valid JSON then English explanation; `json.loads(full_text)` always failed → **greedy brace extraction** fixes it; non-greedy `\{.*?\}` breaks on nested `metric_changes`
2. **max_completion_length too large**: 256 tokens → model always hit limit, `clipped_ratio=1.0` → reduced to 96 (single-step) and 128 (episodic inference)
3. **reward_compact_fn dead weight**: std=0 across 135 steps → removed, episode_return reweighted 1.0→2.0
4. **HF Serverless doesn't support LoRA adapters**: `InferenceClient.text_generation()` with a LoRA repo ID always fails silently → removed that path entirely; local GPU weights are priority
5. **Parallel GPU inference timeout**: 3 ThreadPoolExecutor threads calling `model.generate()` simultaneously → serialized by PyTorch anyway, hit 65s timeout → switched to sequential execution for local model
6. **on_hf_spaces OOM guard**: forced `api_only=True` on Spaces, preventing local model load → removed; Qwen2.5-1.5B (~3GB) fits on T4 alongside Flask+ChromaDB
7. **Unsloth + Torch 2.10**: compiled cache crashes with `ref_hidden_states=None` → `LIFESTACK_NO_UNSLOTH=1` to disable
8. **TRL 0.15.1 + Transformers 5.5**: `_get_train_sampler` signature mismatch → monkey-patch with `inspect.signature` guard

---

## Research Grounding

| Principle | Source | Implementation |
|---|---|---|
| Stress cascade dampening (0.6/hop) | Starcke & Brand (2012) | `DependencyGraph.cascade()` |
| Scarcity bandwidth tax | Mullainathan & Shafir (2013) | Budget depletion blocks actions |
| Multi-objective reward weighting | Roijers et al. (2013) | 9 non-overlapping signals |
| Retrieval-augmented moderation | RAM / RAG | ChromaDB `LifeStackMemory` |

---

## Models on Hugging Face

| Model | Checkpoint | Best for |
|---|---|---|
| [jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) | v1 Run 4 | Single-step consistency (45/50 non-failing) |
| [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) | v4 episodic | Multi-step planning, active production model |

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
tokenizer = AutoTokenizer.from_pretrained("jdsb06/lifestack-grpo-v4")
model = PeftModel.from_pretrained(base, "jdsb06/lifestack-grpo-v4")
model.eval()
```

```python
import re, json, torch

prompt = """You are LifeStack. Return ONLY compact JSON.
Task: Survive Friday 6PM Crisis
Metrics: career.workload=85, finances.liquidity=30, mental_wellbeing.stress_level=75
Budget: time=10h, money=$200, energy=40
Required keys: action_type, target_domain, metric_changes, resource_cost, reasoning"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, temperature=0.3, do_sample=True)

raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
match = re.search(r'\{.*\}', raw, re.DOTALL)
action = json.loads(match.group()) if match else {}
print(action)
```

---

## Training (T4 / Colab)

Pinned stack: **Unsloth 2026.4.8**, **TRL 0.15.1**, **Transformers 5.5.0**, **Torch 2.10+cu128**

```bash
# Episodic v3-style run
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py \
  --episode-train \
  --stages 3 \
  --episodes-per-stage 60 \
  --episode-horizon 3 \
  --output-dir ./lifestack_model_v3 \
  --push-to-hub \
  --hub-repo-id jdsb06/lifestack-grpo-v4
```

```bash
# Evaluate baseline (no LoRA)
python scripts/eval_baseline.py --output baseline_results.json
```

---

## Repository Layout

```
Meta-R2/
├── core/              # LifeStackEnv, DependencyGraph, reward, life_state
├── agent/             # LifeStackAgent, counterfactuals, conflict_generator
├── intake/            # SimPerson (personality profiles)
├── scripts/           # train_trl.py, eval_baseline.py, gradio_demo.py
├── templates/         # Flask UI templates
├── docs/              # Full documentation (start at docs/README.md)
├── app_flask.py       # Production Flask app (port 7860)
├── server.py          # OpenEnv HTTP server (port 8000)
├── start.sh           # Entrypoint: starts both services
├── Dockerfile
└── openenv.yaml       # OpenEnv manifest
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
