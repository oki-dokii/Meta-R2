# LifeStack GRPO Training Guide

> **Model**: Qwen2.5-1.5B-Instruct → LoRA fine-tuned via GRPO  
> **Algorithm**: Group Relative Policy Optimization (TRL + Unsloth)  
> **Domains**: 8 daily-life domains including `transport_crisis` (5 modes), career, finances, relationships, physical health, mental wellbeing, time, code merge crisis

---

## 1. How GRPO Works in LifeStack

GRPO trains the model by generating **groups of completions** for the same prompt and ranking them by reward. The model learns to prefer higher-reward actions without needing a separate critic network (unlike PPO).

```
Prompt (life scenario)
       │
       ▼
 LLM generates N=4 candidate JSON actions
       │
       ▼
 5 reward functions score each action
   ├── format_compliance    (is it valid JSON?)
   ├── plausibility         (no zero-cost miracle fixes?)
   ├── task_success         (did it actually help the LifeMetrics?)
   ├── milestone            (did it unlock key progress gates?)
   └── reasoning            (is the explanation coherent?)
       │
       ▼
 GRPO updates policy to prefer higher-reward completions
```

The curriculum starts at difficulty 1 (gym skipped, forgotten bill) and advances to difficulty 5 (flight cancelled + card declined + boss moved deadline) only when avg reward > 0.6 on the current level.

---

## 2. Free-Tier GPU Recommendation

### ✅ Use Kaggle — not Colab

| | **Kaggle** ✅ | Colab Free ❌ |
|---|---|---|
| GPU | **T4 × 2** (or P100) | T4 × 1 |
| VRAM | **32 GB** (dual T4) | 16 GB |
| Session limit | **9 hours** | **90 minutes** |
| Weekly GPU quota | **30 hrs / week** | ~12 hrs (varies) |
| Storage between sessions | ✅ Persistent (save as Dataset) | ❌ Wiped on disconnect |
| `bf16` support | ❌ T4 is too old → `fp16` used instead | ❌ same |
| Auto-detects fp16 fallback | ✅ script handles it | ✅ script handles it |

**Bottom line**: Colab free sessions cut off at 90 minutes. Even with checkpoints that means 3–4 restarts for a full 5-stage run. One Kaggle session (9h) completes the entire curriculum in a single stretch — no resume needed.

### Paid cloud (if you need speed)

| Tier | GPU | VRAM | Time / Stage | Cost |
|------|-----|------|-------------|------|
| 🥇 Best | A100 80GB | 80 GB | ~25 min | ~$2.50/hr |
| 🥈 Good | A100 40GB | 40 GB | ~45 min | ~$1.60/hr |
| 🥉 Budget | L4 / RTX 3090 | 24 GB | ~90 min | ~$0.80/hr |

### VRAM math (why any T4 is fine)

| Component | VRAM |
|-----------|------|
| Model (1.5B, 4-bit Unsloth) | ~1.2 GB |
| LoRA adapters (r=16) | ~0.1 GB |
| Optimizer states | ~2.0 GB |
| Activations (batch=2, seq=1024) | ~3.5 GB |
| **Total** | **~7 GB** |

A single T4 (16 GB) has 9 GB headroom. Kaggle's dual T4 = 32 GB total.

---

## 3. Environment Setup

### ✅ Option A — Kaggle (Recommended Free Tier)

Create a new Kaggle Notebook → Settings → Accelerator: **GPU T4 x2**.

```python
# Cell 1 — Install deps
!pip install unsloth trl datasets transformers accelerate matplotlib -q

# Cell 2 — Clone repo
!git clone https://github.com/YOUR_ORG/Meta-R2.git
import os; os.chdir("Meta-R2")

# Cell 3 — Smoke test (makes sure everything imports correctly)
!python scripts/train_trl.py --dry-run

# Cell 4 — Full curriculum (completes in ~5–6 hrs on T4 x2)
OUTPUT = "/kaggle/working/lifestack_model"
!python scripts/train_trl.py --stages 5 --prompts-per-stage 200 --output-dir {OUTPUT}
```

**Saving across sessions** (so you can resume if you hit 9h or re-run next week):

```python
# After training, save the output as a Kaggle Dataset via the notebook sidebar:
# Notebook → Data → + Add Output → name it "lifestack-model"
# Next session: attach that dataset and pass --resume
!python scripts/train_trl.py --resume --output-dir /kaggle/input/lifestack-model/lifestack_model
```

### Option B — Google Colab (Secondary, needs Drive)

Colab sessions cut at 90 min. You **must** mount Drive to survive disconnects.

```python
# Cell 1 — Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
OUTPUT = "/content/drive/MyDrive/lifestack_model"

# Cell 2 — Install & clone
!pip install unsloth trl datasets transformers accelerate matplotlib -q
!git clone https://github.com/YOUR_ORG/Meta-R2.git
import os; os.chdir("Meta-R2")

# Cell 3 — Smoke test
!python scripts/train_trl.py --dry-run

# Cell 4 — First run
!python scripts/train_trl.py --stages 5 --prompts-per-stage 200 --output-dir {OUTPUT}

# Cell 5 — After disconnect, re-run cells 1-2, then:
!python scripts/train_trl.py --resume --output-dir {OUTPUT}
```

> ⚠️ Without mounting Drive, every Colab disconnect loses all progress regardless of checkpoints.

### Option C — Local / Cloud GPU (Linux)

```bash
git clone https://github.com/YOUR_ORG/Meta-R2.git && cd Meta-R2
python3 -m venv .venv && source .venv/bin/activate
pip install unsloth trl datasets transformers accelerate matplotlib

# On A100 (CUDA 12.x), use the fast Unsloth build:
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

python -c "import torch; print(torch.cuda.get_device_name(0))"
python scripts/train_trl.py --dry-run
python scripts/train_trl.py --stages 5 --prompts-per-stage 200
```

---

## 4. Checkpoint & Resume System

Every **25 optimiser steps** the Trainer writes a checkpoint. If the session dies mid-stage, it picks up exactly where it left off.

### What gets saved

```
lifestack_model/
├── curriculum_state.json          ← {"completed_stage": 2, "next_difficulty": 3}
├── stage_1/
│   ├── checkpoint-25/             ← step 25 snapshot (weights + optimizer)
│   ├── checkpoint-50/             ← step 50 snapshot
│   ├── checkpoint-75/             ← step 75 (oldest auto-deleted at 4th save)
│   └── model.safetensors          ← written when stage completes cleanly
├── stage_2/
│   └── checkpoint-25/             ← mid-stage when session was cut
└── stage_3/ ...
```

Only the **3 most recent checkpoints** per stage are kept (`save_total_limit=3`) to save disk.

### Resume commands

```bash
# Kaggle / Colab: auto-resume after any disconnect
python scripts/train_trl.py --resume

# Jump to a specific stage (e.g. re-run stage 3 from scratch)
python scripts/train_trl.py --start-stage 3

# Resume + change number of stages (e.g. add 2 more stages)
python scripts/train_trl.py --resume --stages 7
```

How `--resume` works:
1. Reads `curriculum_state.json` → knows stage 2 completed, next is stage 3
2. Calls `find_latest_checkpoint("stage_3/")` → finds `checkpoint-25`
3. `trainer.train(resume_from_checkpoint="stage_3/checkpoint-25")` → restores weights + optimizer state → continues from step 25

---

## 5. Training Commands

### Dry-Run — No GPU Required
```bash
python scripts/train_trl.py --dry-run
```
- 1 step, 4 prompts, CPU only, ~30 seconds
- Expected output: `✅ DRY-RUN PASSED`

### Full Curriculum (Kaggle / cloud)
```bash
python scripts/train_trl.py --stages 5 --prompts-per-stage 200 --output-dir ./lifestack_model
```

### Fast Dev Run (1 stage, test iterations)
```bash
python scripts/train_trl.py --stages 1 --prompts-per-stage 50
```

### All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | — | 1-step CPU smoke test |
| `--stages` | `5` | Number of curriculum stages |
| `--prompts-per-stage` | `100` | Prompts per stage |
| `--output-dir` | `./lifestack_model` | Model save path |
| `--resume` | `False` | Resume from `curriculum_state.json` + latest checkpoint |
| `--start-stage` | `None` | Force-start from a specific stage number |

---

## 6. What Gets Trained On

The dataset covers **all 8 domains equally** using round-robin sampling:

| # | Domain | Scenario Examples |
|---|--------|-----------------|
| 1 | `career` | Boss drops 10-hr task at 5 PM / performance review rumours |
| 2 | `finances` | Card declined, late fee / tax audit, emergency fund needed |
| 3 | `relationships` | Partner feels like a roommate / sibling needs emergency help |
| 4 | `physical_health` | Fainting spell at office / warning signs ignored too long |
| 5 | `mental_wellbeing` | Burnout, inbox at 500 / panic attack at work |
| 6 | `time` | Double-booked all weekend / drowning in obligations |
| 7 | `transport_crisis` | **5 sub-modes** — see below |
| 8 | `code_merge_crisis` | Botched merge took down staging / CTO asking for ETA |

### `transport_crisis` sub-modes (randomly drawn each time)

| Sub-type | Scenario |
|----------|---------|
| `flight_crisis` | Flight cancelled + card declined + deadline moved to Sunday |
| `train_delay` | Signal failure, 90-min delay, 9 AM client meeting |
| `car_breakdown` | Engine seized on highway, tow + rental = $400, rental shortage exo-event |
| `rideshare_surge` | 9x surge pricing, major presentation in 2 hours |
| `transit_strike` | City-wide indefinite strike, e-bike shortage exo-event |

With 5 personalities × 5 difficulty levels × 8 domains, a 200-prompt stage has strong variation across ~3,000+ unique scenario combinations.

---

## 7. Reward Functions

| Function | What it checks | Range |
|----------|---------------|-------|
| `reward_format_fn` | Valid JSON + all required fields | `[-1, 1]` |
| `reward_plausibility_fn` | No miracle zero-cost fixes | `{-1, 1}` |
| `reward_task_success_fn` | LifeMetrics improved + no cascade spread | `[-1, 1]` |
| `reward_milestone_fn` | Logical progress gates hit | `[0, 1]` |
| `reward_reasoning_fn` | Reasoning coherence + domain keywords | `[-0.1, 0.1]` |

```
+1.0 │ Perfect JSON, all metrics improved, milestone hit
 0.5 │ Reasonable action, some metrics improved
 0.0 │ Neutral / no change
-0.5 │ PLAUSIBILITY_VIOLATION or CASCADE_SPREAD_WIDER
-1.0 │ Refusal / empty / broke multiple metrics
```

---

## 8. Monitoring Training

### TensorBoard (local/cloud only)
```bash
tensorboard --logdir ./lifestack_model   # open http://localhost:6006
```
Watch: `train/reward` rising toward 0.5+, `train/kl_divergence` staying < 0.5.

### Console log (every 5 steps)
```
[step 25] reward=0.312 | outcome=0.124 | containment=0.800 | efficiency=0.710
[ckpt] Curriculum state saved → stage=1, next_diff=2
```

### Live JSONL log
```bash
tail -f training_logs/generations.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d['step']} reward={d['reward']:.3f} action={d['action'].get('action_type')}\")
"
```

---

## 9. Expected Training Results

| Stage | Difficulty | Expected Avg Reward | Progression Rule |
|-------|-----------|---------------------|-----------------|
| 1 | 1 — flat tyre, forgotten bill | 0.55 – 0.70 | advances if > 0.60 |
| 2 | 2 — project surge, train delay | 0.45 – 0.65 | advances if > 0.60 |
| 3 | 3 — health scare, car breakdown | 0.35 – 0.55 | advances if > 0.60 |
| 4 | 4 — performance review, surge pricing | 0.25 – 0.50 | advances if > 0.60 |
| 5 | 5 — transit strike, total collapse | 0.20 – 0.45 | — |

---

## 10. Post-Training Artifacts

```
lifestack_model/
├── curriculum_state.json          ← curriculum progress tracker
├── model.safetensors              ← final LoRA adapter weights
├── adapter_config.json
├── tokenizer.json / tokenizer_config.json
└── stage_1/ ... stage_5/
    ├── checkpoint-25/ ... checkpoint-75/   ← step snapshots
    └── model.safetensors                   ← completed stage weights

training_logs/
└── generations.jsonl              ← per-step reward breakdown
```

Validate the final save:
```bash
python -c "from scripts.train_trl import validate_saved_model; validate_saved_model('./lifestack_model')"
```

---

## 11. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ImportError: unsloth` | Not installed | `pip install unsloth` |
| `CUDA out of memory` | Batch too large | `per_device_train_batch_size=1` |
| All rewards = -0.5 | Env reset failing | Run `--dry-run` to surface the error |
| KL divergence > 1.0 | LR too high | Lower `learning_rate` to `1e-6` |
| `Task missing required fields` | Domain generator bug | Check `TaskGenerator.generate()` |
| reward stuck at 0.0 | Model refuses JSON | Check `reward_format_fn` — should be -1.0 not 0.0 |
| Colab disconnect lost progress | Drive not mounted | Mount Drive before running; use `--resume` |
| `checkpoint-*` dirs missing | `save_steps` too high | Already set to 25 in this script |

---

## 12. Quick Reference

```bash
# Smoke test (CPU, ~30s)
python scripts/train_trl.py --dry-run

# Kaggle full run (~5-6 hr, T4 x2)
python scripts/train_trl.py --stages 5 --prompts-per-stage 200

# Resume after any disconnect
python scripts/train_trl.py --resume

# Jump to stage 3 (e.g. stages 1-2 already done)
python scripts/train_trl.py --start-stage 3

# Validate model saved correctly
python -c "from scripts.train_trl import validate_saved_model; validate_saved_model('./lifestack_model')"

# Plot evaluation reward curve
python -c "from scripts.train_trl import evaluate_and_plot; evaluate_and_plot('./lifestack_model')"
```
