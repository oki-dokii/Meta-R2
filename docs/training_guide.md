# LifeStack GRPO Training Guide

> **Model**: Qwen2.5-1.5B-Instruct → LoRA fine-tuned via GRPO  
> **Algorithm**: Group Relative Policy Optimization (TRL + Unsloth)  
> **Domains**: 8 daily-life scenarios (career, finances, relationships, physical health, mental wellbeing, time, flight crisis, code merge crisis)

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

## 2. GPU Recommendation

### ✅ Recommended: NVIDIA A100 40GB (Best)

| Tier | GPU | VRAM | Batch Size | Time / Stage | Cost (Lambda/RunPod) |
|------|-----|------|-----------|-------------|----------------------|
| 🥇 Best | A100 80GB | 80 GB | 8 | ~25 min | ~$2.50/hr |
| 🥈 Good | A100 40GB | 40 GB | 4 | ~45 min | ~$1.60/hr |
| 🥉 Budget | L4 / 3090 | 24 GB | 2 | ~90 min | ~$0.80/hr |
| 🆓 Free | Colab T4 | 16 GB | 1 | ~3.5 hr | free (slow) |

**Recommended choice: A100 40GB on Lambda Labs or RunPod.**  
Reasons:
- 4-bit quantization (Unsloth) keeps the 1.5B model at ~1.2 GB VRAM — 40 GB is massively headroom
- `bf16` precision is natively supported (unlike T4 which needs fp16)
- 5-stage curriculum completes in **~4 hours** end-to-end

> **Minimum viable**: Google Colab free tier (T4, 16 GB). The dry-run works on CPU. Full curriculum on T4 takes ~18 hours.

### VRAM math

| Component | VRAM |
|-----------|------|
| Model (1.5B, 4-bit) | ~1.2 GB |
| LoRA adapters (r=16) | ~0.1 GB |
| Optimizer states | ~2.0 GB |
| Activations (batch=2, seq=1024) | ~3.5 GB |
| **Total** | **~7 GB** |

Even a 16 GB T4 has 9 GB headroom. 40 GB A100 lets you bump `per_device_train_batch_size` to 8 for faster convergence.

---

## 3. Environment Setup

### Option A — Google Colab (Quick Start)

```python
# Cell 1: Install deps
!pip install unsloth trl datasets transformers accelerate matplotlib -q

# Cell 2: Clone repo
!git clone https://github.com/YOUR_ORG/Meta-R2.git
%cd Meta-R2

# Cell 3: Copy env
import os
os.environ["OPENAI_API_KEY"] = "sk-..."   # optional, only needed for agent.py

# Cell 4: Run
!python scripts/train_trl.py --dry-run        # smoke test first
!python scripts/train_trl.py --stages 5       # full run
```

### Option B — Local / Cloud GPU (Linux)

```bash
# 1. Clone & enter
git clone https://github.com/YOUR_ORG/Meta-R2.git && cd Meta-R2

# 2. Create venv
python3 -m venv .venv && source .venv/bin/activate

# 3. Install
pip install unsloth trl datasets transformers accelerate matplotlib

# (On A100 with CUDA 12.x, use the fast unsloth build)
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

# 4. Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 5. Smoke test (CPU, ~30 sec)
python scripts/train_trl.py --dry-run

# 6. Full curriculum
python scripts/train_trl.py --stages 5 --prompts-per-stage 200 --output-dir ./lifestack_model
```

---

## 4. Training Commands

### Dry-Run (Smoke Test) — No GPU Required
```bash
python scripts/train_trl.py --dry-run
```
- Runs **1 training step** on 4 prompts
- Verifies all imports, reward functions, and model.save_pretrained()
- Completes in **< 2 minutes on CPU**
- Expected output: `✅ DRY-RUN PASSED`

### Standard Curriculum Training
```bash
python scripts/train_trl.py \
    --stages 5 \
    --prompts-per-stage 200 \
    --output-dir ./lifestack_model
```

### Fast Dev Run (Single Stage)
```bash
python scripts/train_trl.py \
    --stages 1 \
    --prompts-per-stage 50 \
    --output-dir ./lifestack_model_dev
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | — | 1-step smoke test, CPU-safe |
| `--stages` | `5` | Number of curriculum stages |
| `--prompts-per-stage` | `100` | Prompts generated per stage |
| `--output-dir` | `./lifestack_model` | Where to save the final model |

---

## 5. What Gets Trained On

The dataset covers **all 8 daily-life domains equally** (round-robin):

| # | Domain | Example Scenario |
|---|--------|-----------------|
| 1 | `career` | Boss drops a 10-hour task at 5 PM Friday |
| 2 | `finances` | Credit card declined, late fee applied |
| 3 | `relationships` | Partner feels ignored; needs reconnection |
| 4 | `physical_health` | Fainting spell at office, tests needed |
| 5 | `mental_wellbeing` | Burnout; motivation gone, inbox at 500+ |
| 6 | `time` | Double-booked all weekend, can't say no |
| 7 | `flight_crisis` | Flight cancelled + deadline moved to Sunday |
| 8 | `code_merge_crisis` | Botched merge took down staging environment |

With 5 personalities (Alex, Chloe, Sam, Jordan, Maya) × 5 difficulty levels × 8 domains, a 200-prompt stage has strong variation.

---

## 6. Reward Functions

Each GRPO rollout is scored by 5 functions (all return `float`):

| Function | What it checks | Range |
|----------|---------------|-------|
| `reward_format_fn` | Valid JSON + all required fields | `[-1, 1]` |
| `reward_plausibility_fn` | No miracle zero-cost fixes | `{-1, 1}` |
| `reward_task_success_fn` | LifeMetrics improved + no cascade | `[-1, 1]` |
| `reward_milestone_fn` | Logical progress gates hit | `[0, 1]` |
| `reward_reasoning_fn` | Reasoning coherence + domain keywords | `[-0.1, 0.1]` |

The overall reward landscape:

```
+1.0 │ Perfect JSON, all metrics improved, milestone hit
 0.5 │ Reasonable action, some metrics improved
 0.0 │ Neutral / no change
-0.5 │ PLAUSIBILITY_VIOLATION or CASCADE_SPREAD_WIDER
-1.0 │ Refusal / empty / broke multiple metrics
```

---

## 7. Monitoring Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir ./lifestack_model
# Open http://localhost:6006
```
Watch for:
- `train/reward` increasing from ~0 toward 0.5+
- `train/kl_divergence` staying below 0.5 (above that = unstable policy)

### Console Logs
Every 20 reward calls, the script prints:
```
[step 20] reward=0.312 | outcome=0.124 | containment=0.800 | efficiency=0.710
```

### JSONL Generation Log
```bash
tail -f training_logs/generations.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"step={d['step']} reward={d['reward']:.3f} action={d['action'].get('action_type')}\")
"
```

---

## 8. Expected Training Results

| Curriculum Stage | Difficulty | Expected Avg Reward | Progression Rule |
|-----------------|-----------|---------------------|-----------------|
| Stage 1 | 1 (gym, bill) | 0.55 – 0.70 | advances if > 0.60 |
| Stage 2 | 2 (project surge, car) | 0.45 – 0.65 | advances if > 0.60 |
| Stage 3 | 3 (interview, health scare) | 0.35 – 0.55 | advances if > 0.60 |
| Stage 4 | 4 (performance review, audit) | 0.25 – 0.50 | advances if > 0.60 |
| Stage 5 | 5 (Friday 6PM, perfect storm) | 0.20 – 0.45 | — |

A healthy training run shows **monotonically increasing average reward per stage** and very few `PLAUSIBILITY_VIOLATION` or `INACTION_PENALTY` events by stage 3+.

---

## 9. Post-Training Artifacts

After `train_curriculum()` completes:

```
lifestack_model/
├── model.safetensors          # LoRA adapter weights
├── adapter_config.json        # LoRA config
├── tokenizer.json
├── tokenizer_config.json
└── stage_1/ ... stage_5/      # Intermediate checkpoints

training_logs/
└── generations.jsonl          # Per-step reward breakdown log
```

To verify the save was real (not a placeholder):
```bash
python -c "from scripts.train_trl import validate_saved_model; validate_saved_model('./lifestack_model')"
```

---

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ImportError: unsloth` | Not installed | `pip install unsloth` |
| `CUDA out of memory` | Batch too large | Set `per_device_train_batch_size=1` |
| All rewards = -0.5 | Env reset failing | Run `--dry-run` to see the actual error |
| KL divergence > 1.0 | LR too high | Lower `learning_rate` to `1e-6` |
| `Task missing required fields` | Domain generator bug | Check `TaskGenerator.generate()` |
| reward stuck at 0.0 | Model refuses JSON | Add `reward_format_fn` penalty signal first |

---

## 11. Quick Reference Cheat Sheet

```bash
# Smoke test (CPU, ~30s)
python scripts/train_trl.py --dry-run

# Fast dev run (GPU, ~15 min)
python scripts/train_trl.py --stages 1 --prompts-per-stage 50

# Full production run (A100 40GB, ~4 hr)
python scripts/train_trl.py --stages 5 --prompts-per-stage 200

# Watch TensorBoard
tensorboard --logdir ./lifestack_model

# Validate saved model
python -c "from scripts.train_trl import validate_saved_model; validate_saved_model('./lifestack_model')"

# Run evaluation + plot
python -c "from scripts.train_trl import evaluate_and_plot; evaluate_and_plot('./lifestack_model')"
```
