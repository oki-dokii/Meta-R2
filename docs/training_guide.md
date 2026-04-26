# LifeStack — training guide (from zero to Hub)

This guide walks from **install** → **clone** → **GRPO train** → **push** for **LifeStack** ([repo](https://github.com/oki-dokii/Meta-R2)).

**Stack reference:** Qwen2.5-1.5B-Instruct (Unsloth 4-bit), **TRL 0.15.1**, **Unsloth**, optional **`LIFESTACK_NO_UNSLOTH=1`** on Torch **2.10**.

---

## 1. Prerequisites

- **Python 3.10+** (3.12 OK if wheels exist for your stack).
- **NVIDIA GPU** with enough VRAM for 1.5B + LoRA (**T4 15GB** works).
- **Hugging Face account** if using `--push-to-hub`.

**Pin for reproducibility (recommended):**

```text
trl==0.15.1
transformers==5.5.0   # or the pair you validated
torch 2.10+cu128      # example
unsloth               # or skip via env var below
```

The repo `requirements.txt` may allow wider ranges; for **GRPO** as developed here, **`trl==0.15.1`** avoids optional **`mergekit`** imports in newer TRL.

---

## 2. Clone and environment

```bash
git clone https://github.com/oki-dokii/Meta-R2.git
cd Meta-R2
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# If training, ensure GPU torch matches your CUDA build.
```

Log in for Hub pushes:

```bash
huggingface-cli login
# or export HF_TOKEN=...
```

---

## 3. Colab / Kaggle setup

1. Upload the repo or `git clone` in a notebook cell.
2. Install PyTorch for the runtime’s CUDA, then `pip install trl==0.15.1 unsloth peft accelerate bitsandbytes transformers`.
3. **If Unsloth import or compiled kernels crash**, export:

```python
import os
os.environ["LIFESTACK_NO_UNSLOTH"] = "1"
```

before importing or running `scripts/train_trl.py`.

4. Prefer **long sessions** (e.g. Kaggle limits) or plan for **`--resume`**.

---

## 4. Server / local GPU setup

Same as above. For **Torch 2.10** + Unsloth cache issues:

```bash
export LIFESTACK_NO_UNSLOTH=1
```

Run training from repo root so `core.*` imports resolve.

---

## 5. Smoke test (no full GPU burn)

```bash
python scripts/train_trl.py --dry-run
python scripts/train_trl.py --episode-train --dry-run
```

---

## 6. Full training commands

### Single-step curriculum (baseline style)

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py \
  --stages 5 \
  --prompts-per-stage 100 \
  --max-prompt-length 2048 \
  --max-completion-length 224 \
  --output-dir ./lifestack_model \
  --push-to-hub \
  --hub-repo-id YOUR_USER/lifestack-grpo
```

### Episodic v4-style (current best path)

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py \
  --episode-train \
  --stages 3 \
  --episodes-per-stage 60 \
  --episode-horizon 3 \
  --episode-warmup-stages 1 \
  --prompts-per-stage 60 \
  --num-train-epochs 3 \
  --max-prompt-length 4096 \
  --output-dir ./lifestack_model_v4 \
  --push-to-hub \
  --hub-repo-id jdsb06/lifestack-grpo-v4
```

See [train_trl.md](train_trl.md) for every flag.

---

## 7. Resume training

If a run stops mid-stage:

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py --episode-train --resume \
  --output-dir ./lifestack_model_v4 \
  --stages 3 \
  --episodes-per-stage 60 \
  --episode-horizon 3 \
  --episode-warmup-stages 1 \
  --prompts-per-stage 60 \
  --num-train-epochs 3 \
  --max-prompt-length 4096
```

**Mechanics:**

- `curriculum_state.json` stores completed stage + next difficulty.
- Latest `checkpoint-*` under `episode_stage_N/` (episodic) or `stage_N/` (single-step) is passed to `trainer.train(resume_from_checkpoint=...)`.

---

## 8. Interpreting logs

| Field | Meaning |
|-------|--------|
| **`reward` / `train/reward`** | Weighted sum of reward heads for GRPO (sign is **not** “human happiness” — it’s training objective). |
| **`reward_std` / group stats** | Spread of total reward within each prompt group — **too small** → weak learning signal. |
| **`frac_reward_zero_std`** | Fraction of reward components with **zero** std across the group — **high** → heads not discriminating samples. |
| **`clipped_ratio`** | How often policy updates hit the clip limit. **Sustained ~1.0** often correlates with **noisy long completions** or **tiny advantages**. |

**Honest calibration:** many eval curves stay **near zero** even when the policy improves structurally — the simulation penalizes several failure modes.

---

## 9. Common errors and fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Unsloth crash / `ref_hidden_states` | Compiled path vs Torch | `export LIFESTACK_NO_UNSLOTH=1` |
| `mergekit` / `weave` ImportError | TRL optional deps | Use **`trl==0.15.1`**; script installs **shims** for many cases |
| JSON reward always **−0.5** | Trailing prose after JSON | Greedy `\{.*\}` or `JSONDecoder.raw_decode` path — see [reward.md](reward.md) |
| Truncated JSON | `max_completion_length` too low/high | Single-step: **128–224**; episodic: use default auto or `--episodic-max-completion` |
| TypeError on `GRPOTrainer` / sampler | TRL / Transformers mismatch | Use pinned TRL; script patches `_get_train_sampler` when needed |
| OOM | Batch × length | Lower `--max-prompt-length` or completion cap; use T4-friendly defaults |

---

## 10. After training

- **Evaluate:** `python scripts/train_trl.py --full-episode --output-dir ./lifestack_model_v4`
- **Plot (single-step path):** training may emit `grpo_reward_curve.png` via `evaluate_and_plot`.
- **Artifacts:** `curriculum_state.json`, per-stage checkpoints, final `adapter_config.json` / weights in `--output-dir`.

---

## 11. Hackathon checklist

- **OpenEnv:** `openenv.yaml` points to `core.lifestack_env:LifeStackEnv`.
- **Minimal training:** `scripts/train_trl.py` + TRL/Unsloth.
- **Story:** see [blog.md](blog.md) and [model_card.md](model_card.md).

---

## See also

- [train_trl.md](train_trl.md) — CLI reference  
- [reward.md](reward.md) — reward heads  
- [configuration.md](configuration.md) — `GRPOConfig`  
- [README.md](README.md) — full documentation index  
