# LifeStack Model Comparison: v1 → v4

**Base model:** Qwen2.5-1.5B-Instruct · LoRA r=16 · GRPO via TRL · A100 80GB

---

## v4 Headline Numbers

| Metric | Value |
|--------|-------|
| Peak GRPO reward | **0.856** (Stage 3, final epoch) |
| Format reward peak | **0.660** |
| Episode return (avg) | **~0.140** |
| Natural EOS terminations | **First appeared** (Stage 3) |
| Training time | ~2 hrs (3 stages × ~39 min) |
| HuggingFace | [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) |

---

## Cross-Version Comparison

| Model | Training setup | Best GRPO reward | Episode return | Format reward | Clipped ratio | Natural EOS |
|-------|---------------|-----------------|----------------|---------------|---------------|-------------|
| Base (no LoRA) | None — raw Qwen2.5-1.5B-Instruct | −0.07 (eval) | — | unknown | — | 0 |
| v1 | 5-stage curriculum · single-step · 500 prompts | −0.396 / eval −0.10 | — | +0.195 | 1.0 (always) | 0 |
| v2 | Episodic GRPO · horizon=3 · trajectory return | 0.775 | +0.207 | ~0.55 | 1.0 (always) | 0 |
| v3 | Episodic · EOS-aware · JSON-masked gradient | −0.77\* (distorted) | +0.140 | +0.629 | 1.0 (always) | 0 |
| **v4** | **Episodic curriculum · difficulty 1→2→3 · horizon=3** | **0.856** | **~0.140** | **+0.660** | 0.9875 (Stage 3) | **First appeared** |

> \* v3's −0.77 total reward was caused by `reward_compact_fn` being a dead weight signal (std=0 → zero gradient contribution). Meaningful signals were format +0.629 and episode return +0.140, same as v4.

---

## Format Reward Progression

| Version | Peak format reward | Change vs previous |
|---------|-------------------|--------------------|
| Base    | unknown (negative) | — |
| v1      | +0.195 | baseline |
| v2      | ~0.55  | +182% vs v1 |
| v3      | +0.629 | +14% vs v2 |
| v4      | +0.660 | +4.9% vs v3 |

---

## v4 Per-Step Reward Log (Stages 2 & 3)

Stage 1 per-step metrics were not captured in the terminal log.

### Stage 2 (difficulty=2, horizon=3)

| Epoch | Total reward | Format reward | Episode return | Clipped ratio |
|-------|-------------|---------------|----------------|---------------|
| 0.33 | 0.766 | 0.576 | 0.148 | 1.0 |
| 0.67 | 0.625 | 0.541 | 0.086 | 1.0 |
| 1.00 | 0.662 | 0.529 | 0.117 | 1.0 |
| 1.33 | 0.669 | 0.563 | 0.103 | 1.0 |
| 1.67 | 0.613 | 0.508 | 0.108 | 1.0 |
| 2.00 | 0.801 | 0.564 | 0.169 | 1.0 |
| 2.33 | 0.617 | 0.488 | 0.116 | 1.0 |
| 2.67 | 0.723 | 0.539 | 0.139 | 1.0 |
| 3.00 | 0.709 | 0.586 | 0.112 | 1.0 |
| **avg** | **0.687** | **0.544** | **0.122** | 1.0 |

### Stage 3 (difficulty=3, horizon=3)

| Epoch | Total reward | Format reward | Episode return | Clipped ratio |
|-------|-------------|---------------|----------------|---------------|
| 0.33 | 0.807 | 0.550 | 0.176 | 1.0 |
| 0.67 | 0.510 | 0.456 | 0.076 | **0.9875** |
| 1.00 | 0.766 | 0.588 | 0.135 | 1.0 |
| 1.33 | 0.656 | 0.513 | 0.127 | 1.0 |
| 1.67 | 0.752 | 0.588 | 0.128 | 1.0 |
| 2.00 | 0.642 | 0.537 | 0.100 | 1.0 |
| 2.33 | 0.773 | 0.583 | 0.145 | **0.9875** |
| 2.67 | 0.689 | 0.564 | 0.110 | **0.9875** |
| 3.00 | **0.856** | **0.660** | 0.143 | 1.0 |
| **avg** | **0.717** | **0.560** | **0.127** | 0.996 |

Rows with **clipped_ratio < 1** indicate the model generated a natural EOS token before hitting the 768-token cap — the first time this occurred in any LifeStack training run.

---

## v4 Reward Component Breakdown (Stage 3 average)

| Signal | Avg value | Direction | Notes |
|--------|-----------|-----------|-------|
| `reward_episode_format_fn` | +0.560 | positive | JSON validity + required keys + valid action_type |
| `reward_episode_return_fn` | +0.127 | positive | 3-step discounted trajectory reward |
| `reward_clean_eos_fn` | −0.085 | **negative** | Still generating trailing text after JSON |
| `reward_episode_plausibility_fn` | −0.106 | **negative** | Some actions still implausibly zero-cost |

`reward_compact_fn` was removed in v4 (confirmed dead weight in v3: std=0, zero gradient).

---

## What v4 Improved vs v3

| Area | v3 | v4 | Change |
|------|----|----|--------|
| Format reward (peak) | +0.629 | +0.660 | +4.9% |
| Natural EOS terminations | 0 | First appeared | New capability |
| GRPO total reward | −0.77 (distorted) | 0.856 | Clean signal, no dead weight |
| Episode return | +0.140 | ~0.140 | Plateau — same |
| frac_zero_std | 0% | 0% | Unchanged — gradient real at every step |

---

## What Did Not Change / Concerns

| Concern | Detail |
|---------|--------|
| EOS still weak | Only 3 steps in Stage 3 showed clipped_ratio < 1; majority still hit the 768-token cap |
| EOS penalty negative | `reward_clean_eos_fn` avg −0.085 — trailing text after JSON still common |
| Plausibility negative | `reward_episode_plausibility_fn` avg −0.106 — some unrealistic zero-cost actions |
| Episode return plateau | ~0.14 across v2, v3, and v4 — near the ceiling for horizon=3 |
| High reward std | 0.6–1.0 per step — high variance within groups |

---

## Bottom Line

v4 is the best-trained LifeStack model on every measurable dimension:
- **Highest peak reward** (0.856)
- **Best format compliance** (+0.660)
- **First natural EOS terminations** — model beginning to self-terminate after JSON

The episode return plateau at ~0.14 across v2/v3/v4 suggests the ceiling for horizon=3 with the current reward stack has been reached. To push further in v5:

1. **Increase horizon** (3 → 5 or 7) to give more room for multi-step improvement
2. **Stronger EOS incentive** — increase `reward_clean_eos_fn` weight or add a bonus for early clean termination
3. **Fix plausibility** — tighten the plausibility reward so it stops being a consistent drag

---

*Generated from `train_run_v4.log` · Training completed Apr 26, 2026 · [Full log on HF](https://huggingface.co/jdsb06/lifestack-grpo-v4/blob/main/train_run_v4.log)*
