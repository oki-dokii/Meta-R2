# Configuration — LifeStack training & runtime

This document covers **`GRPOConfig`** usage in **`scripts/train_trl.py`**, differences between **single-step** and **episodic** training, and **reward weight** layout for **v4**.

---

## `GRPOConfig` (TRL)

The script builds configs via **`_make_grpo_config(**kwargs)`**, which **filters** keyword arguments to fields that exist on the installed **`GRPOConfig`** (TRL minor versions vary). Unsupported keys are printed as skipped.

### Fields commonly set (single-step & episodic)

| Parameter | Typical value | Notes |
|-----------|---------------|--------|
| `output_dir` | `stage_N` or `episode_stage_N` | Per-stage checkpoint root |
| `num_train_epochs` | `1` (single-step default) / `1–5` (episodic) | Episodic uses CLI `--num-train-epochs` |
| `per_device_train_batch_size` | `4` | Must work with `num_generations` |
| `gradient_accumulation_steps` | `4` | Effective batch scaling |
| `learning_rate` | Stage schedule, e.g. **8e-6 → 1e-6** (single-step) or **3e-6 → 1e-6** (episodic) | See `train_curriculum` / `train_episodic_curriculum` |
| `warmup_ratio` | `0.05` | |
| `max_prompt_length` | `2048`–`4096` | CLI `--max-prompt-length` |
| `max_completion_length` | **224** (single-step default) | Episodic auto: `min(1024, max(512, 256 * horizon))` unless overridden |
| `temperature` | **0.7** (single-step), **0.9** (episodic) | Higher episodic temp restores exploration / EOS diversity |
| `num_generations` | `4` | GRPO group size; must divide batch per TRL rules |
| `bf16` / `fp16` | From **`_dtype_flags(model)`** | Avoids GradScaler + float16 clash on Unsloth |
| `save_strategy` | `"steps"` | |
| `save_steps` | `5` | |
| `save_total_limit` | `3` | |
| `logging_steps` | `5` | |
| `report_to` | `"tensorboard"` or `"none"` | Auto if tensorboard missing |
| `reward_weights` | list of floats | **Must align** with `reward_funcs` length |

**Unsloth-specific:**

- `config.unsloth_num_chunks = -1` is set after construction when the field exists.

---

## Single-step vs episodic

| Aspect | Single-step (`train_curriculum`) | Episodic (`train_episodic_curriculum`) |
|--------|----------------------------------|----------------------------------------|
| **Dataset** | `generate_dataset` — one action JSON per prompt | `generate_episodic_dataset` — sequence JSON per prompt |
| **Trainer** | `GRPOTrainer` | `LifeStackGRPOTrainer` |
| **Completion budget** | `--max-completion-length` | Auto or `--episodic-max-completion` |
| **Reward heads** | Stage 1: format + EOS + route; later: + plausibility, task, milestones, … | v4: episode format, EOS, episode plausibility, episode return |
| **Warm-up** | N/A | `--episode-warmup-stages` runs single-step first |

---

## Reward weight configuration

### Single-step stage 1 (warm-up)

```text
reward_funcs = [reward_format_fn, reward_clean_eos_fn, reward_route_target_fn]
reward_weights = [1.0, 1.5, 1.0]
```

### Single-step stages ≥ 2

Full stack including simulation and long-horizon reward; weights are listed in `train_curriculum` (ten heads).

### Episodic v4

```text
reward_funcs = [
  reward_episode_format_fn,
  reward_clean_eos_fn,
  reward_episode_plausibility_fn,
  reward_episode_return_fn,
]
reward_weights = [1.0, 0.5, 0.5, 2.0]
```

**v3 note:** included `reward_compact_fn`, which empirically added **constant** negative bias with **no variance** — removed in v4.

---

## Curriculum / difficulty

- **`curriculum_state.json`**: `{ "completed_stage": int, "next_difficulty": int }`.
- **Advance rule (code):** if mean logged reward **≥ 0** at end of stage and difficulty **< 5**, increment difficulty.

---

## Runtime environment variables

| Variable | Effect |
|----------|--------|
| `LIFESTACK_NO_UNSLOTH=1` | Skip Unsloth import; use stock **`GRPOTrainer`** path |
| `HF_TOKEN` | Hugging Face Hub auth for `--push-to-hub` |

---

## `openenv.yaml`

Declares **`lifestack-v1`**, entry **`core.lifestack_env:LifeStackEnv`**, action/observation classes, and metadata (domains, difficulty range). Used for OpenEnv CLI compatibility and hackathon requirements.

---

## See also

- [train_trl.md](train_trl.md) — flags  
- [reward.md](reward.md) — head definitions  
- [training_guide.md](training_guide.md) — operations  
