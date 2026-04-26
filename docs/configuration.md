# Configuration

**Source file:** `scripts/train_trl.py`

---

## `GRPOConfig` via `_make_grpo_config()`

`train_trl.py` builds all `GRPOConfig` instances through `_make_grpo_config(**kwargs)`, which silently drops kwargs not present in the installed TRL version's `GRPOConfig` dataclass. This makes the script compatible across TRL 1.x minor versions where fields differ.

### Common parameters

| Parameter | Single-step default | Episodic default | Notes |
|-----------|--------------------|-----------------|----|
| `per_device_train_batch_size` | 4 | 4 | |
| `gradient_accumulation_steps` | 4 | 4 | Effective batch = 16 |
| `num_generations` | 4 | 4 | GRPO group size; must divide batch size |
| `max_prompt_length` | 2048 | 2048 | CLI `--max-prompt-length` |
| `max_completion_length` | 224 | auto (min 512, max 1024) | Episodic: `min(1024, max(512, 256 × horizon))` |
| `temperature` | 0.7 | 0.9 | Higher episodic temp for EOS exploration |
| `warmup_ratio` | 0.05 | 0.05 | |
| `save_strategy` | `"steps"` | `"steps"` | |
| `save_steps` | 5 | 5 | |
| `save_total_limit` | 3 | 3 | |
| `logging_steps` | 5 | 5 | |
| `report_to` | `"tensorboard"` if available, else `"none"` | same | |

### Learning rate schedule

Single-step curriculum (stages 1–5): `8e-6 → 5e-6 → 3e-6 → 2e-6 → 1e-6`

Episodic curriculum (stages 1–2 default): `3e-6 (stage 1), 2e-6 (stage 2), 1e-6 (stage 3+)`

### `bf16` / `fp16` flags

Set via `_dtype_flags(model)` which inspects the actual loaded model parameter dtypes:

- bfloat16 model → `(bf16=True, fp16=False)` — clean, no GradScaler
- float16 model (Unsloth 4-bit) → `(bf16=False, fp16=False)` — no AMP, Unsloth handles precision internally
- No GPU → `(False, False)`

Never hard-code these flags — always use `_dtype_flags(model)`.

### Unsloth-specific

```python
config.unsloth_num_chunks = -1
```

Set after `GRPOConfig` construction. The field may not exist in non-Unsloth builds; setting it has no effect if Unsloth isn't installed.

---

## Single-step vs episodic

| Aspect | Single-step (`train_curriculum`) | Episodic (`train_episodic_curriculum`) |
|--------|----------------------------------|----------------------------------------|
| Trainer class | `GRPOTrainer` | `LifeStackGRPOTrainer` |
| Dataset function | `generate_dataset()` | `generate_episodic_dataset()` |
| Completion format | Single JSON object | `{"actions": [...]}` |
| Primary reward signal | `reward_task_success_fn` (env simulation) | `reward_episode_return_fn` (trajectory, weight 2.0) |
| JSON masking | No | Yes (`LifeStackGRPOTrainer._prepare_inputs`) |
| `reward_compact_fn` | Not used (removed) | Not used (removed in v4) |

---

## Reward weights

Stage 1 (single-step warm-up):
```python
reward_weights = [1.0, 1.5, 1.0]
# [reward_format_fn, reward_clean_eos_fn, reward_route_target_fn]
```

Stages 2–5 (single-step full signal):
```python
reward_weights = [1.0, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.5]
# [format, clean_eos, route_target, plausibility, task_success,
#  milestone, replan, reasoning, human_feedback, longterm]
```

Episodic (v4):
```python
reward_weights = [1.0, 0.5, 0.5, 2.0]
# [episode_format, clean_eos, episode_plausibility, episode_return]
```

---

## Related files

- `scripts/train_trl.py` — all config construction
- `docs/train_trl.md` — full training reference
- `docs/training_guide.md` — end-to-end guide
