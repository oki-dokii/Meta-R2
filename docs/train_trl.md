# Training with TRL + GRPO

**Source file:** `scripts/train_trl.py`

---

## Overview

`train_trl.py` trains `Qwen2.5-1.5B-Instruct` on the LifeStack environment using Group Relative Policy Optimization (GRPO) via HuggingFace TRL. Two training modes exist: `train_curriculum()` (5 stages, single-step actions) and `train_episodic_curriculum()` (multi-stage, multi-step action sequences). Both support checkpoint resumption via `curriculum_state.json`.

---

## Quick start

```bash
# Full curriculum (5 stages, ~2h on A100 80GB)
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py

# Dry run — validates the full pipeline on CPU in ~60s (uses tiny-gpt2 as placeholder)
python scripts/train_trl.py --dry-run

# Resume from checkpoint
python scripts/train_trl.py --resume

# Start from a specific stage
python scripts/train_trl.py --start-stage 3

# Push to HF Hub after training
python scripts/train_trl.py --push-to-hub
```

---

## Model loading (`load_model()`)

`load_model()` tries two paths in order:

**Path 1 (Unsloth):** `FastLanguageModel.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct", load_in_4bit=True)` with QLoRA (`r=16, lora_alpha=16`, all 7 projection layers). If any part of this fails, falls back to Path 2.

**Path 2 (HF+PEFT):** `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")` in bf16 (Ampere+) or fp16 (older GPUs), then `get_peft_model()` with the same `LoraConfig(r=16, lora_alpha=16)`.

Set `LIFESTACK_NO_UNSLOTH=1` to skip Path 1 entirely. This is recommended on A100 80GB — the HF+PEFT path is bf16 end-to-end with no Unsloth kernel surprises. The reason this flag exists is a dtype mismatch: Unsloth's 4-bit checkpoint bakes in `bnb_4bit_compute_dtype=float16`, which collides with LoRA gradient precision in ways that can't be fixed by changing the `dtype=` parameter.

`load_model_for_dry_run()` loads `sshleifer/tiny-gpt2` (CPU-compatible, no download) and is only used when `--dry-run` is passed.

---

## LifeStackGRPOTrainer

A `GRPOTrainer` subclass that adds JSON-boundary gradient masking.

```python
class LifeStackGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs): ...
```

**Problem:** After learning to write valid JSON, the model continues generating hundreds of tokens of free-form explanation. All 480 max-completion tokens are included in the loss, so the gradient for the ~100 meaningful JSON tokens is diluted by 380 tokens of trailing text.

**Fix:** `_prepare_inputs()` decodes each completion, finds the character position of the first complete JSON object's closing brace via `_find_json_end_text()`, estimates the corresponding token index, and zeroes out `completion_mask` beyond that cutoff. The token IDs are not changed — no distribution shift. Only the mask changes, so KL and advantage terms see only the JSON payload.

Effect: effective gradient length drops from ~480 to ~100 tokens → roughly 3–5× sharper policy updates per step.

`LifeStackGRPOTrainer` is used in `train_episodic_curriculum()` only. The single-step `train_curriculum()` uses plain `GRPOTrainer` — the episodic setting benefits more from the masking because action sequences are inherently longer.

---

## Single-step curriculum (`train_curriculum()`)

```python
def train_curriculum(n_stages=5, n_prompts_per_stage=100, output_dir="./lifestack_model",
                     resume=False, start_stage=None, max_prompt_length=2048,
                     max_completion_length=224)
```

Five stages. Each uses `GRPOTrainer` with `per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4`.

| Stage | LR | Reward functions | Notes |
|-------|----|-----------------|-------|
| 1 | 8e-6 | `format`, `clean_eos`, `route_target` — weights `[1.0, 1.5, 1.0]` | JSON warm-up only |
| 2 | 5e-6 | All 10 — weights `[1.0, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.5]` | Full signal introduced |
| 3 | 3e-6 | All 10 | Difficulty advances if stage-end reward ≥ 0.0 |
| 4 | 2e-6 | All 10 | |
| 5 | 1e-6 | All 10 | |

Dataset is generated fresh per stage via `generate_dataset(n_prompts, difficulty=curr_diff)`. Prompts are sampled round-robin across all 8 domains (`career`, `finances`, `relationships`, `physical_health`, `mental_wellbeing`, `time`, `transport_crisis`, `code_merge_crisis`) with difficulty cycling 1–5 unless fixed.

Each stage checkpoint is saved to `{output_dir}/stage_{N}/` every 5 optimizer steps. Curriculum state (completed stage + current difficulty) is persisted to `curriculum_state.json` after each stage, enabling clean resume across interrupted sessions.

---

## Episodic curriculum (`train_episodic_curriculum()`)

```python
def train_episodic_curriculum(n_stages=2, n_episodes_per_stage=40,
                               output_dir="./lifestack_model_v2", episode_horizon=3, ...)
```

Instead of single-action prompts, each prompt asks the model to return `{"actions": [...]}` — a sequence of up to `episode_horizon` actions. The real `LifeStackEnv` executes the sequence step-by-step and returns a trajectory-level reward.

Four reward functions with weights `[1.0, 0.5, 0.5, 2.0]`:

| Function | Weight | Signal |
|----------|--------|--------|
| `reward_episode_format_fn` | 1.0 | Average `reward_format_compliance()` across all actions in the sequence |
| `reward_clean_eos_fn` | 0.5 | EOS cleanliness (same as single-step) |
| `reward_episode_plausibility_fn` | 0.5 | Average `reward_plausibility_check()` across all actions |
| `reward_episode_return_fn` | 2.0 | Discounted trajectory reward + terminal bonus (±0.25) |

`reward_compact_fn` was removed in v4. It was flat at `-0.5` with std ≈ 0 throughout the v3 training run (pure constant drag on the reward total, zero gradient). Once `LifeStackGRPOTrainer`'s JSON boundary masking handles compactness structurally, the explicit penalty adds nothing.

---

## Prompt format

Each training prompt is built by `build_prompt_for_task()`:

```
You are LifeStack. Return ONLY compact JSON.
<SYSTEM_METADATA>
{"domain":"flight_crisis","disruption":{...},"difficulty":3,"seed":12345,"step":0,"route_ids":["rebook_premium","wait_lounge"],"budget":{"time":20.0,"money":500.0,"energy":100.0}}
</SYSTEM_METADATA>
Task: Survive Airport Cancellation
Story: A major storm grounded commercial flights.
Key metrics:
- career.workload: 75.0
- finances.liquidity: 30.0
- relationships.romantic: 55.0
- physical_health.energy: 60.0
- mental_wellbeing.stress_level: 80.0
Budget: time=20.0, money=500.0, energy=100.0
Routes (max 2):
- rebook_premium: Rebook Premium Option (needs communicate, execute)
- wait_lounge: Accept Delay & Work (needs wait, plan)
Required keys: action_type, target_domain, metric_changes, resource_cost, reasoning.
{"action_type": "negotiate|communicate|...", ...}
```

The `<SYSTEM_METADATA>` block is parsed by `_metadata_from_prompt()` inside `get_lifestack_evaluation()` to reconstruct the exact task, seed, and budget state for reward computation. Without this block, the reward functions cannot evaluate the completion.

Episodic prompts append an `episode_history` block and ask for `{"actions": [...]}` format.

---

## Reward function caching

Five reward functions call `get_lifestack_evaluation()` which constructs a full `LifeStackEnv` instance. Without caching, a batch of 4 completions would construct 5×4 = 20 separate environments. `_cached_lifestack_evaluation()` caches by `(completion, prompt)` tuple. `_clear_eval_cache()` is called at the start of `reward_task_success_fn` (which is always first in the function list), clearing stale entries from the previous batch.

---

## TRL compatibility shims

`train_trl.py` installs lightweight shims for `mergekit`, `llm_blender`, `weave`, and `vllm` before importing TRL. These packages are imported at TRL's module level but never used in GRPO codepaths. The shims prevent `ImportError` on Colab/Kaggle images that don't have these packages.

`_make_grpo_config()` wraps `GRPOConfig(**kwargs)` and silently drops any kwargs not supported by the installed TRL version. This makes the script compatible across TRL 1.x minor versions where `GRPOConfig` fields differ.

---

## Checkpoint management

```bash
# State file location
{output_dir}/curriculum_state.json
# Content: {"completed_stage": 3, "next_difficulty": 2}

# Resume training
python scripts/train_trl.py --resume --output-dir ./lifestack_model

# The trainer also saves step-level checkpoints inside each stage dir:
{output_dir}/stage_3/checkpoint-25/
```

If a stage is interrupted mid-way, `--resume` uses `find_latest_checkpoint()` to locate the most recent `checkpoint-N` folder and passes it to `trainer.train(resume_from_checkpoint=...)`. The Trainer reloads model weights + optimizer state from the checkpoint.

---

## Related files

- `core/reward.py` — standalone reward scoring functions
- `core/lifestack_env.py` — `LifeStackEnv` used in reward evaluation
- `agent/conflict_generator.py` — `TaskGenerator` with all 8 domains
- `intake/simperson.py` — `SimPerson` personality profiles (5 pooled during training)
- `notebooks/Colab_GRPO_Training.ipynb` — re-runnable Colab notebook version
