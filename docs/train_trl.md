# train_trl.py — GRPO Training Reference

`scripts/train_trl.py` — Curriculum GRPO training via HuggingFace TRL + Unsloth.

---

## Overview

Trains a small LLM (default: `Qwen2.5-1.5B-Instruct`) to resolve LifeStack life conflicts
using **Group Relative Policy Optimization (GRPO)**. Implements a success-based curriculum
that automatically increases difficulty when the agent's average reward exceeds 0.6.

Requires: `unsloth`, `trl`, `datasets`, `transformers`, `accelerate` (Colab / GPU).

---

## Usage

```bash
# Full curriculum training (5 stages × 100 prompts)
python scripts/train_trl.py
```

No CLI args — edit constants at the top of the file to change stages/prompts/output dir.

---

## Architecture

### Reward Functions (multi-signal GRPO)

| Function | Signal |
|---|---|
| `reward_format_fn` | JSON format compliance |
| `reward_plausibility_fn` | Penalises zero-cost metric changes |
| `reward_task_success_fn` | Core env-step outcome reward |
| `reward_milestone_fn` | Milestone progress bonus |
| `reward_reasoning_fn` | Planning coherence score |
| `reward_human_feedback_fn` | Alignment with past real-world outcome feedback |

### `get_lifestack_evaluation(completion, prompt) -> dict`

The central reward computation function. Parses the LLM's JSON completion, reconstructs
the Task from the prompt's `<SYSTEM_METADATA>` block, steps the env, and returns:

```python
{
    "reward": float,
    "breakdown": dict,   # from obs.metadata["breakdown"]
    "action": LifeStackAction
}
```

Returns `{"reward": -0.5, "breakdown": {"error": ...}}` on any parse or env failure.

#### Task Construction Hardening (2026-04-23)

The `Task(...)` call inside `get_lifestack_evaluation` is wrapped in its own
`try/except`. On exception, logs `[reward] Task construction failed: <error>` and
returns the `-0.5` fallback immediately. A field-presence check on
`(id, goal, constraints, mutable_world, visible_world)` follows construction.

### Curriculum (`train_curriculum`)

```
Stage 1: difficulty=1 → train → eval → if avg_reward > 0.6: difficulty++
Stage 2: difficulty=2 → ...
...
Stage 5: difficulty=5 → final save
```

### Dataset (`generate_dataset`)

Generates `N` prompts by:
1. Sampling a `TaskGenerator` task (flight_crisis or code_merge_crisis)
2. Merging a legacy `ConflictEvent` disruption for variety
3. Cascading the disruption through the `DependencyGraph`
4. Embedding task metadata in a `<SYSTEM_METADATA>` block for reward reconstruction

---

## Outputs

| Path | Contents |
|---|---|
| `./lifestack_model/` | Final saved model + tokenizer |
| `./lifestack_model/stage_N/` | Per-stage checkpoints |
| `training_logs/generations.jsonl` | Sampled generations (every 20 reward calls) |
| `grpo_reward_curve.png` | 50-episode eval reward curve |

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | `Task()` construction wrapped in try/except + field validation; returns -0.5 fallback on failure |
