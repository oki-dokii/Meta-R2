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

# CPU smoke test for the single-step path
python scripts/train_trl.py --dry-run

# CPU smoke test for the episode-level reward path
python scripts/train_trl.py --episode-train --dry-run

# Episodic v2 fine-tune after a short single-step warm-up
python scripts/train_trl.py --episode-train --stages 2 --episodes-per-stage 40 --episode-horizon 3

# Push a validated adapter
python scripts/train_trl.py --episode-train --push-to-hub --hub-repo-id jdsb06/lifestack-grpo-v2
```

Key CLI args: `--stages`, `--prompts-per-stage`, `--output-dir`, `--resume`,
`--start-stage`, `--full-episode`, `--episode-train`, `--episode-horizon`,
`--episodes-per-stage`, `--episode-warmup-stages`, `--push-to-hub`, and
`--hub-repo-id`.

---

## Architecture

### Reward Functions (single-step GRPO)

| Function | Signal |
|---|---|
| `reward_format_fn` | JSON format compliance |
| `reward_clean_eos_fn` | Stops cleanly after the first JSON object |
| `reward_route_target_fn` | Rewards using listed route ids so milestones/completion can fire |
| `reward_plausibility_fn` | Penalises zero-cost metric changes |
| `reward_task_success_fn` | Core env-step outcome reward |
| `reward_milestone_fn` | Milestone progress bonus |
| `reward_replan_fn` | Recovery after exogenous events |
| `reward_reasoning_fn` | Planning coherence score |
| `reward_human_feedback_fn` | Alignment with past real-world outcome feedback |
| `reward_longterm_fn` | 7-step discounted rollout after the chosen action |

### Reward Functions (episodic v2)

| Function | Signal |
|---|---|
| `reward_episode_format_fn` | Validates every action in `{"actions": [...]}` |
| `reward_clean_eos_fn` | Penalises trailing text after the episode JSON |
| `reward_episode_plausibility_fn` | Averages anti-hacking plausibility over the sequence |
| `reward_episode_return_fn` | Executes the action sequence in `LifeStackEnv` and scores discounted trajectory return + terminal success |

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

### Episodic Curriculum (`train_episodic_curriculum`)

The v2 path trains on prompts that ask for compact action sequences:

```json
{"actions":[{"action_type":"execute","target_domain":"<route_id>","metric_changes":{},"resource_cost":{"time":0.5,"money":0,"energy":2},"reasoning":"brief"}]}
```

`reward_episode_return_fn` parses the sequence, steps the real environment up to
`--episode-horizon`, and returns a clipped trajectory reward. The post-training
`--full-episode` runner still performs closed-loop evaluation where the model
generates one action, observes the new state, then generates the next action.

### Dataset (`generate_dataset`)

Generates `N` prompts by:
1. Sampling a `TaskGenerator` task across the 8 supported domains
2. Merging a legacy `ConflictEvent` disruption for variety
3. Cascading the disruption through the `DependencyGraph`
4. Embedding task metadata in a `<SYSTEM_METADATA>` block for reward reconstruction

`generate_episodic_dataset` follows the same reconstruction pattern but adds
route ids and horizon metadata so route-completion rewards can be assigned
without confusing route ids with life-domain names.

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
| 2026-04-26 | Added explicit route-id format credit, neutral missing human feedback, `--episode-train`, episodic dry-run, and action-sequence trajectory reward |
