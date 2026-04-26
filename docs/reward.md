# Reward Design

**Source files:** `core/reward.py`, `scripts/train_trl.py`

---

## Overview

LifeStack uses two reward functions internally and ten GRPO reward functions during training. The internal functions compute the per-step signal during episodes. The GRPO functions wrap those plus environment simulation and run in parallel during each training batch.

The split exists because GRPO needs independent signals with different variances — a single scalar would collapse everything into one gradient, preventing the model from learning which specific behavior to improve.

---

## Internal reward functions

### `compute_reward()`

The base per-step reward used when no task is active (legacy path). It computes four components:

| Component | Formula | Weight |
|-----------|---------|--------|
| `outcome_score` | Weighted sum of positive metric deltas, normalized to `[0,1]` per domain | 0.50 |
| `cascade_containment_score` | `1 - (metrics_worsened / total_metrics)` | 0.25 |
| `resource_efficiency_score` | `1 - ((time/20 + money/500×0.5 + energy/100) / 3)` | 0.15 |
| `relationship_preservation_score` | Sigmoid of average relationship delta: `1 / (1 + exp(-Δrel/10))` | 0.10 |

```
base_reward = 0.50×outcome + 0.25×containment + 0.15×efficiency + 0.10×preservation
```

Then applies penalties and clamps to `[-1.0, 1.0]`.

**Penalties fired inside `compute_reward()`:**

| Name | Condition | Amount |
|------|-----------|--------|
| `CRITICAL_FLOOR_VIOLATION` | Any metric < 20 after the step | -0.50 |
| `CASCADE_SPREAD_WIDER` | `metrics_worsened > disruption_baseline` | -0.30 |
| `INACTION_PENALTY` | `actions_taken == 0` | -0.60 |
| `REST_NOT_JUSTIFIED` | `action_type == "rest"` AND `avg_energy > 30` | -0.25 |
| `RELATIONSHIP_COLLAPSE` | Average relationship delta < -20 | -0.15 |
| `PLAUSIBILITY_VIOLATION` | `reward_plausibility_check()` returns negative | varies |

Hard guarantee: if `actions_taken == 0`, final reward is capped at `min(final_reward, -0.05)` regardless of other components.

---

### `compute_task_reward()`

The task-aware orchestrator called from `LifeStackEnv.step()` when a task is active. It calls `compute_reward()` internally and extracts the `outcome_score` component, then combines it with task-specific signals:

| Component | Source | Weight |
|-----------|--------|--------|
| `milestone_score` | `compute_milestone_reward()` — fraction of milestone rewards earned | 0.35 |
| `completion_score` | `compute_task_completion_reward()` — 1.0 if any success condition met | 0.25 |
| `outcome_score_local` | From `compute_reward()` components — raw metric improvement | 0.15 |
| `replan_score` | `compute_replan_bonus()` — milestones hit after ExoEvents / events seen | 0.10 |
| `efficiency_score` | From `compute_reward()` components | 0.10 |
| `preservation_score` | From `compute_reward()` components | 0.05 |

```
base_reward = 0.35×milestone + 0.25×completion + 0.15×outcome + 0.10×replan + 0.10×efficiency + 0.05×preservation
```

**Penalties fired inside `compute_task_reward()`:**

| Name | Condition | Amount |
|------|-----------|--------|
| `TIMEOUT` | Episode reached `max_steps` without completion | -0.20 |
| `DEAD_END` | No viable routes remaining | -0.50 |
| `ROLLBACK_USED` | Agent used the rollback action | -0.10 |
| `CASCADE_COLLAPSE` | Any metric dropped below 20 between steps | -0.30 |
| `TASK_INACTION_PENALTY` | `actions_taken == 0` | -0.20 |
| `CUMULATIVE_RELATIONSHIP_EROSION` | `cumulative_rel_delta < -20` across episode | -0.15 |

---

## Standalone scoring functions

These live in `core/reward.py` and are called both from within the reward computation and directly as GRPO reward functions in `scripts/train_trl.py`.

### `reward_format_compliance(completion, valid_route_ids=None)`

Scores JSON structure and field validity.

| Return | Condition |
|--------|-----------|
| `+1.0` | Valid JSON, all 5 required keys present and non-None, valid `action_type` in `VALID_ACTION_TYPES`, valid `target_domain` in `VALID_DOMAINS` or a listed route ID |
| `+0.5` | Valid JSON, all 5 keys, but unrecognized `action_type` or `target_domain` |
| `+0.2` | Valid JSON but missing required keys |
| `-0.5` | JSON parse failure |
| `-1.0` | Empty or too-short string, or refusal content ("I cannot", "I'm sorry", "as an ai") |

Required keys: `action_type`, `target_domain`, `metric_changes`, `resource_cost`, `reasoning`.

`VALID_ACTION_TYPES = frozenset({"negotiate", "communicate", "delegate", "spend", "reschedule", "rest", "deprioritize", "execute", "prepare", "self_care"})`

---

### `reward_plausibility_check(metric_changes, resource_cost)`

Anti-gaming gate. Checks whether claimed metric improvements are proportional to resources spent.

```python
ratio = total_metric_delta / max(0.01, normalized_cost)
# normalized_cost = time/20 + money/500 + energy/100
```

| Condition | Return |
|-----------|--------|
| Zero/empty `resource_cost` AND `total_delta > 3.0` | `-0.30` |
| `ratio > 150` | `-0.30` (massive claim, virtually free) |
| `ratio > 80` | `-0.10` (suspicious efficiency) |
| Otherwise | `0.0` |

---

### `reward_timeout_check(step_count, max_steps, done)`

Known limitation: this function only fires when `step_count >= max_steps AND done == False`. Because `done` is already `True` by the time the step terminates (the env sets it before calling the reward), this function currently returns `0.0` in normal operation. The `TIMEOUT` penalty in `compute_task_reward()` is applied directly rather than through this function.

---

### `reward_reasoning_coherence(reasoning, action_type="")`

Checks logical quality and action alignment of the reasoning string. Range: `[-0.30, +0.30]`.

1. `+0.05` if reasoning contains logical connectors ("because", "since", "therefore", "due to", "resulting in", "consequently")
2. If `action_type` is in the keyword map: `+0.10` if reasoning contains action-specific keywords, `-0.20` if it doesn't

Action keyword map (abbreviated):

| `action_type` | Required keywords |
|--------------|------------------|
| `rest` | "energy critically", "exhausted", "burned out", "recharge", "energy below" |
| `communicate` | "talk", "discuss", "speak", "message", "call", "explain" |
| `delegate` | "hand off", "assign", "help", "junior", "colleague" |
| `spend` | "cost", "price", "expensive", "money", "budget", "finance" |
| `execute` | "route", "plan", "action", "implement", "complete", "resolve", "execute" |

A short reasoning string (< 20 chars) returns `-0.20` immediately.

---

## The 10 GRPO reward functions (`scripts/train_trl.py`)

These are the functions passed to `GRPOTrainer` / `LifeStackGRPOTrainer` as `reward_funcs=`. Each takes `completions: list[str]` and `prompts: list[str]`.

| Function | Implementation | Range |
|----------|---------------|-------|
| `reward_format_fn` | Calls `reward_format_compliance()` per completion | `[-1.0, +1.0]` |
| `reward_clean_eos_fn` | Measures trailing chars after JSON close: ≤8→+0.20, ≤32→+0.10, ≤64→0.0, >64→-0.10 | `[-0.10, +0.20]` |
| `reward_route_target_fn` | +0.30 if `target_domain` matches a listed route AND `action_type="execute"`, +0.15 if matches route only | `[0.0, +0.30]` |
| `reward_plausibility_fn` | Calls `reward_plausibility_check()` per completion | `[-0.30, 0.0]` |
| `reward_task_success_fn` | Full env simulation via `get_lifestack_evaluation()`, returns `completion` component | varies |
| `reward_milestone_fn` | Full env simulation, returns `milestone` component | `[0.0, +1.0]` |
| `reward_replan_fn` | Full env simulation, returns `replan` component | `[0.0, +0.5]` |
| `reward_reasoning_fn` | Calls `reward_reasoning_coherence()` per completion | `[-0.30, +0.30]` |
| `reward_human_feedback_fn` | ChromaDB similarity match + `compute_human_feedback_reward()` | `[0.0, +1.0]` |
| `reward_longterm_fn` | 7-day γ=0.9 discounted rollout from `get_lifestack_evaluation()` | `[-1.0, +1.0]` |

`reward_task_success_fn`, `reward_milestone_fn`, `reward_replan_fn`, and `reward_longterm_fn` all call `_cached_lifestack_evaluation()` which runs a single `LifeStackEnv` instance per `(completion, prompt)` pair. The cache is cleared at the start of each batch — `_clear_eval_cache()` is called at the top of `reward_task_success_fn`. This means the 5 env-dependent reward functions share one environment construction per completion rather than constructing 5 separate environments.

`reward_human_feedback_fn` requires ChromaDB and a populated `LifeStackMemory`. On Colab sessions where ChromaDB isn't installed or the DB is empty, it returns `[0.0] * len(completions)` (abstain) rather than penalizing the model.

---

## Stage weights

Stage 1 (JSON warm-up):
```
reward_funcs = [reward_format_fn, reward_clean_eos_fn, reward_route_target_fn]
reward_weights = [1.0, 1.5, 1.0]
```

Stages 2–5 (full signal):
```
reward_funcs = [all 10 above]
reward_weights = [1.0, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.5]
```

Episodic training (v3/v4):
```
reward_funcs = [reward_episode_format_fn, reward_clean_eos_fn,
                reward_episode_plausibility_fn, reward_episode_return_fn]
reward_weights = [1.0, 0.5, 0.5, 2.0]
```

---

## Related files

- `core/reward.py` — all reward logic
- `scripts/train_trl.py` — all `reward_*_fn` GRPO functions
- `core/lifestack_env.py` — `step()` calls `compute_task_reward()`
- `core/feedback.py` — `OutcomeFeedback`, `compute_human_feedback_reward()`
- `core/verifier.py` — `LifeStackVerifier` (used by `get_lifestack_evaluation()`)
