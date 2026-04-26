# LifeStack Environment

**Source file:** `core/lifestack_env.py`

---

## Overview

`LifeStackEnv` is the central environment class. It inherits from `openenv.core.Environment` when the modern API is available, and falls back through two shim layers when it isn't. The `USING_MODERN_API` flag at module level controls which base class is used — the environment is fully functional in both modes.

---

## API compatibility shim

`lifestack_env.py` tries three import paths in order:

1. `from openenv.core import Environment, Action, Observation, State` — modern API (preferred)
2. `from openenv.env import Env as Environment` — legacy openenv path
3. A minimal inline shim with no-op `reset()` and `step()` — for import-only contexts (docs, CI)

The `USING_MODERN_API` flag is set `True` only when path 1 succeeds. `server.py` checks this flag before starting `uvicorn` and exits gracefully if it's `False`.

---

## Core classes

### `LifeStackAction`

The agent's output. Inherits from `openenv.core.Action`.

| Field | Type | Purpose |
|-------|------|---------|
| `action_type` | `str \| None` | One of the 10 valid types, or `"inspect"`, `"rollback"`, `"wait"` |
| `target` | `str \| None` | Target domain or route ID |
| `metric_changes` | `dict[str, float]` | Requested delta per metric path |
| `resource_cost` | `dict[str, float]` | `{"time": h, "money": $, "energy": pts}` |
| `reasoning` | `str \| None` | Plain text reasoning for reward scoring |
| `completion` | `str \| None` | The raw model completion (full JSON string) |
| `is_rollback` | `bool` | Set `True` to trigger the rollback action |
| `inspect_target` | `str \| None` | Hidden-state key to reveal |
| `actions_taken` | `int` | Count of atomic actions (used in reward) |

### `LifeStackObservation`

What the environment returns after each step.

| Field | Description |
|-------|-------------|
| `metrics` | Flattened 23-value dict: `"career.satisfaction": 72.5`, etc. |
| `resources` | `{"time": h, "money": $, "energy": pts}` remaining |
| `step` | Current step count |
| `done` | Whether the episode has ended |
| `reward` | Scalar reward for this step |
| `metadata` | Dict with `world_state`, `goal`, `active_route`, `milestones`, `events`, `success`, `failure`, `failure_reason`, `routes_remaining`, `conflict_title`, `person` |

After `step()`, `metadata` also contains `breakdown` (full reward component dict) and `info` (list of string log messages like `"WAIT_CAP_EXCEEDED"`, `"ROUTE_SUCCESS: ..."`, etc.)

### `LifeStackState`

Internal mutable state. Not returned to the agent directly; accessible via `env.state`.

| Field | Description |
|-------|-------------|
| `current_metrics` | `LifeMetrics` instance |
| `budget` | `ResourceBudget` instance |
| `step_count` | Current episode step |
| `consecutive_waits` | Counter for WAIT_CAP_EXCEEDED trigger |
| `used_rollback` | Whether rollback has been used this episode |
| `rollback_penalty_charged` | Prevents double-charging the rollback penalty |
| `previous_metrics` | Snapshot for rollback (overwritten each non-rollback step) |
| `previous_budget` | Snapshot for rollback |
| `current_task` | Active `Task` object |
| `world_state` | Mutable world dict (deep copy of `task.mutable_world`) |
| `hidden_state` | Hidden truth dict (agent can only read via inspect) |
| `inspected_keys` | List of keys the agent has revealed via inspect |
| `milestones_achieved` | List of milestone IDs hit this episode |
| `fired_event_ids` | List of ExoEvent IDs that have fired |
| `exo_events_seen` | Count of events fired (for replan bonus) |
| `milestones_after_event` | Milestones hit after at least one event fired |
| `closed_route_ids` | Set of route IDs closed by events or previous routes |
| `cumulative_rel_delta` | Running sum of relationship metric delta (for erosion penalty) |

---

## Methods

### `reset(seed, episode_id, task, conflict, budget, person, agent_history)`

Signature:
```python
def reset(self, seed=None, episode_id=None, task=None, conflict=None,
          budget=None, person=None, agent_history=None, **kwargs) -> LifeStackObservation
```

1. Seeds `random` if `seed` is provided (for reproducibility)
2. Sets `current_task` to the provided `task`, or falls back to `FlightCrisisTask()`
3. Sets `max_steps = task.horizon`
4. Resets all `LifeStackState` fields to zero/empty
5. Deep-copies `task.mutable_world` → `world_state`, `task.hidden_state` → `hidden_state`
6. Scales `ResourceBudget` from task constraints: `base_budget × max(1.0, max_steps/5.0)`
7. If `conflict` is provided (legacy support), runs `DependencyGraph.cascade()` on the disruption dict to seed metric state
8. Constructs a new `WorldEngine(task)`
9. Returns `_get_obs()`

---

### `step(action, timeout_s=None)`

Executes one environment step. Order of operations inside `step()`:

1. **Personality drift & legacy escalation** — if `state.person` is set, calls `person.drift(step_count)` for personality-driven metric nudges. At step 2, calls `adaptive_escalate()` to potentially increase conflict difficulty.
2. **ExoEvent injection** — `WorldEngine.inject_events()` fires any events scheduled for this step (or probabilistic events), mutating `world_state` and `hidden_state`.
3. **Tool dispatch** — determines `tool_type` from `action.action_type`, `action.is_rollback`, or `action.inspect_target`.
4. **Rollback handling** — if `tool_type == "rollback"`: checks `used_rollback` flag, restores previous metrics/budget if available, charges `-0.1` reward.
5. **Pre-action snapshot** — saves `current_metrics` and `budget` to `previous_*` for future rollback.
6. **Inspect handling** — adds `action.target` to `inspected_keys`, logs revealed value if it's in `hidden_state`.
7. **Wait cap** — increments `consecutive_waits`; at 4+, forces `+15.0` to `mental_wellbeing.stress_level`.
8. **Route execution** — if `action.target` matches a `viable_routes` entry and preconditions are met, applies route consequences to `world_state`.
9. **Resource deduction** — calls `budget.deduct()`. If the budget is insufficient, discards all `metric_changes` (the action is blocked but the step still counts).
10. **Metric updates and cascade** — changes `≤ 5.0` in absolute value are applied directly via `_update_metric()`. Changes `> 5.0` are passed through `DependencyGraph.cascade()` with the full propagation logic.
11. **Task progression** — `LifeStackVerifier` checks success, failure, and new milestones.
12. **Reward calculation** — calls `compute_task_reward()` (or `compute_reward()` if no task).
13. **Termination check** — episode terminates on: any success condition met (truncated), `step_count >= max_steps` (truncated), any metric `<= 10` (terminated), task failure conditions (terminated), or all routes exhausted (failure).

Returns `LifeStackObservation`.

---

### `rollout(n_steps=7, gamma=0.9)`

```python
def rollout(self, n_steps=7, gamma=0.9) -> dict
```

Runs `n_steps` null rest actions from the current state and returns the discounted cumulative reward plus a step-by-step trajectory. Fully side-effect-free — deep-copies `_internal_state` before starting and restores it after.

Return value:
```python
{
    "discounted_reward": float,     # γ-discounted cumulative reward
    "trajectory": [                 # one entry per simulated step
        {
            "step": int,
            "reward": float,
            "metrics": dict,        # flattened LifeMetrics snapshot
            "discounted_contribution": float,
        }
    ],
    "n_steps_completed": int
}
```

This is the signal used by `reward_longterm_fn` in `scripts/train_trl.py`.

---

## WorldEngine

`WorldEngine` manages `ExoEvent` injection and route closure tracking.

```python
class WorldEngine:
    def __init__(self, task: Task)
    def inject_events(self, step, world, hidden) -> list[ExoEvent]
    def get_closed_routes(self) -> set[str]
```

`inject_events()` fires events when `event.step == current_step` (deterministic) or when `random.random() < event.probability` and `event.step == -1` (probabilistic). Fired events mutate `world` and `hidden` in place and add their `closes_routes` to the closed set.

---

## PartialObsFilter

Controls what the agent sees as `world_state` in the observation:

```python
class PartialObsFilter:
    @staticmethod
    def filter(task: Task, revealed_keys: list) -> dict
```

Starts with a deep copy of `task.visible_world` (the always-visible subset). For each key in `revealed_keys` (keys the agent has `inspect`ed), overlays the corresponding value from `mutable_world` if found, or from `hidden_state` wrapped as `{"value": ..., "source": "inspect"}` if it's a hidden field.

This is the mechanism that makes `hidden_state` meaningful — the agent genuinely cannot see `card_available: True` until it spends an inspect action.

---

## Rollback mechanic

`action_type="rollback"` is available once per episode. The `LifeStackState.used_rollback` flag prevents using it more than once. When triggered:

1. If `used_rollback == True`: returns `reward=-0.1` immediately (denied)
2. If `previous_metrics` is `None`: returns `reward=0.0` (nothing to restore)
3. Otherwise: restores `current_metrics` and `budget` from `previous_*` snapshots, sets `used_rollback = True`, returns `reward=-0.1`

The `-0.1` penalty fires on the rollback action itself. The `ROLLBACK_USED` penalty in `compute_task_reward()` applies separately for the episode-level record.

---

## Related files

- `core/life_state.py` — `LifeMetrics`, `DependencyGraph`, `ResourceBudget`
- `core/task.py` — `Task`, `ExoEvent`, `Route`, `Milestone`, `FlightCrisisTask`
- `core/reward.py` — `compute_task_reward()`, `compute_reward()`
- `core/verifier.py` — `LifeStackVerifier`
- `server.py` — starts `create_app(env=LifeStackEnv)` on port 8000
