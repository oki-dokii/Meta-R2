# lifestack_env.py — Environment Reference

`core/lifestack_env.py` — The main OpenEnv-compatible RL environment for LifeStack.

---

## Overview

`LifeStackEnv` wraps the full simulation: metric cascades, world events, partial
observability, route execution, milestone tracking, and reward calculation.

Key classes in this file:

| Class | Role |
|---|---|
| `LifeStackAction` | Pydantic action schema (metric_changes, resource_cost, action_type, …) |
| `LifeStackObservation` | Pydantic observation schema (metrics, resources, step, done, reward, metadata) |
| `LifeStackState` | Internal state (current_metrics, budget, task, world_state, hidden_state, …) |
| `PartialObsFilter` | Converts full world state into the agent's partial observation |
| `WorldEngine` | Fires deterministic/probabilistic ExoEvents each step |
| `LifeStackEnv` | The environment itself — inherits from OpenEnv `Environment` |

---

## API

### `LifeStackEnv.__init__(seed, task, max_steps=30)`

```python
env = LifeStackEnv()
env = LifeStackEnv(seed=42, max_steps=50)
```

### `LifeStackEnv.reset(...) -> LifeStackObservation`

```python
obs = env.reset(task=my_task, episode_id="ep_001")
```

Parameters:
- `task` — a `Task` object (from `core/task.py`). Defaults to `FlightCrisisTask()`.
- `seed` — optional int for reproducibility.
- `conflict` — legacy `ConflictEvent` for metric disruption on reset.
- `budget` — dict with `time`, `money`, `energy` overrides.
- `person` — optional `SimPerson` for personality-driven drift.

### `LifeStackEnv.step(action) -> LifeStackObservation`

```python
obs = env.step(LifeStackAction(action_type="execute", target="rebook_premium"))
```

Supported `action_type` values:

| Type | Effect |
|---|---|
| `inspect` | Reveals a hidden-state key into the observation |
| `execute` | Attempts to activate a Route by `target` (route id) |
| `wait` | Passes the step; triggers stress penalty after 4 consecutive waits |
| `rollback` | Reverts metrics/budget to the previous step (one-time per episode) |
| `plan` / `communicate` / `spend` / `delegate` | Apply `metric_changes` and `resource_cost` |

### `LifeStackEnv.render()`

Prints a colour-coded terminal summary of the current state and task progress.

---

## PartialObsFilter

```python
PartialObsFilter.filter(task, revealed_keys) -> dict
```

- Base: `task.visible_world` (always visible).
- Keys in `revealed_keys` that exist in `task.mutable_world` → added as-is.
- Keys in `revealed_keys` that exist in `task.hidden_state` → wrapped as  
  `{"value": <val>, "source": "inspect"}` to signal the agent they came from inspect.

---

## Observation `metadata` fields

```python
obs.metadata = {
    "world_state":     dict,   # partial view after filter
    "goal":            str,
    "active_route":    str | None,
    "milestones":      list[str],
    "events":          list[str],
    "success":         bool,
    "failure":         bool,
    "failure_reason":  str,
    "routes_remaining": int,
    "breakdown":       dict,   # reward component breakdown
    "info":            list[str],  # step-level diagnostic messages
}
```

Key `info` message prefixes:

| Prefix | Meaning |
|---|---|
| `INSPECT_REVEALED:` | Key added to inspected list |
| `INSPECT_REVEALED_HIDDEN:` | Key was in `hidden_state` — value included |
| `INSPECT_REDUNDANT:` | Key already revealed, no-op |
| `ROUTE_SUCCESS:` | Route executed and consequences applied |
| `ROUTE_BLOCKED:` | Route was closed by a prior ExoEvent |
| `PRECONDITIONS_FAILED:` | Route preconditions not met |
| `MILESTONE_UNLOCKED:` | A milestone condition was met |
| `EVENT_FIRED:` | An ExoEvent triggered this step |
| `WAIT_CAP_EXCEEDED:` | 4+ consecutive waits — stress penalty applied |

---

## End Conditions

| Condition | `done` | `success` | `failure` |
|---|---|---|---|
| `step_count >= max_steps` | ✅ | depends | — |
| All `success_conditions` met | ✅ | ✅ | — |
| `failure_condition` met | ✅ | — | ✅ |
| Any metric hits 0 | ✅ | — | ✅ |

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | `PartialObsFilter.filter()` now reads `mutable_world` + `hidden_state` directly from `Task`; removed `world` param; hidden keys wrapped with `source: inspect`; `INSPECT_REVEALED_HIDDEN` info message added |
