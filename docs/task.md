# Task Schema

**Source file:** `core/task.py`

---

## Overview

A `Task` object defines everything about one episode: the goal, the state the agent operates in, what can happen during the episode, how success is measured, and how long the agent has. `LifeStackEnv.reset()` takes a `Task` and builds the entire episode state from it.

---

## `Task` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique task identifier |
| `domain` | `str` | Task domain: `"flight_crisis"`, `"code_merge_crisis"`, etc. |
| `goal` | `str` | Human-readable objective shown to the agent |
| `constraints` | `dict` | Budget and deadline keys, e.g. `{"budget_max": 400, "deadline_step": 18}` |
| `hidden_state` | `dict` | Full truth; agent cannot see directly (e.g. `{"card_available": True}`) |
| `mutable_world` | `dict` | Partial truth; some keys visible, some only revealed by inspect |
| `visible_world` | `dict` | Always-visible subset of `mutable_world` |
| `success_conditions` | `list[dict]` | Terminal predicates, e.g. `[{"key": "flight_rebooked", "value": True}]` |
| `failure_conditions` | `list[dict]` | Episode-ending failure predicates |
| `event_schedule` | `list[ExoEvent]` | Events that fire during the episode |
| `viable_routes` | `list[Route]` | Paths the agent can execute |
| `milestones` | `list[Milestone]` | Intermediate progress gates |
| `horizon` | `int` | Max steps (e.g. 30 for `FlightCrisisTask`, 10 for `CodeMergeCrisisTask`) |
| `difficulty` | `int` | 1–5 curriculum index |
| `domain_metadata` | `dict` | Story text and domain-specific generator hints |

---

## `Route` dataclass

A route is a structured path the agent can execute by targeting it with `action_type="execute"` (or any matching `required_action_types` entry). When a route's preconditions are met and it's executed, its `consequences` are applied to `world_state`, which can trigger success conditions.

| Field | Description |
|-------|-------------|
| `id` | Route identifier, shown to agent in prompt |
| `name` | Human-readable name |
| `required_action_types` | The model must use one of these action types to execute the route |
| `preconditions` | World/hidden state checks that must be true before the route is available |
| `consequences` | World state mutations on route completion |
| `closes_routes` | Route IDs that become unavailable after this route is taken |
| `milestones_unlocked` | Milestone IDs this route can trigger |
| `final_reward` | Bonus added on route completion |

Example from `FlightCrisisTask`:
```python
Route(
    id="rebook_premium",
    name="Rebook Premium Option",
    required_action_types=["communicate", "execute"],
    preconditions={"card_available": True},
    consequences={"flight_rebooked": True},
    closes_routes=["wait_lounge"],
    final_reward=2.5
)
```

---

## `Milestone` dataclass

Intermediate checkpoints that reward partial progress. `LifeStackVerifier.check_new_milestones()` scans milestones every step.

| Field | Description |
|-------|-------------|
| `id` | Milestone identifier |
| `condition_key` | World/hidden key to check |
| `condition_value` | Required value |
| `reward` | Added to episode reward when milestone is hit |

---

## `ExoEvent` dataclass

World events that fire during the episode, potentially changing state and closing routes.

| Field | Description |
|-------|-------------|
| `step` | Fire at this step; -1 = probabilistic |
| `probability` | 1.0 = always fire; <1.0 = fire with this probability when step=-1 |
| `world_mutation` | Dict applied to `world_state` when event fires |
| `hidden_state_mutation` | Dict applied to `hidden_state` when event fires |
| `closes_routes` | Route IDs made unavailable after this event |

---

## Built-in task factories

`FlightCrisisTask()` — "Survive Airport Cancellation", horizon=30, difficulty=4. Two competing routes (`rebook_premium` requires `card_available=True`; `wait_lounge` requires `lounge_access=True`). Two timed events: a `price_surge` at step 5 sets `card_available=False`, and `lounge_full` at step 8 closes `wait_lounge`.

`CodeMergeCrisisTask()` — "Resolve Production Outage", horizon=10, difficulty=4. Two competing routes: revert the commit (`revert_commit`) or push a hotfix (`hotfix`). No scheduled events.

`TaskGenerator` in `core/task.py` holds only these two. The richer `TaskGenerator` in `agent/conflict_generator.py` covers all 8 task domains with template-based generation and is the one used by `scripts/train_trl.py`.

---

## Related files

- `core/lifestack_env.py` — `WorldEngine` fires `ExoEvent` objects during `step()`
- `core/verifier.py` — `LifeStackVerifier` checks success/failure/milestone conditions
- `agent/conflict_generator.py` — full `TaskGenerator` for all 8 domains
- `docs/lifestack_env.md` — how tasks integrate with the environment
