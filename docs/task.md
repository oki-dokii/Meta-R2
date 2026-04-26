# `core/task.py` — task schema

Dataclasses that define a **long-horizon episode**: goals, routes, milestones, hidden state, and scheduled events.

---

## `Task`

| Field | Meaning |
|-------|---------|
| `id` | Stable task id |
| `domain` | e.g. `flight_crisis`, `code_merge_crisis`, `transport_crisis` |
| `goal` | Human-readable objective |
| `constraints` | Budget / deadline keys |
| `hidden_state` / `mutable_world` / `visible_world` | Partial observability layers |
| `success_conditions` / `failure_conditions` | Terminal predicates |
| `event_schedule` | `ExoEvent` list — timed or probabilistic world changes |
| `viable_routes` | `Route` objects the agent can `execute` |
| `milestones` | Progress gates for rewards |
| `horizon` | Max steps |
| `difficulty` | **1–5** curriculum index |
| `domain_metadata` | Extra story / generator hints |

---

## `Route`

Defines an actionable path: preconditions, metric effects, costs, and narrative id.

---

## `Milestone`

Intermediate success checkpoints — exposed to **milestone** reward heads in full single-step curriculum stages.

---

## `ExoEvent`

World events (deterministic or sampled) that may fire during `env.step` and alter routes or metrics.

---

## JSON actions (GRPO)

The trained policy outputs JSON aligned with **`LifeStackAction`** fields:

- `action_type` ∈ {`negotiate`, `communicate`, `delegate`, `spend`, `reschedule`, `rest`, `deprioritize`, `execute`}  
- `target_domain` or route id must match **`VALID_DOMAINS`** / prompt route list  

See `core/reward.py:reward_format_compliance`.

---

## See also

- [lifestack_env.md](lifestack_env.md)  
- [conflict_generator.md](conflict_generator.md)  
- [reward.md](reward.md)  
