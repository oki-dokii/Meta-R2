# `lifestack_env.py` — LifeStackEnv reference

`core/lifestack_env.py` — OpenEnv-compatible environment for **LifeStack**: multi-domain metrics, **dependency graph** cascades, tasks, events, and rewards.

**Manifest:** `openenv.yaml` → `core.lifestack_env:LifeStackEnv`

---

## Architecture overview

| Piece | Role |
|-------|------|
| **Eight domains** | `career`, `finances`, `relationships`, `physical_health`, `mental_wellbeing`, `time`, `transport_crisis`, `code_merge_crisis` (see also legacy `flight_crisis` in some tasks) |
| **Sub-metrics** | Nested paths such as `career.job_security` — flattened for rewards and observations |
| **DependencyGraph** | Directed influences between metrics — e.g. job loss → stress → sleep → health |
| **TaskGenerator** | Domain-specific long-horizon `Task` objects (`core/task.py`, `agent/conflict_generator.py`) |
| **SimPerson** | Big-Five profiles (`intake/simperson.py`) — training prompts use personas such as **Alex, Chloe, Sam, Jordan, Maya**; packaged demo data may list variants (e.g. **Leo** in `data/simperson_profiles.json`) |
| **ResourceBudget** | `time_hours`, `money_dollars`, `energy_units` (exposed as time / money / energy in actions) |
| **WorldEngine / events** | ExoEvents can fire mid-episode and block routes or shift metrics |
| **`rollout(n_steps=7, gamma=0.9)`** | Long-horizon discounted trajectory used in some reward/analysis paths |

---

## Episodic training (GRPO)

Training does not only score **one** JSON action. In **episodic** mode (`scripts/train_trl.py --episode-train`):

- **Horizon** (default **3**): the model proposes a short **sequence** of actions.
- After each **step**, the environment updates metrics, fires cascades, and returns **`obs.reward`**.
- **Episode return** in rewards is a **discounted sum** of step rewards plus terminal shaping (see `get_episode_evaluation` in `train_trl.py`).
- **Curriculum difficulty** advances when stage mean reward ≥ **0** (see `curriculum_state.json`).

This matches the design: crises **compound** across domains over several decisions.

---

## Key classes

| Class | Role |
|-------|------|
| `LifeStackAction` | Pydantic action — `action_type`, `target`, `metric_changes`, `resource_cost`, … |
| `LifeStackObservation` | Metrics, resources, step, `done`, `reward`, `metadata` |
| `LifeStackState` | Internal metrics, budget, task, world / hidden state |
| `PartialObsFilter` | Hides `hidden_state` until `inspect` |
| `WorldEngine` | Schedules / samples **ExoEvents** |
| `LifeStackEnv` | `reset`, `step`, `render`; subclasses OpenEnv **`Environment`** |

---

## API sketch

### `LifeStackEnv.__init__(seed=None, task=None, max_steps=30)`

```python
env = LifeStackEnv()
env = LifeStackEnv(seed=42, max_steps=50)
```

### `LifeStackEnv.reset(...) -> LifeStackObservation`

```python
obs = env.reset(task=my_task, episode_id="ep_001", conflict=disruption_dict)
```

Common kwargs: `task`, `seed`, `conflict` (disruption dict), `budget`, `person` (`SimPerson`).

### `LifeStackEnv.step(action) -> LifeStackObservation`

```python
obs = env.step(LifeStackAction(action_type="execute", target="rebook_premium"))
```

Supported `action_type` values include: `inspect`, `execute`, `wait`, `rollback`, and declarative types such as `plan`, `communicate`, `spend`, `delegate` that apply `metric_changes` / `resource_cost` (see source for the full dispatch table).

---

## Observation `metadata` (representative)

```python
obs.metadata = {
    "world_state": dict,
    "goal": str,
    "active_route": str | None,
    "milestones": list,
    "events": list,
    "success": bool,
    "failure": bool,
    "failure_reason": str,
    "routes_remaining": int,
    "breakdown": dict,
    "info": list[str],
}
```

`info` prefixes include `ROUTE_SUCCESS:`, `EVENT_FIRED:`, `MILESTONE_UNLOCKED:`, `WAIT_CAP_EXCEEDED:`, etc.

---

## End conditions

| Condition | `done` | `success` / `failure` |
|-----------|--------|------------------------|
| Step limit | ✅ | context-dependent |
| All success conditions | ✅ | `success` |
| Failure condition | ✅ | `failure` |
| Metric floor (implementation) | ✅ | `failure` may set |

---

## Change log

| Date | Change |
|------|--------|
| 2026-04-26 | Doc refresh: eight domains, episodic GRPO, personas, dependency graph |
| 2026-04-23 | `PartialObsFilter` reads `mutable_world` + `hidden_state` from `Task` |

---

## See also

- [task.md](task.md) — `Task` / `Route` schema  
- [reward.md](reward.md) — how `obs.reward` feeds GRPO  
- [train_trl.md](train_trl.md) — episodic flags  
