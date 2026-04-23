# task.py — Task Schema Reference

`core/task.py` — Dataclass definitions for the LifeStack long-horizon episode schema.

---

## Overview

A `Task` is the complete specification of a single episode. It defines what the agent
must achieve, how the world can change around it, and what routes are available.

---

## Dataclasses

### `Task`

```python
@dataclass
class Task:
    id: str                          # Unique task identifier
    domain: str                      # e.g. "flight_crisis", "code_merge_crisis"
    goal: str                        # Human-readable goal description
    constraints: dict                # e.g. {"budget_max": 800, "deadline_step": 10}
    hidden_state: dict               # Keys not visible without inspect
    mutable_world: dict              # Keys that can change during the episode
    visible_world: dict              # Keys always visible in the observation
    success_conditions: list[dict]   # [{key, value}] — all must be met
    failure_conditions: list[dict]   # [{key, value}] — any triggers failure
    event_schedule: list[ExoEvent]   # Deterministic/probabilistic events
    viable_routes: list[Route]       # Available action paths
    milestones: list[Milestone]      # Progress checkpoints
    horizon: int                     # Max steps per episode
    difficulty: int                  # 1–5 scale
    domain_metadata: dict            # Free-form extra info (e.g. {"story": "..."})
```

### `Route`

```python
@dataclass
class Route:
    id: str
    name: str
    description: str
    required_action_types: list[str]  # e.g. ["communicate", "spend"]
    preconditions: dict               # World/hidden state conditions that must be true
    consequences: dict                # World state mutations on success
    closes_routes: list[str]          # Route IDs that become unavailable after this
    milestones_unlocked: list[str]    # Milestone IDs unlocked on route success
    final_reward: float               # Bonus reward on route completion
```

### `Milestone`

```python
@dataclass
class Milestone:
    id: str
    description: str
    condition_key: str    # World/hidden state key to check
    condition_value: Any  # Value it must equal for milestone to be met
    reward: float         # Reward added when milestone is first reached
```

### `ExoEvent`

```python
@dataclass
class ExoEvent:
    step: int             # Step at which to fire (-1 = probabilistic each step)
    probability: float    # Firing probability if step == -1
    id: str
    description: str
    world_mutation: dict         # Applied to mutable_world on fire
    hidden_state_mutation: dict  # Applied to hidden_state on fire
    closes_routes: list[str]     # Routes closed when this event fires
```

---

## Built-in Tasks

| Class | Domain | Description |
|---|---|---|
| `FlightCrisisTask` | `flight_crisis` | Cancelled flight — rebook or work from lounge |

---

## Creating a Custom Task

```python
from core.task import Task, Route, Milestone, ExoEvent

my_task = Task(
    id="my_task",
    domain="my_domain",
    goal="Do the thing",
    constraints={"budget_max": 500, "deadline_step": 8},
    hidden_state={"secret_key": True},
    mutable_world={},
    visible_world={"public_info": "visible"},
    success_conditions=[{"key": "done", "value": True}],
    failure_conditions=[],
    event_schedule=[],
    viable_routes=[
        Route(id="r1", name="Route One", description="...",
              required_action_types=["execute"],
              preconditions={}, consequences={"done": True},
              closes_routes=[], milestones_unlocked=[], final_reward=1.0)
    ],
    milestones=[],
    horizon=20,
    difficulty=2,
    domain_metadata={"story": "A short story about the crisis."}
)
```

Then pass it to the environment:

```python
env = LifeStackEnv()
obs = env.reset(task=my_task)
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | Initial doc created |
