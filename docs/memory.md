# memory.md — LifeStackMemory Reference

`agent/memory.py` — ChromaDB-backed trajectory and human-feedback storage.

---

## Overview

`LifeStackMemory` persists two types of data:

| Collection | What's stored |
|---|---|
| `collection` (trajectories) | Successful episode decisions — action type, reward, reasoning |
| `feedback_collection` | Real-world outcome feedback submitted via the Follow-up tab |

Only trajectories with `total_reward >= 2.0` are stored (threshold prevents noise).

---

## API

### Instantiation

```python
from agent.memory import LifeStackMemory

memory = LifeStackMemory(silent=True)           # default path
memory = LifeStackMemory(silent=True, path="./my_memory")  # custom path
```

The module-level singleton in `app.py` is named `MEMORY`:

```python
MEMORY = LifeStackMemory(silent=True)
```

### `store_trajectory(...)`

```python
memory.store_trajectory(
    conflict_title="Friday 6PM",
    route_taken="communicate",
    total_reward=2.5,
    metrics_diff_str="career.workload: -15.0",
    reasoning="Delegating resolved workload spike",
)
```

Silently skips storage if `total_reward < 2.0`.

### `store_feedback(feedback: OutcomeFeedback)`

```python
from core.feedback import OutcomeFeedback

feedback = OutcomeFeedback(
    episode_id="A1B2C3D4",
    overall_effectiveness=8,
    domains_improved=["career", "mental_wellbeing"],
    domains_worsened=[],
    unexpected_effects="Felt more confident",
    resolution_time_hours=2.0,
)
memory.store_feedback(feedback)
```

Used by the **Follow-up** tab in `app.py`.

### `get_stats() -> dict`

```python
stats = memory.get_stats()
# {
#   "total_memories": 42,
#   "average_reward": 2.71,
#   "by_action_type": {"communicate": 18, "delegate": 12, ...}
# }
```

### `query(conflict_description, n_results=3) -> list[dict]`

Retrieves the most semantically similar past decisions for a given situation description.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | `AGENT_MEMORY` reference in `app.py` corrected to `MEMORY` (the actual singleton) |
