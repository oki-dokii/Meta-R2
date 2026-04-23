# reward.md — Reward System Reference

`core/reward.py` — Task-aware reward orchestrator.

---

## Overview

Two reward functions are available:

| Function | Used when |
|---|---|
| `compute_reward(...)` | Legacy / no-task episodes |
| `compute_task_reward(...)` | All task-driven episodes (v2.0+) |

---

## `compute_task_reward` — Components

```
reward = (0.10 × metric_delta)       # Local step improvement
       + (0.40 × milestone_reward)   # Reaching key progress markers
       + (0.30 × completion_reward)  # Final goal achievement
       + (0.10 × replan_bonus)       # Recovery after ExoEvents
       + (0.10 × efficiency)         # Resource preservation
       + penalties
```

### Penalties

| Penalty | Value | Trigger |
|---|---|---|
| `INACTION_PENALTY` | `-0.20` | `actions_taken == 0` |
| `CRITICAL_FLOOR_VIOLATION` | `-0.30` | Any metric drops below 20 |
| `DEAD_END_PENALTY` | `-0.50` | All viable routes closed, no success |
| `ROLLBACK_PENALTY` | `-0.10` | Agent used rollback action |
| `CASCADE_COLLAPSE` | `-0.30` | Metric drops from safe zone (>20) to critical (<10) |
| `WAIT_CAP_PENALTY` | `-0.50` | 4+ consecutive wait actions |
| `PLAUSIBILITY_VIOLATION` | `-0.20` | Zero-cost non-zero metric changes |

---

## Return Value

Both functions return `(reward: float, breakdown: dict)`.

```python
breakdown = {
    "components": {
        "metric_delta": float,
        "milestone": float,
        "completion": float,
        "replan": float,
        "efficiency": float,
        "format_compliance": float,
        "reasoning": float,
    },
    "penalties_fired": list[str],
    "total": float,
}
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | Initial doc created |
