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
reward = (0.35 × milestone)          # Reaching key progress markers
       + (0.25 × completion)         # Final goal achievement (binary 1.0 if any goal met)
       + (0.15 × outcome)            # Isolated local metric improvement
       + (0.10 × replan_bonus)       # Recovery after ExoEvents
       + (0.10 × efficiency)         # Resource preservation relative to delta
       + (0.05 × reasoning)          # Logical coherence & action alignment
       + penalties
```

### Penalties

| Penalty | Value | Level | Trigger |
*   **INACTION_PENALTY** | `-0.40` | Step | `actions_taken == 0` |
*   **TASK_INACTION** | `-0.20` | Task | `actions_taken == 0` (additive to step penalty) |
*   **CRITICAL_FLOOR** | `-0.50` | Step | Any metric drops below 20 |
*   **DEAD_END** | `-0.50` | Task | All viable routes closed without success |
*   **CASCADE_SPREAD** | `-0.30` | Step | Changes spread wider than disruption baseline |
*   **REL_COLLAPSE** | `-0.15` | Step | Relationships drop >20 pts in one step |
*   **REL_EROSION** | `-0.15` | Episode | Cumulative relationship drop >20 pts |
*   **PLAUSIBILITY** | `-0.10 to -0.30` | Step | Implausible metric/cost ratio |
*   **TIMEOUT** | `-0.20` | Task | Max steps reached without resolution |

---

## Return Value

Both functions return `(reward: float, breakdown: dict)`.

```python
breakdown = {
    "components": {
        "outcome": float,
        "containment": float,
        "efficiency": float,
        "preservation": float,
        "format_compliance": float,
        "plausibility": float,
        "reasoning_alignment": float,
        "replan": float,
        "milestone": float,
        "completion": float
    },
    "penalties_fired": list[str],
    "base_reward": float,
    "total": float,
}
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | Initial doc created |
