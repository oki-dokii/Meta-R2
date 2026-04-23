# conflict_generator.md — Conflict Generator Reference

`agent/conflict_generator.py` — ConflictEvent templates and TaskGenerator.

---

## Overview

Two parallel systems for generating crises:

| System | Purpose |
|---|---|
| `ConflictEvent` + `TEMPLATES` | 15 handcrafted conflicts at difficulty 1–5 |
| `TaskGenerator` | Generates long-horizon `Task` objects (two domains) |

---

## `ConflictEvent` (Legacy)

```python
@dataclass
class ConflictEvent:
    id: str
    title: str
    story: str
    primary_disruption: dict   # Metric deltas applied on env reset
    decisions_required: list[str]
    resource_budget: dict      # {"time", "money", "energy"}
    difficulty: int            # 1–5
```

### Helper functions

```python
conflict = generate_conflict()              # random from all 15
conflict = generate_conflict(difficulty=3)  # difficulty-3 pool
escalated = escalate_conflict(conflict)     # 1.4× disruption, 0.7× budget
new, reason = adaptive_escalate(conflict, agent_history)  # auto-tune
```

---

## `TaskGenerator`

```python
generator = TaskGenerator()
task = generator.generate()
task = generator.generate(domain="flight_crisis", difficulty=4)
task = generator.generate(domain="code_merge_crisis")
```

### Supported Domains

| Domain | Goal |
|---|---|
| `flight_crisis` | Survive Airport Cancellation |
| `code_merge_crisis` | Resolve Production Outage |

Unknown domains fall back to `flight_crisis`.

---

## Adding a New Domain

1. Add `generate_<domain>(self, difficulty) -> Task` to `TaskGenerator`.
2. Add to the `if/elif` in `generate()`.
3. Update this file and `docs/INDEX.md` and `README.md`.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | Initial doc created |
