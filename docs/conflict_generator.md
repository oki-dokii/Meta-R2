# `agent/conflict_generator.py` — conflicts and tasks

Generates **crisis scenarios** and long-horizon **`Task`** objects for **LifeStackEnv**.

---

## Two layers

| Layer | Purpose |
|-------|---------|
| **`ConflictEvent` + `TEMPLATES`** | Hand-crafted narrative conflicts with difficulty **1–5**, disruption dicts, and resource hints |
| **`TaskGenerator`** | Builds structured **`Task`** graphs (routes, milestones, events) — used heavily in **GRPO** prompt generation |

Domains align with the environment: **career, finances, relationships, physical_health, mental_wellbeing, time, transport_crisis / flight_crisis, code_merge_crisis** (see `openenv.yaml` and `core/reward.py:VALID_DOMAINS`).

---

## Conflict helpers

```python
conflict = generate_conflict()
conflict = generate_conflict(difficulty=3)
escalated = escalate_conflict(conflict)
new, reason = adaptive_escalate(conflict, agent_history)
```

---

## Task generation

```python
from agent.conflict_generator import TaskGenerator

gen = TaskGenerator()
task = gen.generate(domain="flight_crisis", difficulty=2)
```

Training scripts embed **metadata** (seed, disruption, route ids) into prompts so reward functions can reconstruct the same world state.

---

## Training connection

- **Single-step** GRPO: one action JSON per prompt.  
- **Episodic** GRPO: prompts request **`{"actions": [...]}`** with **horizon** steps; `TaskGenerator` / episode dataset builders set difficulty for **curriculum** stages.

---

## See also

- [task.md](task.md)  
- [lifestack_env.md](lifestack_env.md)  
- [train_trl.md](train_trl.md)  
