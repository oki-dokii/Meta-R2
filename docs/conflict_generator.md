# Conflict Generator

**Source file:** `agent/conflict_generator.py`

---

## Overview

`agent/conflict_generator.py` provides two layers of scenario generation: legacy `ConflictEvent` templates (hand-authored narratives for the demo), and the full `TaskGenerator` (structured `Task` objects for all 8 domains used in GRPO training).

---

## Two layers

### Layer 1: `ConflictEvent` + `TEMPLATES`

Hand-crafted narrative conflicts with difficulty 1–5, primary metric disruptions, and resource hints. Used in the Flask demo and by `generate_dataset()` in `train_trl.py` to overlay disruptions onto generated tasks.

```python
conflict = generate_conflict()            # random conflict, any difficulty
conflict = generate_conflict(difficulty=3) # specific difficulty
escalated = escalate_conflict(conflict)    # increase difficulty by 1
new, reason = adaptive_escalate(conflict, agent_history)  # smart escalation
```

`adaptive_escalate()` examines the agent's history to decide whether to escalate the conflict at step 2 of an episode (called from `LifeStackEnv.step()`).

### Layer 2: `TaskGenerator` (all 8 domains)

Builds structured `Task` objects with routes, milestones, events, success/failure conditions — everything `LifeStackEnv` needs for a full episode.

```python
from agent.conflict_generator import TaskGenerator

gen = TaskGenerator()
task = gen.generate(domain="flight_crisis", difficulty=3)
task = gen.generate(domain="code_merge_crisis", difficulty=5)
task = gen.generate(domain="career", difficulty=2)
```

Supported domains: `career`, `finances`, `relationships`, `physical_health`, `mental_wellbeing`, `time`, `transport_crisis`, `code_merge_crisis`.

`transport_crisis` randomly dispatches to one of five transport modes: flight, train, car, rideshare, or transit-strike.

---

## Training connection

`train_trl.py` uses this `TaskGenerator` (not the minimal one in `core/task.py`) inside `get_lifestack_evaluation()` to reconstruct the task from the prompt's `<SYSTEM_METADATA>` block. The domain and difficulty are stored in the metadata, and `gen.generate(domain, difficulty)` must return a task with the same structure the model saw during prompt generation. This is why both `generate_dataset()` (which builds prompts) and `get_lifestack_evaluation()` (which evaluates completions) call the same `TaskGenerator`.

For both single-step and episodic training, the dataset generation embeds the task seed in `<SYSTEM_METADATA>`. `random.seed(eval_seed)` is set before `gen.generate()` in `get_lifestack_evaluation()` to ensure deterministic task construction for the same prompt.

---

## Related files

- `core/task.py` — `Task`, `Route`, `Milestone`, `ExoEvent` dataclasses
- `scripts/train_trl.py` — `generate_dataset()`, `generate_episodic_dataset()`, `get_lifestack_evaluation()`
- `core/lifestack_env.py` — `adaptive_escalate()` is called from `step()` at step 2
