# LifeStack — Documentation Index

> **Rule:** Every new feature, script, or module must add a one-line entry here.  
> See [CONTRIBUTING.md](CONTRIBUTING.md) for the full documentation rule.

---

## Core Modules

| Doc | Module | Description |
|---|---|---|
| [lifestack_env.md](lifestack_env.md) | `core/lifestack_env.py` | Main OpenEnv environment — step, reset, observation, WorldEngine, PartialObsFilter |
| [reward.md](reward.md) | `core/reward.py` | Task-aware reward orchestrator with milestone, cascade, and efficiency components |
| [task.md](task.md) | `core/task.py` | Task / Route / Milestone / ExoEvent dataclass schema |
| [memory.md](memory.md) | `agent/memory.py` | ChromaDB-backed trajectory + feedback storage |
| [conflict_generator.md](conflict_generator.md) | `agent/conflict_generator.py` | ConflictEvent templates and TaskGenerator |

## Application

| Doc | File | Description |
|---|---|---|
| [app.md](app.md) | `app.py` | Gradio multi-tab interface — tabs, callbacks, module-level singletons |

## Scripts

| Doc | Script | Description |
|---|---|---|
| [eval.md](eval.md) | `scripts/eval.py` | Standalone random-baseline evaluation runner |
| [train_trl.md](train_trl.md) | `scripts/train_trl.py` | GRPO curriculum training via HuggingFace TRL + Unsloth |
| [scripts.md](scripts.md) | `scripts/` (others) | run_episode, smoke_test, test_lifestack, longitudinal_demo |

## Configuration & Operations

| Doc | File | Description |
|---|---|---|
| [configuration.md](configuration.md) | `.env`, `openenv.yaml` | Environment variables, secrets, server config |

---

*Last updated: 2026-04-23 — add a row here whenever a new doc is created.*
