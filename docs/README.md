# LifeStack documentation

All project documentation lives under **`docs/`** (except the repository **[README.md](../README.md)** at the root, which is the HuggingFace Space and GitHub entry point).

**Repository:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)

---

## Quick links

| Doc | Use |
|-----|-----|
| [training_guide.md](training_guide.md) | Install, train, resume, push to Hub |
| [train_trl.md](train_trl.md) | `scripts/train_trl.py` CLI reference |
| [reward.md](reward.md) | All 10 GRPO reward functions and anti-hacking design |
| [model_card.md](model_card.md) | Hugging Face model card (post to Hub) |
| [blog.md](blog.md) | Engineering retrospective / HF blog draft |

---

## Core system

| Doc | Code | Description |
|-----|------|-------------|
| [lifestack_env.md](lifestack_env.md) | `core/lifestack_env.py` | `LifeStackEnv`, `WorldEngine`, `reset/step/rollout`, OpenEnv shim |
| [reward.md](reward.md) | `core/reward.py`, `scripts/train_trl.py` | `compute_reward`, `compute_task_reward`, 10 GRPO reward heads |
| [task.md](task.md) | `core/task.py` | `Task`, `Route`, `Milestone`, `ExoEvent` dataclasses |
| [memory.md](memory.md) | `agent/memory.py` | `LifeStackMemory`, ChromaDB collections, few-shot injection |
| [conflict_generator.md](conflict_generator.md) | `agent/conflict_generator.py` | Crisis scenario templates, `TaskGenerator`, 8 domains |

---

## Training & configuration

| Doc | Description |
|-----|-------------|
| [training_guide.md](training_guide.md) | End-to-end guide: Colab, server, resume, common errors |
| [train_trl.md](train_trl.md) | `LifeStackGRPOTrainer`, JSON-boundary masking, 5-stage curriculum |
| [configuration.md](configuration.md) | `GRPOConfig` via `_make_grpo_config`, reward weights per stage |
| [kaggle_train.md](kaggle_train.md) | Notebook-specific setup (T4, session timeouts, OOM tips) |

---

## Scripts & evaluation

| Doc | Description |
|-----|-------------|
| [eval.md](eval.md) | `scripts/eval.py` random-policy baseline |
| [scripts.md](scripts.md) | All scripts under `scripts/` with usage examples |

---

## Application & deployment

| Doc | Description |
|-----|-------------|
| [app.md](app.md) | Flask demo `app_flask.py` on port 7860, routes, `MODEL_REGISTRY` |
| [DEPLOYMENT.md](DEPLOYMENT.md) | `start.sh` dual-service, ports 7860 + 8000, env vars, Docker |

---

## Meta & pitch

| Doc | Description |
|-----|-------------|
| [implementation_summary.md](implementation_summary.md) | Engineering summary, training timeline, code map |
| [mentor_pitch.md](mentor_pitch.md) | 60–90 second pitch with honest numbers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Doc-first contribution rules, commit format |

---

## Maintenance rule

When you add a feature or a new doc file, add a one-line entry in the appropriate table above. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full doc-first rule.

*Last updated: 2026-04-26.*
