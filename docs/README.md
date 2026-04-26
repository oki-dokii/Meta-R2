# LifeStack documentation

All project documentation lives under **`docs/`** (except the repository **[README.md](../README.md)** at the root, which is the entry point for GitHub).

**Repository:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)

---

## Quick links

| Doc | Use |
|-----|-----|
| [training_guide.md](training_guide.md) | Install, train, resume, push to Hub |
| [train_trl.md](train_trl.md) | `scripts/train_trl.py` CLI reference |
| [reward.md](reward.md) | GRPO reward heads and JSON parsing |
| [model_card.md](model_card.md) | Hugging Face model card (copy to Hub if needed) |
| [blog.md](blog.md) | Narrative post / HF blog draft |

---

## Core system

| Doc | Code | Description |
|-----|------|-------------|
| [lifestack_env.md](lifestack_env.md) | `core/lifestack_env.py` | **LifeStackEnv** — OpenEnv, domains, dependency graph |
| [reward.md](reward.md) | `core/reward.py`, `scripts/train_trl.py` | Rewards, `frac_reward_zero_std`, parsing |
| [task.md](task.md) | `core/task.py` | Task, route, milestone schema |
| [memory.md](memory.md) | `agent/memory.py` | ChromaDB trajectories and feedback |
| [conflict_generator.md](conflict_generator.md) | `agent/conflict_generator.py` | Crisis templates and task generation |

---

## Training & configuration

| Doc | Description |
|-----|-------------|
| [training_guide.md](training_guide.md) | End-to-end training guide (Colab, server, logs, errors) |
| [train_trl.md](train_trl.md) | Flags, `LIFESTACK_NO_UNSLOTH`, episodic mode, resume |
| [configuration.md](configuration.md) | `GRPOConfig`, single-step vs episodic weights |
| [kaggle_train.md](kaggle_train.md) | Kaggle / notebook training notes |

---

## Scripts & evaluation

| Doc | Description |
|-----|-------------|
| [eval.md](eval.md) | `scripts/eval.py` random baseline |
| [scripts.md](scripts.md) | Other scripts under `scripts/` |

---

## Application & operations

| Doc | Description |
|-----|-------------|
| [app.md](app.md) | Gradio UI (`app.py`) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment notes |

---

## Meta & pitching

| Doc | Description |
|-----|-------------|
| [implementation_summary.md](implementation_summary.md) | Engineering summary and code map |
| [mentor_pitch.md](mentor_pitch.md) | Short mentor / demo pitch |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute and doc rules |

---

## Maintenance rule

When you add a feature or a new doc file, add a **one-line entry** in the appropriate table above.

*Last updated: 2026-04-26.*
