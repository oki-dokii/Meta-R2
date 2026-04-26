# Contributing to LifeStack

This document defines the **documentation rule** for the project.  
**Nothing ships without its matching doc entry.**

---

## The rule: doc-first development

Every change that adds, removes, or significantly modifies a feature must include
**all three** of the following before the commit is made:

| # | Action | Where |
|---|--------|--------|
| 1 | **Create or update a doc file** | `docs/<topic>.md` |
| 2 | **Update [README.md](../README.md)** | File structure table + relevant section |
| 3 | **Update [docs/README.md](README.md)** | Add a one-line entry in the right table |

> [!IMPORTANT]
> A pull request or commit that adds a new script, module, or feature **without**
> updating **docs/README.md** and the root **README.md** is considered incomplete and should
> not be merged.

---

## What counts as a feature

| Change type | Doc required? |
|-------------|---------------|
| New Python module (`core/`, `agent/`, `intake/`) | Yes — `docs/<module>.md` |
| New script (`scripts/*.py`) | Yes — entry in `docs/scripts.md` |
| New tab or API route in `app_flask.py` | Yes — entry in `docs/app.md` |
| New CLI argument to an existing script | Yes — update the relevant doc |
| Bug fix with no API surface change | No (note in changelog if breaking) |
| Refactor with no API surface change | No |
| New environment variable / secret | Yes — `docs/configuration.md` |
| New dependency in `requirements.txt` | Yes — note in relevant doc + README |

---

## Doc file conventions

- All documentation lives in **`docs/`**. The **only** markdown file at the repository root is **`README.md`** (GitHub project landing page).
- Prefer **lowercase with underscores**: `docs/lifestack_env.md`, `docs/kaggle_train.md`.
- Promotional or Hugging Face collateral also lives under `docs/` (e.g. `blog.md`, `model_card.md`).
- Each doc starts with a `# Title` heading and a one-line summary.
- Code blocks must have a language tag (` ```python `, ` ```bash `).

---

## Checklist (copy into PR or commit message)

```
Docs checklist:
[ ] docs/<topic>.md created or updated
[ ] docs/README.md updated with new entry
[ ] README.md file structure / quickstart updated (if needed)
```

---

## Docs folder layout (representative)

```
docs/
├── README.md              ← Master index (always update for new docs)
├── CONTRIBUTING.md        ← This file
├── training_guide.md
├── train_trl.md
├── configuration.md
├── reward.md
├── lifestack_env.md
├── task.md
├── memory.md
├── conflict_generator.md
├── eval.md
├── scripts.md
├── app.md
├── DEPLOYMENT.md
├── blog.md
├── model_card.md
├── mentor_pitch.md
├── implementation_summary.md
└── kaggle_train.md
```

---

## Commit message format

```
<type>: <short description>

- <file changed>: <what changed>
- docs/<doc>.md: <created|updated>
- docs/README.md: <added entry for X>
- README.md: <updated section Y>

Docs checklist: all items updated
```

Types: `feat` | `fix` | `refactor` | `docs` | `test` | `chore`
