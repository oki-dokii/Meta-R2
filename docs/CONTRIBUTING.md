# Contributing to LifeStack

This document defines the **documentation rule** for the project.  
**Nothing ships without its matching doc entry.**

---

## The Rule: Doc-First Development

Every change that adds, removes, or significantly modifies a feature must include
**all three** of the following before the commit is made:

| # | Action | Where |
|---|---|---|
| 1 | **Create or update a doc file** | `docs/<topic>.md` |
| 2 | **Update README.md** | File Structure table + relevant section |
| 3 | **Update `docs/INDEX.md`** | Add a one-line entry for the new doc |

> [!IMPORTANT]
> A pull request / commit that adds a new script, module, or feature **without**
> updating `docs/INDEX.md` and `README.md` is considered incomplete and should
> not be merged.

---

## What Counts as "a Feature"

| Change type | Doc required? |
|---|---|
| New Python module (`core/`, `agent/`, `intake/`) | ✅ Yes — `docs/<module>.md` |
| New script (`scripts/*.py`) | ✅ Yes — entry in `docs/scripts.md` |
| New Gradio tab in `app.py` | ✅ Yes — entry in `docs/app.md` |
| New CLI argument to an existing script | ✅ Yes — update relevant doc |
| Bug fix with no API surface change | ❌ No (but update changelog if breaking) |
| Refactor with no API surface change | ❌ No |
| New environment variable / secret | ✅ Yes — update `docs/configuration.md` |
| New dependency in `requirements.txt` | ✅ Yes — note in relevant doc + README |

---

## Doc File Conventions

- All docs live in `docs/`. No `.md` files at repo root except `README.md` and this file.
- File names are lowercase with underscores: `docs/lifestack_env.md`, `docs/eval.md`.
- Each doc starts with a `# Title` h1 and a one-line summary.
- Use `## Overview`, `## Usage`, `## API / Parameters`, `## Examples` sections.
- Code blocks must have a language tag (` ```python `, ` ```bash `).

---

## Checklist (copy into every PR / commit message)

```
Docs checklist:
[ ] docs/<topic>.md created or updated
[ ] docs/INDEX.md updated with new entry
[ ] README.md File Structure table updated
[ ] README.md Quickstart / relevant section updated (if CLI changed)
```

---

## Docs Folder Structure

```
docs/
├── INDEX.md              ← Master index of all docs (ALWAYS update this)
├── CONTRIBUTING.md       ← This file — the rule
├── lifestack_env.md      ← core/lifestack_env.py reference
├── reward.md             ← core/reward.py reference
├── task.md               ← core/task.py schema reference
├── memory.md             ← agent/memory.py reference
├── conflict_generator.md ← agent/conflict_generator.py reference
├── app.md                ← app.py Gradio interface reference
├── eval.md               ← scripts/eval.py reference
├── train_trl.md          ← scripts/train_trl.md reference
├── scripts.md            ← All other scripts reference
└── configuration.md      ← Env vars, secrets, openenv.yaml
```

---

## Commit Message Format

```
<type>: <short description>

- <file changed>: <what changed>
- docs/<doc>.md: <created|updated>
- docs/INDEX.md: <added entry for X>
- README.md: <updated section Y>

Docs checklist: ✅ all three updated
```

Types: `feat` | `fix` | `refactor` | `docs` | `test` | `chore`
