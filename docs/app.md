# Flask Demo Interface

**Source files:** `app_flask.py`, `templates/index.html`

---

## Overview

`app_flask.py` is the demo application. It runs a Flask server on port 7860 and serves a single-page application with 10 tabs. The frontend uses Chart.js for reward/metric plots and vis-network for the cascade dependency graph visualization. There is no Gradio in the stack — `app.py` is a legacy file; `app_flask.py` is the production demo.

---

## Architecture

Flask handles all HTTP routes. The frontend is a full HTML/JS application in `templates/index.html`. Tabs communicate with the backend via `/api/*` endpoints that return JSON. The vis-network cascade animation calls `core/cascade_utils.py:animate_cascade()` to get frame-by-frame cascade data.

---

## Module-level singletons

These are instantiated at startup:

| Variable | Type | Purpose |
|----------|------|---------|
| `agent` | `LifeStackAgent` | GRPO model + Groq fallback |
| `memory` | `LifeStackMemory` | ChromaDB trajectory + feedback store |
| `MODEL_REGISTRY` | dict | Maps model label → HF repo ID for v1–v4 |
| `_load_grpo_model` | function | Lazy-loads a specific LoRA adapter |

`MODEL_REGISTRY` allows the demo to switch between v1, v3, and v4 adapters on the fly via a dropdown in the UI.

---

## Key API routes

| Route | Purpose |
|-------|---------|
| `GET /` | Serves `templates/index.html` |
| `POST /api/run_demo` | Runs the agent on a preset conflict, returns step-by-step cascade frames and action JSON |
| `POST /api/custom_run` | Runs the agent on a user-typed situation |
| `POST /api/submit_feedback` | Stores `OutcomeFeedback` in ChromaDB memory |
| `GET /api/cascade_animate` | Returns frame list from `animate_cascade()` for the vis-network graph |
| `GET /api/memory_stats` | Returns `LifeStackMemory.get_stats()` |
| `GET /api/health` | Basic health check |

---

## Running

```bash
python app_flask.py      # serves on http://localhost:7860
```

In Docker, `start.sh` runs this as the foreground process. HuggingFace Spaces health-checks port 7860 — Flask must stay alive for the Space to remain healthy.

---

## Related files

- `templates/index.html` — full frontend (Chart.js, vis-network)
- `core/cascade_utils.py` — `animate_cascade()` drives the dependency graph visualization
- `agent/agent.py` — `LifeStackAgent` with GRPO model
- `agent/memory.py` — `LifeStackMemory`
- `start.sh` — starts this as foreground service on port 7860
