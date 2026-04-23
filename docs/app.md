# app.md — Gradio Interface Reference

`app.py` — Gradio multi-tab interactive interface for LifeStack.

---

## Overview

`app.py` is the entry point for the demo. It wires together all LifeStack modules into
a single Gradio `Blocks` application served on `http://127.0.0.1:7860`.

---

## Module-level Singletons

These are instantiated once at import time:

| Variable | Type | Purpose |
|---|---|---|
| `MEMORY` | `LifeStackMemory` | ChromaDB trajectory + feedback store |
| `AGENT` | `LifeStackAgent` | LLM-backed decision agent |
| `INTAKE` | `LifeIntake` | NL → structured conflict parser |
| `DEMO_CONFLICT` | `ConflictEvent` | Fixed "Friday 6PM" conflict for tab 1 |
| `DEMO_PREDICTOR` | `TrajectoryPredictor` | 7-day risk score tracker |
| `LONG_DEMO` | `LongitudinalDemo` | Arjun's multi-week journey |
| `GMAIL` | `GmailSignalExtractor` | Optional Gmail stress signal extractor |

---

## Tabs

| Tab | Label | Key Function |
|---|---|---|
| 1 | 🎯 Live Demo | `run_demo(person_label, conflict_label)` |
| 2 | 💭 Try Your Situation | `run_custom(situation, sliders..., gmail_signals)` |
| 3 | 📊 Training Results | `load_training_tab()` |
| 4 | 🗓️ Arjun's Journey | `LONG_DEMO.show_longitudinal_comparison()` |
| 5 | 🗺️ Task Explorer | `load_demo_task()` |
| 6 | 📬 Follow-up | `submit_outcome_feedback(...)` |

---

## Key Functions

### `submit_outcome_feedback(ep_id, score, domains_up, domains_down, notes, time_spent)`

Stores real-world outcome data into ChromaDB via `MEMORY.store_feedback(feedback)`.

> **Note:** Uses `MEMORY` (the module-level `LifeStackMemory` instance). The previously
> undefined `AGENT_MEMORY` reference was corrected to `MEMORY` on 2026-04-23.

### `run_demo(person_label, conflict_label)`

Generator — yields `(pred_html, before_html, narrative, decision_html)` tuples for each
animation frame. Runs cascade animation then agent intervention.

### `run_custom(situation, ...)`

Calls `INTAKE.full_intake()` to parse NL input, then `AGENT.get_action()`, steps the env,
returns `(life_html, after_html, plan_html)`.

---

## Running

```bash
python app.py
```

Starts on port `7860` with `share=False`. Edit `__main__` block to change port/theme.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | `AGENT_MEMORY` undefined crash fixed — replaced with `MEMORY` in `submit_outcome_feedback` |
