# LifeStack Hackathon Sprint — Implementation Plan

## Context

**Submission deadline:** 26 Apr 5 PM. Offline from 25 Apr 8 AM. ~30 hours of offline build time.

The LifeStack Flask demo (`app_flask.py` + `templates/index.html`) already ships 10 API endpoints, a 6-tab UI, and a working agent/memory/cascade/reward pipeline. This sprint adds **13 additive features** (demo panels, APIs, RLHF loop, multi-step training, real-data connectors, tests, blog) without breaking existing endpoints. All work is additive.

Budget: **$90 HF credits** — T4 Small for the always-on demo Space, A10G for GRPO training runs, HF Inference API for the NLP panel. Target trained checkpoint: **`jdsb06/lifestack-grpo-v2`** (user will push).

Key reusable primitives already in repo (do not rebuild):
- `core/cascade_utils.py:5 animate_cascade()` — returns list of 4 frames with `flat` + `status` dicts
- `agent/counterfactuals.py:10 generate_counterfactuals()` — returns list of alternatives
- `agent/memory.py:74 LifeStackMemory.store_trajectory()` and `:128 store_feedback(OutcomeFeedback)`
- `core/feedback.py OutcomeFeedback` + `compute_human_feedback_reward()`
- `core/life_state.py:61 LifeMetrics.flatten()` — 23 metric paths
- `agent/conflict_generator.py TEMPLATES` (13 scenarios) + `generate_conflict()`
- `core/metric_schema.py VALID_METRIC_PATHS`

Already wired in `app_flask.py`: `/api/feedback/submit` (Feature 9 backend is done — scope of F9 reduces to frontend panel + training integration); `/api/simulation/cascade` (kept intact, new `/api/cascade/frames` added alongside).

---

## Implementation Order (Offline Sprint)

1. F1 Trained-vs-Baseline comparison (impact demo)
2. F5 Domain risk heatmap (sidebar, always visible)
3. F3 "Try Your Own" NLP + HF Inference fallback
4. F2 D3 cascade visualisation
5. F4 Personality comparison with OCEAN radar
6. F6 Counterfactual explorer panel
7. F8 Multi-step GRPO training loop + `push_to_hub`
8. F9 RLHF feedback panel + training integration
9. F7 Cold-vs-warm memory ablation demo
10. F10 Health + calendar uploads
11. F11 BLOG.md (~700 words)
12. F12 Four tests
13. F13 Episode history/replay

Before starting, run smoke tests (`scripts/smoke_test.py`, `scripts/eval.py --episodes 5`, cascade/counterfactual imports). Fix before adding features.

---

## Cross-Cutting Changes

### `requirements.txt` — add
- `huggingface_hub` (for F3 InferenceClient and F8 push_to_hub)
- `icalendar` (F10 calendar upload)

### `intake/intake.py` — LLM fallback chain (F3 dependency)
Refactor `_call_llm()` (~line 44) to cascade: **HF Inference API (`HF_TOKEN`) → Groq (`GROQ_API_KEY`) → empty-string fallback** (existing behaviour). `LifeIntake.__init__` constructs both an `InferenceClient(model="Qwen/Qwen2.5-1.5B-Instruct", token=HF_TOKEN)` when `HF_TOKEN` is present and the existing Groq `OpenAI` client when `GROQ_API_KEY` is present. `extract_conflict()` already returns an empty `ConflictEvent` when the LLM returns empty — keyword fallback below strengthens that path.

**Keyword fallback:** add `_match_template_by_keywords(text: str) -> ConflictEvent | None` that scans `TEMPLATES` for overlap with user text and returns the best match. Called inside `extract_conflict()` when both LLM clients fail.

### `app_flask.py` — shared helpers (used by F1, F4, F5, F7)
- `_run_episode(person, conflict, steps, seed, agent_fn) -> list[step_dict]`: initialises a fresh `LifeStackEnv`, applies the conflict disruption, loops `steps` iterations calling `agent_fn(metrics, budget, conflict, person)` to pick an action, runs `env.step()`, and collects `{step, action_type, target, reward, metrics, cost}`. `agent_fn` is injected so F1 can pass a random-action picker and a `LifeStackAgent.get_action`-wrapped version.
- `_random_action(metrics, budget, conflict, person) -> AgentAction`: samples uniformly from `core.action_space.EXAMPLE_ACTIONS` (line 98–196) and jitters `metric_changes` slightly so the baseline isn't deterministic. Same return shape as `AGENT.get_action()`.
- `compute_domain_health(flat_metrics: dict) -> dict[str, float]`: averages sub-metrics per domain, inverts `INVERTED_METRICS` (line 67, already defined), returns `{career, finances, relationships, physical_health, mental_wellbeing, time}` each in [0,1].

### `templates/index.html` — UI integration pattern
Every new feature adds one new tab button in the nav bar (line 37–44) and one content `<div id="content-X">` in the main section (line 46–202). Reuse existing classes: `.glass`, `.tab-active`, `.metric-bar`, Tailwind (`.rounded-2xl`, `.p-6`, `.space-y-6`, `.grid grid-cols-2 gap-6`, `.text-slate-400`, `.bg-indigo-500/10`). Chart.js is already loaded via CDN (line 8); D3 v7 to be added.

---

## Feature-by-Feature

### F1 — Trained vs Baseline Comparison
**Backend — `app_flask.py`:**
- `POST /api/comparison/run` → body `{conflict, person, steps=5, seed=42}`.
  - Resolve `conflict` via `CONFLICT_CHOICES`, `person` via `PERSONS`.
  - Call `_run_episode(..., agent_fn=_random_action)` → `baseline`.
  - Call `_run_episode(..., agent_fn=lambda m,b,c,p: AGENT.get_action(m,b,c,p))` with identical seed → `trained`.
  - Compute `reward_delta = sum(trained_rewards) - sum(baseline_rewards)`.
  - Return `{baseline: [...], trained: [...], reward_delta}`.

**Frontend:**
- New tab "Comparison". Two side-by-side `.glass` cards titled "Baseline (Random)" and "GRPO-Trained". For each step, render action-type badge + reward bar. Delta banner at the bottom (`bg-indigo-500/10`) showing `+X.XX`.

### F2 — Live Cascade Visualisation (D3)
**Backend:**
- `POST /api/cascade/frames` → body `{primary_disruption: {metric_path: delta}}`. Calls `animate_cascade(primary_disruption, LifeMetrics())` and returns `{frames}`. Keeps existing `/api/simulation/cascade` untouched.

**Frontend:**
- Add D3 v7 CDN line in `<head>`.
- New section inside the "Situational Portal" tab (below the existing cascade timeline at line ~70): `<svg id="cascade-graph" width="720" height="420">`.
- JS module `renderCascade(frames)`: creates 23 nodes from `VALID_METRIC_PATHS`, clusters by domain (6 cluster centres at: career TL, finances TR, relationships ML, physical_health MR, mental_wellbeing BC, time TC), draws edges from a hardcoded copy of the 20+ edges in `DependencyGraph.edges`. Iterates frames with 600ms `setTimeout`, recolouring nodes based on `frames[i].status[metric]`: `unchanged→#334155`, `primary→#ef4444`, `first→#f97316`, `second→#facc15`.
- Called from the existing simulation-action flow after each `/api/simulation/action` response.

### F3 — "Try Your Own Situation" NLP Panel
**Backend:**
- `/api/custom/run` already exists (line 162) and is fully wired. No route changes.
- `intake/intake.py` cross-cutting change above adds HF→Groq→keyword fallback.

**Frontend:**
- Existing "Try Your Case" tab (`#tab-custom`) is currently slider-heavy. Add a prominent textarea + Submit above the sliders. On submit, `fetch('/api/custom/run', {situation: text})` → render a card with detected domain(s), recommended action type/target, metric deltas as coloured badges (green for positive on positive-sense metrics, red otherwise, using `INVERTED_METRICS` set), reward bar.

### F4 — Personality Comparison
**Backend:**
- `POST /api/personality/compare` → body `{conflict_id="d5_friday", person_a, person_b, steps=3}`.
  - Look up persons from `PERSONS`. Run `_run_episode` twice with the trained agent on the same conflict + seed.
  - Return `{person_a: {name, actions, total_reward, ocean: {O,C,E,A,N}}, person_b: {...}, dominant_trait: "neuroticism"}` where `dominant_trait = argmax(|ocean_a[t] - ocean_b[t]|)`.

**Frontend:**
- New tab "Personality". Two `.glass` columns. Each has a Chart.js radar chart (already CDN-loaded) with 5 axes (OCEAN). Below the radar: action sequence + total reward. Banner highlighting the dominant trait.

### F5 — Domain Risk Heatmap
**Backend:** `compute_domain_health()` helper added (cross-cutting section). Every response from `/api/simulation/start`, `/api/simulation/action`, `/api/custom/run` gets an extra `domain_health` field derived from the metrics already in the payload — no new route.

**Frontend:** Persistent top bar above tab nav (inserted at ~line 35): 6 cells (2×3 grid on small, 6×1 on large). Each cell shows the domain emoji from `DOMAIN_EMOJI` and a pill background coloured via `hsl((1 - h) * 120, 70%, 45%)`. Re-rendered from every simulation response.

### F6 — Counterfactual Explorer
**Backend:**
- `POST /api/counterfactuals/generate` → body `{conflict, person, chosen_action: {...}}`. Reconstructs state, calls `generate_counterfactuals(AGENT, metrics, budget, conflict, person, chosen_action)`, returns `{chosen: {...}, alternatives: [3 items from the list]}`. (Counterfactuals already appear inside `/api/simulation/action` response — this route is the on-demand variant Feature 6 wants.)

**Frontend:** "What If?" collapsible panel appended below each step output. 3 alternative cards sorted by predicted reward. Chosen action outlined in indigo, best alt in green, worst in red.

### F7 — Memory Ablation (Cold vs Warm)
**Backend:**
- `POST /api/memory/ablation` → body `{conflict, person, steps=5}`.
  - Episode 1: pass `memory=None` (or a fresh `LifeStackAgent()` with empty `.memory`). Record actions + rewards.
  - `MEMORY.store_trajectory(conflict_title=..., route_taken=..., total_reward=..., reasoning=...)` for episode 1.
  - Episode 2: reuse `AGENT` (global — has ChromaDB via `MEMORY`). Query `MEMORY` for similar trajectories (existing retrieval method) and pass the top-k summary into `get_action`'s `few_shot_context` param.
  - Return `{cold: {actions, reward}, warm: {actions, reward, retrieved_context}, improvement_pct}`.

**Frontend:** Two-column timeline in a new "Memory" tab. Callout box with `💡 Agent recalled: …` when warm has retrieved context. Big percentage banner at the bottom.

### F8 — Multi-Step GRPO Training
**`scripts/train_trl.py` (currently 914 lines, single-prompt per scenario):**
- Add `run_full_episode(task, person, model, tokenizer, max_steps=10) -> tuple[list[step_reward], dict]`:
  - For each step: build prompt from current `LifeMetrics` + `ResourceBudget` + conflict, call `model.generate`, parse JSON action, call `env.step()`, append step reward from existing `compute_task_reward()`.
  - Return per-step rewards and a serialised trajectory.
- New CLI flag `--full-episode`. When set, `generate_dataset()` is replaced by `generate_episodic_dataset()` which calls `run_full_episode` per scenario and uses `sum(step_rewards) / max_steps` as the GRPO reward.
- `--dry-run` compatibility: 1 episode × 2 steps with a mock model (existing dry-run path stays valid).
- After `trainer.save_model()` at line 610, add `if not args.dry_run and args.push_to_hub: model.push_to_hub("jdsb06/lifestack-grpo-v2"); tokenizer.push_to_hub("jdsb06/lifestack-grpo-v2")`. New `--push-to-hub` flag guards it.
- Run on HF A10G once built: `python scripts/train_trl.py --full-episode --stages 5 --push-to-hub` (~$5).

### F9 — RLHF Loop
- **Backend:** `/api/feedback/submit` already fully implemented (line 267). No route changes needed.
- **Frontend:** Post-episode feedback panel (rendered after every completed simulation/custom/comparison episode). Slider 0–10, domain checkboxes (6 domains × improved/worsened), textarea. Submit posts `{episode_id, score, improved[], worsened[], notes, time}` to existing endpoint.
- **Training integration (`scripts/train_trl.py`):** New `--with-human-feedback` flag. When set, a new reward component `reward_human_feedback_fn` (hook already exists around line 379) loads stored feedback via `MEMORY.feedback_collection.query()` keyed by episode_id and blends `compute_human_feedback_reward()` output at weight 0.10, rebalancing existing weights proportionally.

### F10 — Real Data Integrations
**Backend:**
- `POST /api/data/health/upload` (multipart): accepts `.json` (Google Fit) or `.xml` (Apple Health). Parse `steps`, `heart_rate_resting`, `sleep_hours` (approximate parse; tolerate missing fields). Map to `physical_health.fitness`, `physical_health.energy`, `physical_health.sleep_quality`. Store in new module-level dict `USER_HEALTH_OVERRIDES`. Return `{parsed_metrics, events_found}`.
- `POST /api/data/calendar/upload` (multipart): `.ics` via `icalendar.Calendar.from_ical()`. Count events in next 7 days → `time.free_hours_per_week` (inverse), `career.workload`. Keyword match ("gym", "run", "yoga") → bump `physical_health.fitness`. Return same shape.
- `/api/simulation/start` and `/api/custom/run` consult `USER_HEALTH_OVERRIDES` when initialising `LifeMetrics()`.

**Frontend:** New "Connect My Data" subsection at the top of "Try Your Case". Two file inputs. After upload, render a chip list with `📊 From your real data — physical_health.fitness: 78`.

### F11 — BLOG.md (~700 words)
Rewrite the 13-line BLOG.md with 5 sections: Problem, What We Built, Key Results (+125%, +155%, +116% — already in README lines 45–71), What We Learned, What's Next. Inline-cite the 4 papers from README lines 233–241 (Starcke & Brand 2012; Roijers et al. 2013; Mullainathan & Shafir 2013; Wang et al. 2024).

### F12 — Four Tests (tests/)
- `test_env_reset.py`: `LifeStackEnv().reset()` → budget is fresh; reset twice → metrics identical. ~20 lines, pytest.
- `test_cascade.py`: `animate_cascade({"mental_wellbeing.stress_level": 30}, LifeMetrics())` returns 4 frames; frame 0 status all `unchanged`; frame 1 has at least one `primary`.
- `test_task_generator.py` (scoped per user answer): asserts `generate_conflict()` returns a valid `ConflictEvent` for each of the 6 life domains and `TEMPLATES` covers difficulties 1–5.
- `test_reward.py`: `compute_reward()` result in `[-1, 1]`; plausibility component penalises a 0-cost, 50-delta action.

### F13 — Episode History
**Backend:** 
- Maintain ring buffer `EPISODE_HISTORY: deque[dict] = deque(maxlen=5)` module-level in `app_flask.py`. After every episode-producing route, append `{id, conflict, steps[], final_reward, timestamp}`.
- `GET /api/history/list` returns summaries. `GET /api/history/replay/<episode_id>` returns full step log.

**Frontend:** New "History" tab, accordion list, click-to-expand per episode.

---

## Critical Files to Modify

| File | Features touching it |
|------|------|
| `app_flask.py` | F1, F2, F4, F5, F6, F7, F10, F13 (7 new routes, 3 helpers, 1 deque) |
| `intake/intake.py` | F3 (LLM fallback chain, keyword match) |
| `templates/index.html` | F1, F2, F3, F4, F5, F6, F7, F9, F10, F13 (new tabs, heatmap bar, D3 SVG, feedback panel) |
| `scripts/train_trl.py` | F8 (`run_full_episode`, `--full-episode`, `--push-to-hub`), F9 (`--with-human-feedback`) |
| `requirements.txt` | `huggingface_hub`, `icalendar` |
| `BLOG.md` | F11 (full rewrite) |
| `tests/test_env_reset.py`, `test_cascade.py`, `test_task_generator.py`, `test_reward.py` | F12 (new files) |

No other files get edited. No existing route or dataclass is modified.

---

## Verification

**Local (no GPU):**
```bash
python scripts/smoke_test.py
python scripts/eval.py --episodes 5
python -m pytest tests/ -v
python scripts/train_trl.py --full-episode --dry-run   # F8 dry-run
python app_flask.py  # open localhost:7860, click through each new tab
```

**HF Inference API check (F3):**
```python
from huggingface_hub import InferenceClient; import os
c = InferenceClient(model="Qwen/Qwen2.5-1.5B-Instruct", token=os.getenv("HF_TOKEN"))
print(c.chat_completion([{"role":"user","content":"Reply OK"}], max_tokens=5).choices[0].message.content)
```

**HF Space (T4, $0.60/hr, leave running 25 Apr 8 AM → 26 Apr 5 PM ≈ $20):**
1. Space settings → hardware: T4 Small.
2. Secrets: `HF_TOKEN`, `GROQ_API_KEY`.
3. Push branch → confirm Flask app starts on port 7860 → open every tab.

**A10G training run (F8, ~$5, one-off):** 
```bash
python scripts/train_trl.py --full-episode --stages 5 --push-to-hub
```
Afterwards: `https://huggingface.co/jdsb06/lifestack-grpo-v2` should show the checkpoint.

**End-to-end demo walkthrough to rehearse before 26 Apr 5 PM:**
1. Open Situational Portal → run Friday 6PM conflict → cascade SVG animates, heatmap shifts red.
2. Switch to Comparison tab → same conflict → watch delta bar fill positive.
3. Personality tab → Alex vs Chloe → radars + different rewards.
4. Try Your Case → paste "I just got fired and rent is due tomorrow" → plan card renders.
5. Memory tab → cold vs warm ablation → +116% banner.
6. Submit a feedback slider → stats endpoint reflects new feedback count.
