# scripts.md — Other Scripts Reference

Reference for scripts not covered by dedicated doc files.

---

## `scripts/run_episode.py`

Runs a single full episode with the LLM agent (requires API key).

```bash
python scripts/run_episode.py
python scripts/run_episode.py --difficulty 3 --verbose
```

Returns a result dict with `total_reward`, `steps`, `domain`.

---

## `scripts/train.py`

Legacy training loop (pre-TRL). Uses a simple policy gradient loop without curriculum.
Prefer `train_trl.py` for new training runs.

---

## `scripts/smoke_test.py`

Quick sanity check — imports all core modules, resets the env once, takes one step.
No agent required. Exits with code 0 on success.

```bash
python scripts/smoke_test.py
```

---

## `scripts/test_lifestack.py`

Full edge-case test suite (11 tests). Does not use pytest runner by default —
run directly or via `pytest scripts/test_lifestack.py`.

```bash
python scripts/test_lifestack.py
pytest scripts/test_lifestack.py -v
```

Tests requiring `OPENAI_API_KEY` are automatically skipped when the key is absent.

### Tests

| # | Name | What it checks |
|---|---|---|
| 1 | Cascade floor | Metrics never go below 0 |
| 2 | Cascade ceiling | Metrics never exceed 100 |
| 3 | Resource exhaustion | `deduct()` returns False without going negative |
| 4 | Inaction penalty | `INACTION_PENALTY` fires when `actions_taken=0` |
| 5 | Critical floor penalty | `CRITICAL_FLOOR_VIOLATION` fires below threshold |
| 6 | Cascade dampening | Second-order deltas < first-order delta |
| 7 | SimPerson uptake bounds | All uptake values in [0.1, 1.0] |
| 8 | Memory threshold | Only reward >= 2.0 stored |
| 9 | Episode termination | `done=True` after horizon steps |
| 10 | Task-driven smoke | Inspect + Route execute without crash |
| 11 | Full episode smoke | `run_episode()` returns float reward *(skipped without API key)* |

---

## `scripts/longitudinal_demo.py`

Seeds Arjun's multi-week journey into ChromaDB and renders a comparison view.
Used by Tab 4 (Arjun's Journey) in `app.py`.

---

## `scripts/validate_simperson.py`

Validates all `SimPerson` personality trait combinations produce valid uptake values.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | `test_lifestack.py` — `steps<=5` assertion fixed to `steps<=30`; `import pytest` added; `@pytest.mark.skipif` added to test 11 |
