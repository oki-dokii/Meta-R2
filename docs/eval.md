# eval.py — Evaluation Runner Reference

`scripts/eval.py` — Standalone LifeStack evaluation runner using a random-action baseline.

No model, no GPU, no API key required.

---

## Overview

Runs N independent episodes against `LifeStackEnv` using uniformly random actions as a
baseline policy. Prints a live per-episode table and aggregate statistics at the end.

Useful for:
- Verifying environment correctness after changes
- Establishing a random-baseline reward floor before training
- CI smoke checks (no external dependencies)

---

## Usage

```bash
# Default: 10 episodes, any domain
python scripts/eval.py

# 20 episodes, flight_crisis domain only
python scripts/eval.py --episodes 20 --domain flight_crisis

# Verbose per-step output
python scripts/eval.py --episodes 5 --verbose
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | `int` | `10` | Number of episodes to run |
| `--domain` | `str` | `None` | Optional domain filter passed to `TaskGenerator.generate()` |
| `--verbose` | flag | `False` | Print per-step action, reward, and done status |

Supported `--domain` values: `flight_crisis`, `code_merge_crisis` (or omit for random).

---

## Output

### Per-episode table

```
   EP   TOTAL REWARD   STEPS  DOMAIN                SUCCESS
  ────  ────────────  ──────  ────────────────────  ───────
     1        0.3120       8  flight_crisis               ✗
     2        1.8450      12  code_merge_crisis            ✓
```

### Aggregate stats

```
  ──────────────────────────────────────────────────────────
  Episodes     : 10
  Mean Reward  : 0.8231
  Success Rate : 30.0%
  Mean Steps   : 10.4
```

---

## Action Space (Random Baseline)

Each step samples uniformly from:
`execute`, `inspect`, `plan`, `wait`, `communicate`, `spend`, `delegate`

- `execute` actions target a real route ID from the active task.
- `inspect` actions target a real hidden-state key from the active task.
- Other actions apply a small random metric nudge and resource cost.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | File created — implements random baseline evaluation runner |
