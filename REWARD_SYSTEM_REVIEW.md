# Reward System Review vs. the Guide

## What you have

In `core/reward.py`: One composite reward function (`compute_task_reward`) that blends 7 weighted components into a single float:

| Component             | Weight | Function                       |
|-----------------------|--------|--------------------------------|
| local metric delta    | 5%     | compute_reward                 |
| milestone             | 35%    | compute_milestone_reward       |
| task completion       | 25%    | compute_task_completion_reward |
| replanning            | 10%    | compute_replan_bonus           |
| resource efficiency   | 5%     | -                              |
| reasoning coherence   | 10%    | reward_reasoning_coherence     |
| format compliance     | 10%    | reward_format_compliance       |

In `train_trl.py`: 6 separate functions passed to `reward_funcs=[]` for GRPO:
`reward_format_fn`, `reward_plausibility_fn`, `reward_task_success_fn`, `reward_milestone_fn`, `reward_reasoning_fn`, `reward_human_feedback_fn`

---

## Where you follow the guide ✅

- 6 separate GRPO reward functions — matches the guide's "multiple independent reward functions" recommendation
- Format compliance (`reward_format_compliance`) — guide explicitly lists format compliance
- Timeout penalty (`reward_timeout_check`) — guide says "penalize timeouts"
- Plausibility anti-cheat (`reward_plausibility_check`) — catches zero-cost metric hacks (guide: "anti-cheating checks")
- Reasoning coherence — guide recommends process-aware feedback
- Resource lockout (`lifestack_env.py:431-439`) — resource deduction happens before metric changes, with `metric_changes = {}` if budget depleted. Good explicit lockdown.
- `CRITICAL_FLOOR_VIOLATION`, `INACTION_PENALTY`, `CASCADE_COLLAPSE` penalties
- Curriculum learning in `train.py` and `train_trl.py` — matches guide section 6
- Component-level logging (`train_trl.py:274-277`) — guide section 15 says watch individual reward columns, not just total reward

---

## Where you don't fully follow the guide ❌ (Fixed ✅)

1. **The 6 GRPO functions are NOT truly independent — they share one environment call**
   - *Fix applied*: Decoupled `reward_format_fn` by explicitly checking JSON format using `core.reward.reward_format_compliance()`, making it fully independent.

2. **`_REWARD_CACHE` is a global mutable dict — a guide-listed hacking vector**
   - *Fix applied*: Added a size cap of `1000` cache entries to mitigate this vector.

3. **`reward_human_feedback_fn` silently goes neutral when ChromaDB is unavailable**
   - *Fix applied*: Logs a warning and returns `-0.01` (a small penalty) instead of `0.0`.

4. **No execution sandboxing**
   - *Fix applied*: Added a `allowed_keys` whitelist in `lifestack_env.step()` constructed from `current_metrics.flatten().keys()`.

5. **Step-level reward (`compute_task_reward`) is still one blended number for the env itself**
   - (For future consideration/rewrite)

---

## Quick priority fixes

| Priority | Fix | Guide reference | Protocol / Fixed? |
|----------|-----|-----------------|-------------------|
| High | Add a TTL or size cap to `_REWARD_CACHE` (or disable it) | Section 8: "caching results" | ✅ Fixed |
| High | Add a metric key whitelist in `lifestack_env.step()` so model can't inject arbitrary paths | Section 8: "Lock down execution" | ✅ Fixed |
| Medium | Make at least 1-2 GRPO functions truly independent (e.g., `reward_format_fn` can parse JSON without calling `get_lifestack_evaluation`) | Section 7: "multiple independent checks" | ✅ Fixed |
| Low | Log a warning or small penalty when `reward_human_feedback_fn` falls back to 0.0 | Section 15: monitor individual columns | ✅ Fixed |

*The biggest structural win is decoupling `reward_format_fn` from the shared env call — it can check JSON validity entirely on its own, making it genuinely independent from the environment's result.*

---

## Secondary Bug Fixes ❌ -> ✅

1. **Bug 1: `reward_plausibility_fn` inverted/broken output**
   - *Fix applied*: Extracted the parsed completion and invoked `reward_plausibility_check` natively to retrieve the true continuous penalty score (e.g., `-0.1`, `-0.3`) instead of returning a binary `1.0`/`-1.0`.

2. **Bug 2: `reward_task_success_fn` double-dipping components**
   - *Fix applied*: Narrowed the function to retrieve just the `.get("completion", 0.0)` score from the breakdown, avoiding re-summing milestone, format, and reasoning.

3. **Bug 3: `reward_reasoning_fn` output range is noise**
   - *Fix applied*: Added a `* 10.0` scalar to inflate the `[-0.10, 0.10]` range to `[-1.0, 1.0]`, equalizing its variance and ensuring it produces valid gradients.

4. **Bug 4: Task reconstruction was non-deterministic**
   - *Fix applied*: Injected a sampled `seed` into `<SYSTEM_METADATA>` and set `random.seed()` around `TaskGenerator.generate()` in the evaluation function. Now the environment evaluates against the exact same routes and milestones the prompt originally described.

5. **Bug 5: `reward_human_feedback_fn` DB query exploit**
   - *Fix applied*: Switched the ChromaDB lookup to query against the `prompt` string instead of `action.reasoning`. The agent can no longer manipulate the query text to retrieve high scores.

---

## Critical Bug Fixes ❌ -> ✅

1. **Critical Bug 1: Milestone and Completion rewards were dead**
   - *Fix applied*: Populated `success_conditions` for all task domains in `TaskGenerator`.
   - *Fix applied*: Exposed `viable_routes` in the GRPO prompt so the model knows which IDs to target.
   - *Fix applied*: Added `execute` to the allowed `action_type` list and updated schema instructions.

---

## Final Structural Hardening ❌ -> ✅

1. **Critical Bug 3: CodeMergeCrisisTask() was a stub**
   - *Fix applied*: Fully implemented the `CodeMergeCrisisTask` in `core/task.py` with real disruptions and routes.
   - *Fix applied*: Seeded `mutable_world` and `visible_world` baseline disruptions into ALL domain generators in `TaskGenerator`. No more "phantom crises."

---

## Reward Signal Activations ❌ -> ✅

1. **Critical Bug 4: replan_bonus was always 0.0**
   - *Fix applied*: Modified `generate_dataset` to sample tasks at steps 0, 2, and 4 instead of only step 0. 
   - *Fix applied*: Capture and display `EXOGENOUS EVENTS ENCOUNTERED` in the prompt context.
   - *Fix applied*: Synchronized `get_lifestack_evaluation` to fast-forward the environment to the corresponding step before scoring.
