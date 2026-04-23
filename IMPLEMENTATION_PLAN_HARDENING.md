# Hardening Audit & Implementation Plan

Now I have the full picture. Let me give you a complete, honest audit and plan.

---

## Part 1: How many reward functions do we have?

Technically 6 functions in `core/reward.py`, but the reality is worse than it looks:

| Function | What it does | Actually wired? |
| :--- | :--- | :--- |
| `compute_reward()` | 4 components + 4 penalties (local step) | ✅ called in `lifestack_env.py:step()` |
| `compute_milestone_reward()` | milestones hit / total possible | ❌ defined, never called in env |
| `compute_task_completion_reward()` | success conditions met | ❌ defined, never called in env |
| `compute_replan_bonus()` | recovery after exo-events | ❌ defined, never called in env |
| `compute_dead_end_penalty()` | no routes remaining | ❌ defined, never called in env |
| `compute_task_reward()` | orchestrator combining all above | ❌ defined, `env.step()` still calls only `compute_reward()` |

**So in practice: 1 reward function is active. 5 are dead code.**

---

## Part 2: Gap vs. hackathon guide

The guide explicitly says (§7, §8, §21):
> "Use multiple independent reward functions. If you only have one, it's easier to hack. Multiple independent checks reduce that risk."
> "Common mistake: using only one reward function"

### Full Gap Analysis:

| Guide Requirement | Our Status | Implementation Detail |
| :--- | :--- | :--- |
| **Execution success** (task completed?) | ❌ Missing | `compute_task_completion_reward` exists but unwired |
| **Correctness** (metrics actually improved?) | ✅ Active | `outcome_score` in `compute_reward` |
| **Format compliance** (valid JSON?) | ❌ Missing | Completely missing in previous version |
| **Timeouts** (step limit hit penalty?) | ❌ Missing | Missing |
| **Resource usage** | ✅ Active | `resource_efficiency_score` |
| **Safety constraints** (floor violations) | ✅ Active | `CRITICAL_FLOOR_VIOLATION` |
| **Anti-cheating checks** | ❌ Missing | Model can claim +50 metric change with 0 resource cost |
| **Process-aware feedback** (step-level) | ❌ Missing | Missing |
| **Multiple independent fns logged** | ❌ Missing | Only one fn running |

**Parameters currently used to compute reward (the one active fn):**
- `outcome_score`: delta across all 23 sub-metrics, domain-weighted 1/6 each
- `cascade_containment_score`: % of metrics that didn't worsen
- `resource_efficiency_score`: 1 - avg(time/20, money/500, energy/100)
- `relationship_preservation_score`: sigmoid on relationship domain average delta
- **Penalties:** CRITICAL_FLOOR (-0.50), CASCADE_SPREAD (-0.30), INACTION (-0.40), RELATIONSHIP_COLLAPSE (-0.15)

**Weights:** 0.40 outcome + 0.25 containment + 0.20 efficiency + 0.15 preservation

---

## Part 3: Delayed Human Outcome Signal

This is excellent and has a formal name: **delayed human outcome signal**. The idea:
> After the agent gives advice → user acts on it → after N hours/days when the effect resolves → user submits: "did it work? what else changed?"

This gives you two things the simulator can't:
1. **Ground truth** on whether advice was correct (human validates predicted changes).
2. **Unmeasured second-order effects** (e.g., trust damage not captured by metrics).

---

## The Plan

### Step 1 — Wire the orchestrator (1 day, critical)
`lifestack_env.py:step()` currently calls `compute_reward()`. Change it to call `compute_task_reward()` when a `Task` is present. This instantly activates milestone + completion + replan rewards without writing new code.

### Step 2 — Add the 3 missing independent reward functions (1 day)
* **reward_format_compliance**: +1.0 for valid JSON, -1.0 for refusals/text. Prevents the most common GRPO failure mode.
* **reward_plausibility_check**: Anti-gaming check. `ratio = sum(abs(metric_changes)) / max(1, sum(resource_costs))`. If ratio > 15, return -0.30.
* **reward_timeout_check**: Penalty if `step_count >= max_steps` and not done.

### Step 3 — Process-aware intermediate reward (1 day)
Add a reasoning coherence check — does the reasoning field actually mention the conflict domain? insegning the same final reward to every token is inefficient.

### Step 4 — Anti-hacking logging
Add "suspicious" flag to logs: `reward > 0.8 and resource_cost == {}`.

### Step 5 — Human outcome feedback loop (new feature, 2-3 days)
Build `core/feedback.py` and Gradio UI for users to submit `OutcomeFeedback`. Store in ChromaDB and wire into retraining loop via `compute_human_feedback_reward`.

---

## Priority Order
1. **Wire compute_task_reward into env.step()** → Immediate 4x more reward signal
2. **Add format_compliance reward fn** → Prevents #1 GRPO failure mode
3. **Add plausibility_check reward fn** → Blocks reward hacking
4. **Log each fn independently in breakdown** → Satisfies guide §15
5. **Build OutcomeFeedback dataclass + app UI** → Differentiator
6. **Wire human feedback into ChromaDB + retraining** → Long-term loop
