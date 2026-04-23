# Implementation Plan: LifeStack Production Hardening

This document outlines the strategic engineering plan to transition the LifeStack simulation engine from a functional prototype to a production-hardened RL training platform.

## 1. Executive Summary: The "Dead Code" Problem
An audit of `core/reward.py` revealed that while we have several robust reward components (milestones, task completion, replan bonuses), the environment's `step()` logic was historically coupled only to `compute_reward()` (local deltas). This created a significant gap where the agent was not being incentivized for long-term task resolution or milestone achievement.

## 2. Gap Analysis vs. Hackathon Production Standards

| Guide Requirement | Status | Implementation Detail |
| :--- | :--- | :--- |
| **Execution Success** | ✅ Wired | `compute_task_reward` now combines completion + milestone signals. |
| **Correctness** | ✅ Wired | `outcome_score` validates metric deltas at each step. |
| **Format Compliance** | ✅ Wired | `reward_format_compliance` penalizes non-JSON or malformed outputs. |
| **Timeouts** | ✅ Wired | `reward_timeout_check` penalizes episodes hitting the horizon. |
| **Anti-Gaming** | ✅ Wired | `reward_plausibility_check` blocks high-magnitude changes for 0 cost. |
| **Process Feedback** | ✅ Wired | `reward_reasoning_coherence` rewards mentions of the conflict domain. |
| **Granular Logging** | ✅ Wired | `GRPOTrainer` now logs 6+ independent reward columns to TensorBoard. |

## 3. High-Priority Hardening Steps

### Step 1: Orchestrator Activation (Critical)
**Action:** Update `lifestack_env.py:step()` to call `compute_task_reward()` when a `Task` is present.
**Rationale:** Instantly activates milestone, completion, and replan signals, providing a 4x richer reward landscape for GRPO training.

### Step 2: Independent Reward Guardrails
To prevent the model from "hacking" a single scalar, we implement independent checks:
- **Format Compliance:** +1.0 for valid JSON, -1.0 for refusals/text. Prevents the most common GRPO failure mode.
- **Plausibility Check:** Penalizes suspicious `metric_change / resource_cost` ratios (>15). Prevents the model from claiming "free" improvements.
- **Timeout Penalty:** Small negative signal for reaching the step limit, encouraging efficiency.

### Step 3: Process-Aware Reasoning (DeepSeek-style PRO)
The guide emphasizes that "naively assigning the same final reward to every token is inefficient."
- **Coherence Check:** Reward the model for explaining its reasoning in a way that actually references the conflict domain (e.g., mention "flight" or "code merge").

## 4. Competitive Differentiator: Human Outcome Feedback (HOF)

The "Delayed Human Outcome Signal" closes the loop between simulator prediction and real-world reality.

### The Feedback Cycle
1. **Simulation:** Agent gives advice; `episode_id` and predicted outcomes are stored.
2. **Real-World:** User acts on advice and returns to the app after the resolution time.
3. **Verification:** User submits `OutcomeFeedback` (effectiveness, domains improved).
4. **Learning:** Feedback is stored in ChromaDB.

### Retraining Integration
```python
def compute_human_feedback_reward(predicted_obs, feedback: OutcomeFeedback) -> float:
    # 50% overlap between predicted vs. actual improvements
    # 50% subjective effectiveness score (0-10)
    return 0.5 * overlap + 0.5 * effectiveness_score
```
This score is used to upweight trajectories in subsequent training runs, teaching the model that its "optimistic" simulator predictions must align with the "sober" reality reported by humans.

## 5. Priority Execution Order
1. **Wire compute_task_reward into env.step()** (Foundation)
2. **Add format_compliance & plausibility checks** (Defensive)
3. **Log each function independently in TensorBoard** (Observability)
4. **Build OutcomeFeedback UI & Backend** (Differentiator)
5. **Close the loop via Feedback-Augmented Training** (Excellence)
