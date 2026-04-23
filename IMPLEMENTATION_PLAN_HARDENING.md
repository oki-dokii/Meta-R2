Now I have the full picture. Let me give you a complete, honest audit and plan.

  ---
  Part 1: How many reward functions do we have?

  Technically 6 functions in core/reward.py, but the reality is worse than it looks:

  ┌──────────────────────────────────┬─────────────────────┬─────────────────────────────────┐
  │             Function             │    What it does     │         Actually wired?         │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │                                  │ 4 components + 4    │ ✅ called in                    │
  │ compute_reward()                 │ penalties (local    │ lifestack_env.py:step()         │
  │                                  │ step)               │                                 │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │ compute_milestone_reward()       │ milestones hit /    │ ❌ defined, never called in env │
  │                                  │ total possible      │                                 │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │ compute_task_completion_reward() │ success conditions  │ ❌ defined, never called in env │
  │                                  │ met                 │                                 │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │ compute_replan_bonus()           │ recovery after      │ ❌ defined, never called in env │
  │                                  │ exo-events          │                                 │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │ compute_dead_end_penalty()       │ no routes remaining │ ❌ defined, never called in env │
  ├──────────────────────────────────┼─────────────────────┼─────────────────────────────────┤
  │ compute_task_reward()            │ orchestrator        │ ❌ defined, env.step() still    │
  │                                  │ combining all above │ calls only compute_reward()     │
  └──────────────────────────────────┴─────────────────────┴─────────────────────────────────┘

  So in practice: 1 reward function is active. 5 are dead code.

  ---
  Part 2: Gap vs. hackathon guide

  The guide explicitly says (§7, §8, §21):

  ▎ "Use multiple independent reward functions. If you only have one, it's easier to hack.
  ▎ Multiple independent checks reduce that risk."
  ▎ "Common mistake: using only one reward function"

  Here's the full gap analysis:

  ┌─────────────────────────────────────┬────────────────────────────────────────────────────┐
  │          Guide requirement          │                     Our status                     │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Execution success (task completed?) │ ❌ missing — compute_task_completion_reward exists │
  │                                     │  but unwired                                       │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Correctness (metrics actually       │ ✅ outcome_score in compute_reward                 │
  │ improved?)                          │                                                    │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Format compliance (valid JSON,      │ ❌ completely missing                              │
  │ required fields?)                   │                                                    │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Timeouts (step limit hit penalty?)  │ ❌ missing                                         │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Resource usage                      │ ✅ resource_efficiency_score                       │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Safety constraints (floor           │ ✅ CRITICAL_FLOOR_VIOLATION                        │
  │ violations)                         │                                                    │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Anti-cheating checks                │ ❌ missing — model can claim +50 metric change     │
  │                                     │ with 0 resource cost                               │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Process-aware feedback (step-level, │ ❌ missing                                         │
  │  not just outcome)                  │                                                    │
  ├─────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Multiple independent fns logged     │ ❌ only one fn running                             │
  │ separately                          │                                                    │
  └─────────────────────────────────────┴────────────────────────────────────────────────────┘

  Parameters currently used to compute reward (the one active fn):
  - outcome_score: delta across all 23 sub-metrics, domain-weighted 1/6 each
  - cascade_containment_score: % of metrics that didn't worsen
  - resource_efficiency_score: 1 - avg(time/20, money/500, energy/100)
  - relationship_preservation_score: sigmoid on relationship domain average delta
  - Penalties: CRITICAL_FLOOR (-0.50), CASCADE_SPREAD (-0.30), INACTION (-0.40),
  RELATIONSHIP_COLLAPSE (-0.15)

  Weights: 0.40 outcome + 0.25 containment + 0.20 efficiency + 0.15 preservation

  ---
  Part 3: Your post-result human feedback idea

  This is excellent and has a formal name: delayed human outcome signal. The idea:

  ▎ After the agent gives advice → user acts on it → after N hours/days when the effect resolves
  ▎ → user submits: "did it work? what else changed?"

  This gives you two things the simulator can't:
  1. Ground truth on whether advice was correct (the sim assumes the agent's claimed metric
  changes are real — a real human validates whether they actually happened)
  2. Unmeasured second-order effects (e.g., delegating to your boss improved work-life balance
  but damaged trust — the sim doesn't know that)

  ---
  The Plan

  Step 1 — Wire the orchestrator (1 day, critical)

  lifestack_env.py:step() currently calls compute_reward(). Change it to call
  compute_task_reward() when a Task is present. This instantly activates milestone + completion +
   replan rewards without writing new code.

  Step 2 — Add the 3 missing independent reward functions (1 day)

  a) reward_format_compliance(completion: str) -> float
  +1.0  valid JSON with all required fields (action_type, metric_changes, resource_cost,
  reasoning)
  +0.5  valid JSON but missing optional fields
  -0.5  invalid JSON / unparseable
  -1.0  empty or refusal
  This is what the guide calls "format compliance" and it directly prevents the most common GRPO
  failure mode (model outputs free text instead of JSON).

  b) reward_plausibility_check(action, resource_cost) -> float
  Anti-gaming check. The model cannot claim a +40 metric change while spending 0 resources.
  ratio = sum(abs(metric_changes.values())) / max(1, sum(resource_cost.values()))
  if ratio > 15: return -0.30   # claiming massive change for free
  if ratio > 8:  return -0.10   # suspicious
  else:          return  0.0    # fine

  c) reward_timeout_check(step_count, max_steps) -> float
  if step_count >= max_steps and not done: return -0.20

  Step 3 — Process-aware intermediate reward (1 day)

  The guide §9 says "naively assigning the same final reward to every token is inefficient." Add
  a reasoning coherence check — does the reasoning field actually mention the conflict domain?
  def reward_reasoning_coherence(action, conflict_domain) -> float:
      reasoning = action.reasoning.lower()
      if conflict_domain in reasoning: return +0.10
      if len(reasoning) < 20: return -0.10   # empty reasoning
      return 0.0
  Small signal, but it teaches the model to explain itself, which prevents degenerate outputs.

  Step 4 — Anti-hacking logging (already partially done, needs completion)

  train_trl.py logs every 20 completions to training_logs/generations.jsonl. Add a flag per log
  entry:
  "suspicious": reward > 0.8 and resource_cost == {}
  The guide says "a rising reward is not enough if the model is learning to exploit bugs" — this
  flag lets you spot it.

  Step 5 — Human outcome feedback loop (new feature, 2-3 days)

  Build a core/feedback.py module:

  @dataclass
  class OutcomeFeedback:
      episode_id: str
      submitted_at: datetime
      # Did the advice work overall? 0-10 scale
      overall_effectiveness: int
      # Which domains actually changed (user-reported)
      domains_improved: list[str]
      domains_worsened: list[str]
      # Free text: what unexpected effects happened?
      unexpected_effects: str
      # Time to resolution (hours)
      resolution_time_hours: float

  How it integrates:
  1. When episode ends in the app, store episode_id + the env's predicted outcome
  2. Show user a follow-up prompt after resolution_time_hours (you can estimate this from
  resource_cost.time)
  3. User submits OutcomeFeedback
  4. Store it in ChromaDB (memory.py) linked to the episode

  How it feeds back into training:
  def compute_human_feedback_reward(predicted_obs, feedback: OutcomeFeedback) -> float:
      # Did predicted domains match reality?
      predicted_improved = set(k.split('.')[0] for k, v in predicted_obs.metrics.items() if v >
  0)
      actual_improved = set(feedback.domains_improved)

      overlap = len(predicted_improved & actual_improved) / max(1, len(predicted_improved |
  actual_improved))
      effectiveness_score = feedback.overall_effectiveness / 10.0

      return 0.5 * overlap + 0.5 * effectiveness_score

  This score gets stored per episode. When you retrain, episodes with human feedback get
  upweighted — the model learns that its predicted metric changes need to match real-world
  outcomes, not just look good in the simulator.

  ---
  Priority Order

  1. Wire compute_task_reward into env.step()        → immediate 4x more reward signal
  2. Add format_compliance reward fn                 → prevents #1 GRPO failure mode
  3. Add plausibility_check reward fn               → blocks reward hacking
  4. Log each fn independently in breakdown         → satisfies guide §15 "monitor individual
  reward columns"
  5. Build OutcomeFeedback dataclass + app UI        → your unique differentiator vs other teams
  6. Wire human feedback into ChromaDB + retraining  → long-term loop

  Steps 1–4 are what separate a project that "has reward code" from one that actually meets the
  guide's standard. Steps 5–6 are your unique story — no other team will have real human outcome
  validation feeding back into training. make an md and push to github for planning
