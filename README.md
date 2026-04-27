---
title: LifeStack
emoji: 🪐
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - life-management
  - curriculum-learning
---

[![HF Space](https://img.shields.io/badge/HF_Space-Live_Demo-yellow?style=flat-square)](https://huggingface.co/spaces/jdsb06/meta-r2)
[![v1 Model](https://img.shields.io/badge/Model-lifestack--grpo-blue?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo)
[![v2 Model](https://img.shields.io/badge/Model-lifestack--grpo--v2-teal?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo-v2)
[![v3 Model](https://img.shields.io/badge/Model-lifestack--grpo--v3-violet?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo-v3)
[![v4 Model](https://img.shields.io/badge/Model-lifestack--grpo--v4-green?style=flat-square)](https://huggingface.co/jdsb06/lifestack-grpo-v4)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.3-orange?style=flat-square)](https://github.com/meta-pytorch/OpenEnv)

---

## Quick Links for Judges

| What | Link |
|------|------|
| 🚀 **Live Demo (HF Space)** | https://huggingface.co/spaces/jdsb06/meta-r2 |
| 📝 **Mini-Blog / Model Card** | https://huggingface.co/jdsb06/lifestack-grpo-v4 |
| 🎓 **Re-runnable Colab Training Notebook** | [Colab_GRPO_Training.ipynb](https://colab.research.google.com/drive/184NGSW_BkLQAmOGgDlgEqrArJqBHr-B9?usp=sharing) |
| 📊 **Training Evidence (plots)** | [plots/](plots/) — reward curve, loss curve, components, summary |
| 🏋️ **v4 Training Log (raw)** | [train_run_v4.log on HF](https://huggingface.co/jdsb06/lifestack-grpo-v4/blob/main/train_run_v4.log) |
| 🏋️ **v1 Training Log (raw)** | [train_run_v1.log](train_run_v1.log) |
| 🤖 **All models** | [v1](https://huggingface.co/jdsb06/lifestack-grpo) · [v2](https://huggingface.co/jdsb06/lifestack-grpo-v2) · [v3](https://huggingface.co/jdsb06/lifestack-grpo-v3) · [v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) |
| 🌐 **GitHub Source** | https://github.com/oki-dokii/Meta-R2 |

---

It's Friday 6PM. Your flight got cancelled. Your card got declined trying to rebook. Slack notification: your boss moved Monday's deadline to Sunday. You have $200 in cash, five hours of usable energy, and four people expecting you in different places. You ask an AI for help. It books the flight — but the only affordable option has a 12-hour layover. It drafts a message to your boss — but the tone triggers a "clarification" meeting that eats more time. Every solution applied in isolation opens a new wound somewhere else.

LifeStack is built around exactly this problem. It is a multi-domain life-management reinforcement learning environment on OpenEnv 0.2.3, modelling a human life as a system of 23 interdependent metrics across 6 life domains, where every action cascades through a 32-edge directed dependency graph with consequences you often cannot predict. We trained `Qwen2.5-1.5B-Instruct` on this environment using GRPO (Group Relative Policy Optimization) — a 5-stage single-step curriculum followed by episodic multi-step fine-tuning — producing a model that allocates time, money, and energy across competing life priorities without collapsing any domain below crisis threshold.

---

## What Judges Can Explore in the Live Demo

The [live HF Space](https://huggingface.co/spaces/jdsb06/meta-r2) runs **v4** on a T4 GPU with ten interactive tabs.

---

### Tab 1 — Situational Portal (main simulation)

The entry point. Pick a persona (one of several pre-built `SimPerson` profiles) and a life conflict from `data/conflicts.json`, then hit **Start Simulation**.

What happens under the hood: `LifeStackEnv.reset()` seeds the 23 metrics from the conflict's `primary_disruption` via `DependencyGraph.cascade()`. The agent builds a compact training-format prompt (conflict title, 5 key metrics, remaining budget, personality hint, strategy line) and runs it through the v4 LoRA adapter. The resulting JSON action is parsed, normalized, and applied to the environment via `env.step()`.

What you see: a **Domain Risk Heatmap** (6 colour-coded blocks that shift red as metrics drop), a live metric bar chart across all 23 values updating after each step, the agent's chosen action with reasoning and a per-step reward score, and an **AI Risk Prediction** panel showing a 7-step γ=0.9 discounted rollout from the post-action state — the model's own forecast of where the cascade leads if nothing else changes.

A **Context Augmentation (RAG)** toggle injects personality-filtered ChromaDB memories as few-shot context. Toggle it off and back on to see the agent's phrasing shift when it has past precedents to draw from.

---

### Tab 2 — Untrained vs GRPO-Trained

Same conflict, same persona, two inference paths running in parallel: vanilla Groq `llama-3.3-70b-versatile` (no RL, no fine-tuning) on the left; the v4 GRPO adapter on the right.

Both sides run through the same `LifeStackEnv`, apply the same cascade graph, and get scored by the same 10 reward functions. The reward delta badge at the bottom shows the numeric gap.

The pattern that surfaces repeatedly: the untrained LLM picks generic, low-specificity actions (`delegate everything`, `take a break`) with vague resource costs. The trained model names a target domain, specifies resource units, and reasons about cascade risk. This is a direct consequence of GRPO training on the plausibility and reasoning-coherence reward functions.

---

### Tab 3 — Model Evolution (v1 → v4)

All four HuggingFace adapter checkpoints (`lifestack-grpo`, `v2`, `v3`, `v4`) are loaded simultaneously and run against the same free-text scenario you type in.

Each version gets its own card showing action type, target domain, reward score, and raw reasoning. The policy shift across versions is visible: v1 defaults to rest or generic delegation; v2 starts using communicate and negotiate; v3 introduces multi-step reasoning fragments; v4 explicitly names resource costs and cascade effects.

The summary cards above the outputs show the dominant action type per version and the reward score, making the training progression concrete without requiring the reader to dig into logs.

---

### Tab 4 — Memory Effect

A controlled ablation of the RAG layer. Pick a conflict and persona, then click **Run Both Episodes**. The left card runs the agent cold — no memory context, no few-shot injection. The right card runs the same scenario with full ChromaDB retrieval enabled.

The warm run shows the retrieved memories inline: which past decisions were surfaced, their original reward, which personality type they came from (after the personality-aware memory update), and how similar they were to the current situation. You can watch the agent's reasoning change register when it has prior experience to reference.

The reward delta between cold and warm is the live version of the `avg_no_memory: 1.13 → avg_with_memory: 2.45` (+116%) comparison in `data/before_after_comparison.json`. Note: this number measures the RAG augmentation effect — the same v4 model, with and without ChromaDB context. It is not a trained-vs-untrained comparison; the GRPO training improvement (baseline −0.07 → v4 peak +0.856) is a separate result shown in the training plots.

---

### Tab 5 — Personality Lab

Two personas face the same crisis side by side. Pick **Persona A** and **Persona B** from the dropdowns (e.g. Alex the high-stress executive vs. Sam the calm pragmatist), choose a conflict, and click **Compare Response**.

Each persona runs through `SimPerson.respond_to_action()` — a Big Five uptake scaling model that modulates how much of the agent's recommended metric change the person actually absorbs. High neuroticism amplifies stress cascades; high conscientiousness increases follow-through on negotiate and prepare actions. The OCEAN scores for each persona are shown in the output card.

Both runs now use personality-filtered RAG: Alex retrieves past decisions tagged with Alex's profile first, Sam gets Sam's. The result is two genuinely divergent action paths from the same underlying crisis — not just different personas receiving the same advice differently, but the agent actively recommending different strategies.

---

### Tab 6 — What-If Lab

After any simulation run, this tab shows the agent's chosen action alongside three counterfactual alternatives generated by forcing specific action types (`rest`, `negotiate`, `delegate`). All four come from the v4 adapter — there is no rule-based fallback.

Each alternative card shows the action description, the projected metric changes, the reward score, and a trade-off summary (e.g. `NEGOTIATE reduces workload but costs 3h time and raises relationship.professional_network by 8`). This is powered by `agent/counterfactuals.py` running the environment forward from the same pre-action state for each forced action type.

The tab demonstrates that the policy has genuine range — v4 can produce meaningfully different plans for the same situation, not just a single memorized response.

---

### Tab 7 — Task Explorer

A read-only browser for all conflict scenarios in `data/conflicts.json` plus the built-in `FlightCrisisTask`. Select a scenario from the dropdown to see its full structure: scenario text, goal, viable routes with action type mappings, milestones with reward weights, resource budget, primary disruption vector, and any scheduled `ExoEvent` objects with their trigger steps.

This is the fastest way to understand what the RL environment actually looks like from the inside — what routes exist, what the success conditions are, and how hard the task is (difficulty rating shown as a badge).

---

### Tab 8 — Analytics

Live session statistics that update as you run episodes across any other tab.

**Training Progression chart** — bar chart comparing eval reward per training run (Base, Run 1–4, v3, v4) against the untrained baseline (dashed line). The numbers are hardcoded from actual training logs.

**Base Model Per-Domain Reward chart** — radar/bar showing where the raw Qwen2.5-1.5B fails most. mental_wellbeing (−0.25), physical_health (−0.17), and career (−0.14) are the worst domains — the ones where the base model actively made things worse.

**Key Findings cards** — four callout boxes covering the fix that mattered (greedy regex extraction → +85.7%), the honest limitation (v1 plateau at difficulty 1), the new capability (multi-step episode return), and v4's peak 0.856 reward.

**Live Session Reward chart** — rolling line chart of reward scores from your current session's episodes. Memory & Agent Stats shows total ChromaDB memories, memory precedents, session average reward, and top action type.

---

### Tab 9 — System Map

An interactive vis-network graph of the 32-edge `DependencyGraph`. Each of the 23 metrics is a node; edges are weighted and colour-coded (green = positive influence, red = negative drag). Edge thickness scales with weight magnitude.

Click any node to highlight its first-order neighbours. The engine note below the graph explains the cascade mechanics: a 10-point workload spike triggers the `career.workload → mental_wellbeing.stress_level (0.70)` edge, which then fires `stress → sleep_quality (−0.55)` and `stress → motivation (−0.40)`, attenuated 40% per hop by the 0.6 dampening factor.

**Recenter Map** and **Freeze Nodes** controls in the bottom-left allow repositioning the layout. The graph is fetched live from `/api/simulation/graph` so it reflects the actual in-memory `DependencyGraph` object.

---

### Tab 10 — Verification (Outcome Signal)

A human-in-the-loop feedback form. After running an episode you can submit the real-world outcome by entering the episode trace ID (shown on every action card), rating actual effectiveness 0–10, logging actual hours spent, and noting any unexpected deviations from the plan.

Submitted feedback is stored in the ChromaDB `feedback_collection` and retrieved by `reward_human_feedback_fn` during future training runs — closing the reinforcement loop with real outcome data rather than simulated reward. The `OutcomeFeedback` object written here is the same type consumed by `compute_human_feedback_reward()` in `core/feedback.py`.

---

## System architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       HuggingFace Space                          │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │   app_flask.py  :7860   │  │      server.py  :8000       │   │
│  │   Flask demo UI         │  │   OpenEnv FastAPI server    │   │
│  └────────────┬────────────┘  └─────────────────────────────┘   │
│               │  (independent services — no direct call between) │
│               ▼                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    LifeStackAgent                          │  │
│  │  ┌──────────────────────────┐  ┌────────────────────────┐  │  │
│  │  │  Qwen2.5-1.5B-Instruct   │  │   LifeStackMemory      │  │  │
│  │  │  + LoRA adapter          │  │   ChromaDB             │  │  │
│  │  │  GRPO fine-tuned (v4)    │  │   decisions (seeded)   │  │  │
│  │  └──────────────────────────┘  │   trajectories         │  │  │
│  │                                │   feedback             │  │  │
│  │                                └────────────────────────┘  │  │
│  └────────────────┬───────────────────────────────────────────┘  │
│                   │  LifeStackAction (JSON)                      │
│                   ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                      LifeStackEnv                          │  │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌───────────────┐  │  │
│  │  │ WorldEngine │  │ DependencyGraph  │  │LifeStackVerif-│  │  │
│  │  │ (ExoEvents) │─►│ 32 edges        │  │ier            │  │  │
│  │  └─────────────┘  │ 0.6 dampening   │  │success /      │  │  │
│  │                   └───────┬─────────┘  │milestone check│  │  │
│  │                           │            └───────────────┘  │  │
│  │   LifeMetrics · 23 values across 6 domains                 │  │
│  │   career · finances · relationships ·                      │  │
│  │   physical_health · mental_wellbeing · time                │  │
│  │                           │                                │  │
│  │   10 reward fns  ·  episodic GRPO · horizon=3              │  │
│  │   group-relative advantage  ·  env.step() anti-hack guards │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

Training (offline · scripts/train_trl.py):
  Qwen2.5-1.5B-Instruct ──► LifeStackGRPOTrainer ──► 5-stage curriculum
  JSON-boundary masking ──► LoRA checkpoint ──► HF Hub (lifestack-grpo-v4)
```

---

## How it works

The RL loop is concrete and sequential:

1. `reset()` picks a life conflict scenario — a `Task` object containing a `visible_world`, `hidden_state`, `event_schedule`, `viable_routes`, and `milestones`. The `LifeMetrics` state (23 values across 6 domains) is seeded via `DependencyGraph.cascade()` on the task's initial disruption.
2. The agent receives a `LifeStackObservation` — flattened 23-metric values, remaining budget (`time_hours`, `money_dollars`, `energy_units`), current step count, visible world state, active milestones, and fired event IDs.
3. The agent submits a `LifeStackAction` — structured JSON with `action_type`, `target_domain`, `metric_changes`, `resource_cost`, and `reasoning`. Ten valid action types exist: `negotiate`, `communicate`, `delegate`, `spend`, `reschedule`, `rest`, `deprioritize`, `execute`, `prepare`, `self_care`.
4. `WorldEngine` injects any scheduled or probabilistic `ExoEvent` objects for this step, mutating world state and potentially closing routes the agent was planning to use.
5. The `DependencyGraph` propagates metric changes through 32 directed edges with a 0.6 dampening factor per hop — touching `career.workload` automatically reduces `time.free_hours_per_week` and raises `mental_wellbeing.stress_level`, which then degrades `physical_health.sleep_quality`, and so on.
6. `LifeStackVerifier` checks success conditions, failure conditions, and newly unlocked milestones. Ten reward functions score the step.
7. `env.rollout(n_steps=7, gamma=0.9)` runs seven null actions from the post-step state, computing the discounted long-horizon consequence signal without mutating the environment.
8. The episode ends when `step_count >= task.horizon`, any metric drops to `<= 10`, explicit failure conditions trigger, or all viable routes are exhausted.

---

## The core mental model

### Visible state (LifeStackObservation)

The agent sees `metrics` (all 23 flattened values), `resources` (time/money/energy remaining), `step` count, `metadata.world_state` (partial view of `mutable_world` plus any keys the agent has explicitly `inspect`ed), `metadata.goal`, `metadata.active_route`, `metadata.milestones`, and `metadata.events`.

### Hidden consequences

The agent cannot directly observe: cascade effects downstream from its action (`DependencyGraph` propagates silently), future `ExoEvent` triggers (some are probabilistic with `step=-1`), `hidden_state` fields (revealed only by an `inspect` action), and long-horizon stress accumulation that makes later steps harder.

### The 6 life domains and 23 metrics

These are the exact `LifeMetrics` fields in `core/life_state.py`. The 8 task domains add `transport_crisis` and `code_merge_crisis` as scenario types but they do not have their own metric subfields — they affect the same 23 values.

| Domain | Metrics | Notes |
|--------|---------|-------|
| `career` | `satisfaction`, `workload`, `stability`, `growth_trajectory` | `workload` is inverted (higher = worse) |
| `finances` | `liquidity`, `debt_pressure`, `monthly_runway`, `long_term_health` | `debt_pressure` is inverted |
| `relationships` | `romantic`, `family`, `social`, `professional_network` | All positive-direction |
| `physical_health` | `energy`, `fitness`, `sleep_quality`, `nutrition` | All positive-direction |
| `mental_wellbeing` | `stress_level`, `clarity`, `motivation`, `emotional_stability` | `stress_level` is inverted |
| `time` | `free_hours_per_week`, `commute_burden`, `admin_overhead` | `commute_burden` and `admin_overhead` are inverted |

All metrics are clamped to `[0, 100]`. Cascade propagation has a floor of `10.0` for downstream nodes (`is_cascade=True`). Any metric reaching `<= 10` terminates the episode as a failure.

---

## Main building blocks

**`core/lifestack_env.py`** — `LifeStackEnv` is the central class. It inherits from OpenEnv's `Environment` base (with a multi-level import shim that falls back gracefully if `openenv-core` is absent, controlled by the `USING_MODERN_API` flag). `reset()` initializes a `LifeStackState` — which holds the current `LifeMetrics`, `ResourceBudget`, inspected keys, consecutive wait count, rollback flag, and task state — then constructs a `WorldEngine` for the task. `step()` handles rollback, inspect, wait, route execution, resource deduction, metric updates with cascade, task progression checks, and reward computation. `rollout()` snapshots `_internal_state`, runs `n_steps` null rest actions, accumulates discounted rewards, then restores the snapshot — no side effects. `PartialObsFilter` controls what the agent sees: `visible_world` plus any keys the agent has previously `inspect`ed (hidden-state reveals are tagged `{"source": "inspect"}`).

**`core/reward.py`** — Contains `compute_reward()` (the base step reward: 4 components with weights 0.50/0.25/0.15/0.10) and `compute_task_reward()` (the task-aware orchestrator: 6 components with weights 0.35/0.25/0.15/0.10/0.05 + penalties). Four standalone scoring functions are also here: `reward_format_compliance()`, `reward_plausibility_check()`, `reward_timeout_check()`, and `reward_reasoning_coherence()`. These are called both from within the environment and directly as GRPO reward functions in `scripts/train_trl.py`.

**`core/life_state.py`** — Defines `LifeMetrics` (6 domain dataclasses, 23 total fields), `ResourceBudget` (with `deduct()` that refuses overdrafts), and `DependencyGraph` (32 directed weighted edges with 0.6 dampening per hop). The cascade algorithm uses a BFS queue with a `per_step_cascade_cap=3` to prevent runaway propagation from a single large action.

**`core/cascade_utils.py`** — `animate_cascade()` replays the same `DependencyGraph` propagation frame-by-frame (primary → first-order → second-order) and returns a list of snapshots. The Flask demo's vis-network cascade visualization is driven by this function.

**`core/verifier.py`** — `LifeStackVerifier` is a stateless utility class with three methods: `check_success()` (evaluates task success conditions against world/hidden state), `check_failure()` (evaluates failure conditions plus metric-death check at `<= 10`), and `check_new_milestones()` (scans task milestones for newly satisfied conditions). `get_route_status()` counts reachable routes given the current closed-route set and precondition state.

**`core/task.py`** — Defines the `Task` dataclass (with `domain`, `goal`, `constraints`, `hidden_state`, `mutable_world`, `visible_world`, `success_conditions`, `failure_conditions`, `event_schedule`, `viable_routes`, `milestones`, `horizon`, `difficulty`). Two concrete task factories: `FlightCrisisTask()` (horizon=30, two competing routes, two timed ExoEvents) and `CodeMergeCrisisTask()` (horizon=10, two competing routes). `TaskGenerator` holds both and calls `get_random_task()`. The richer `TaskGenerator` in `agent/conflict_generator.py` covers all 8 task domains with template-based generation.

**`agent/agent.py`** — `LifeStackAgent` handles prompt construction, model inference, and action parsing. On initialization it starts a background thread to load the GRPO adapter from `DEFAULT_HF_MODEL_REPO = "jdsb06/lifestack-grpo-v4"` (via `PeftConfig` + `PeftModel`). If the local model fails, it falls back to Groq API (`llama-3.3-70b-versatile`). `build_prompt()` injects memory context from `LifeStackMemory.build_few_shot_prompt()` when available. The in-memory `self.memory` list stores per-episode decision history for context.

**`agent/memory.py`** — `LifeStackMemory` wraps three ChromaDB collections: `decisions`, `trajectories`, and `feedback`. It uses `SentenceTransformer('all-MiniLM-L6-v2')` for embeddings, falling back to a hash-based 384-dim vector when the model is unavailable. `retrieve_similar()` does a two-pass personality-aware query: it first filters by `personality_type` metadata (same-profile precedents) using ChromaDB's `where` clause, then fills any remaining slots from the global pool — so the agent always gets `n` results but prefers its own personality's history. `build_few_shot_prompt()` formats retrieved memories as a few-shot context block with per-entry `[personality_type]` labels injected into the agent's prompt. Every `store_decision()` call saves a `personality_type` field alongside the decision so the filter is populated from the first interaction. On first init, if the collection is empty, it auto-hydrates from `data/preseeded_memory*.json`.

**`scripts/train_trl.py`** — The main GRPO training script. `train_curriculum()` runs 5 stages of single-step training using plain `GRPOTrainer` (stage 1 uses 3 reward functions for JSON warm-up, stages 2-5 use 10). `train_episodic_curriculum()` runs 2+ stages of multi-step trajectory training using `LifeStackGRPOTrainer` — a subclass that overrides `_prepare_inputs()` to zero-out `completion_mask` tokens after the first complete JSON object. Both support `--resume` via `curriculum_state.json`. The `load_model()` function tries Unsloth's 4-bit path first, then falls back to plain HF+PEFT (`Qwen/Qwen2.5-1.5B-Instruct` in bf16 on A100+, fp16 otherwise).

---

## Reward design

### Per-step reward (base step, no task)

`compute_reward()` computes four components and combines them:

```
base_reward = (0.50 × outcome_score) +
              (0.25 × cascade_containment_score) +
              (0.15 × resource_efficiency_score) +
              (0.10 × relationship_preservation_score)
```

Then applies penalties before clamping to `[-1.0, 1.0]`.

### Per-step reward (task-aware)

`compute_task_reward()` replaces the base formula with a task-oriented weighting:

```
base_reward = (0.35 × milestone_score) +
              (0.25 × completion_score) +
              (0.15 × outcome_score_local) +
              (0.10 × replan_score) +
              (0.10 × efficiency_score) +
              (0.05 × preservation_score)
```

### The 10 GRPO reward functions (stages 2–5 of single-step curriculum)

| Function | What it measures | Range |
|----------|-----------------|-------|
| `reward_format_fn` | Valid JSON, all 5 required keys, valid `action_type` and `target_domain` | `[-1.0, +1.0]` |
| `reward_clean_eos_fn` | Model stops within 8 chars of closing JSON brace | `[-0.10, +0.20]` |
| `reward_route_target_fn` | `target_domain` matches a listed route ID with `action_type="execute"` | `[0.0, +0.30]` |
| `reward_plausibility_fn` | Ratio of `total_metric_delta / normalized_cost`; penalizes free lunch | `[-0.30, 0.0]` |
| `reward_task_success_fn` | Task completion score from full environment simulation | `[0.0, +1.0]` |
| `reward_milestone_fn` | Milestone achievement score (partial credit per milestone hit) | `[0.0, +1.0]` |
| `reward_replan_fn` | Ability to hit milestones *after* an ExoEvent fires | `[0.0, +0.5]` |
| `reward_reasoning_fn` | Reasoning length + logical-connector check + action-keyword alignment | `[-0.30, +0.30]` |
| `reward_human_feedback_fn` | ChromaDB similarity to past episodes with stored human outcome feedback | `[0.0, +1.0]` |
| `reward_longterm_fn` | 7-day γ=0.9 discounted rollout from current post-action state | `[-1.0, +1.0]` |

Stage 1 (JSON warm-up) uses only `reward_format_fn`, `reward_clean_eos_fn`, `reward_route_target_fn` with weights `[1.0, 1.5, 1.0]`.

Episodic training (v4) uses `reward_episode_format_fn`, `reward_clean_eos_fn`, `reward_episode_plausibility_fn`, `reward_episode_return_fn` with weights `[1.0, 0.5, 0.5, 2.0]`.

### Anti-hacking specifics

The `reward_plausibility_check()` gate fires for any non-trivial metric claim (`total_delta > 3.0`) with zero resource cost, returning `-0.30`. For non-zero cost, the claim-to-cost ratio threshold is 150 (severe: `-0.30`) and 80 (suspicious: `-0.10`). `reward_reasoning_coherence()` requires both structural logic connectors ("because", "therefore", etc.) AND semantic alignment between reasoning text and `action_type` — misaligned reasoning gets `-0.20`.

---

## Training stack

The base model is `Qwen/Qwen2.5-1.5B-Instruct` (1.5B parameters). `load_model()` in `scripts/train_trl.py` first tries the Unsloth path (`unsloth/Qwen2.5-1.5B-Instruct` in 4-bit), then falls back to plain HF+PEFT with `LoraConfig(r=16, lora_alpha=16)` targeting all 7 projection layers. On A100 80GB the HF+PEFT path is recommended — set `LIFESTACK_NO_UNSLOTH=1` to skip Unsloth and avoid dtype collisions with the 4-bit kernel.

`LifeStackGRPOTrainer` (used in episodic training) overrides `_prepare_inputs()` to locate the JSON-object boundary in each completion and zero out `completion_mask` tokens beyond it. The token IDs stay intact — no distribution shift — but the KL and advantage terms see only the ~100 JSON tokens instead of 480 trailing explanation tokens. This gives roughly 3–5× sharper gradient signal per step.

### 5-stage single-step curriculum (`train_curriculum`)

| Stage | LR | Reward functions | Dataset |
|-------|----|-----------------|---------|
| 1 | 8e-6 | `format`, `clean_eos`, `route_target` | 100 prompts, difficulty 1 |
| 2 | 5e-6 | All 10 | 100 prompts, difficulty ≥ 1 |
| 3 | 3e-6 | All 10 | 100 prompts, difficulty ≥ 2 |
| 4 | 2e-6 | All 10 | 100 prompts, difficulty ≥ 3 |
| 5 | 1e-6 | All 10 | 100 prompts, difficulty ≥ 4 |

Difficulty advances when stage-end reward ≥ 0.0. Curriculum state is persisted to `curriculum_state.json` — resume with `--resume`.

### Episodic curriculum (`train_episodic_curriculum`, v3/v4)

Uses `LifeStackGRPOTrainer` + 4 episodic reward functions. Each prompt asks the model to return an `{"actions": [...]}` sequence executed by the real environment for 3 steps (default `episode_horizon=3`). `reward_episode_return_fn` is the dominant signal (weight 2.0) — it computes the discounted trajectory reward plus a `+0.25` terminal bonus on success.

### Key environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENENV_HOST` | `0.0.0.0` | OpenEnv server bind address |
| `OPENENV_PORT` | `8000` | OpenEnv server port |
| `OPENENV_MAX_SESSIONS` | `4` | Max concurrent environment instances |
| `PORT` | `7860` | Flask UI port (set by HF Spaces) |
| `LIFESTACK_NO_UNSLOTH` | unset | Set to `1` to skip Unsloth and use HF+PEFT |
| `GROQ_API_KEY` | unset | Enables Groq API fallback in the agent |
| `HF_TOKEN` | unset | Required for loading private model checkpoints |

### A note on training environment

We ran into consistent import and binary wheel issues on both Kaggle and Colab while iterating on the same dependency stack. The training that produced the v4 logs and plots was done on HuggingFace (Jupyter + terminal on the Space), which stayed stable. The Colab notebook in this repo is for anyone who wants to re-run the same CLI elsewhere — expect hosted runtimes to be hit-or-miss with this stack.

---

## Anti-reward-hacking

Every mechanism described here exists in the actual code:

`reward_plausibility_check()` in `core/reward.py` fires a `-0.30` penalty whenever an action claims non-trivial metric improvements (`total_delta > 3.0`) with zero resource cost. This closes the "free metrics" exploit where the model would request large improvements and specify `resource_cost: {}`.

`REST_NOT_JUSTIFIED` fires in `compute_reward()` with `-0.25` whenever `action_type == "rest"` and the average energy metric is above 30. Without this, the model learned that rest has near-zero cost and positive energy recovery, making it the optimal action regardless of context.

`WAIT_CAP_EXCEEDED` in `step()` applies `+15.0` to `mental_wellbeing.stress_level` after 4 consecutive `wait` actions. Without this cap the model could stall indefinitely.

`CASCADE_SPREAD_WIDER` in `compute_reward()` fires with `-0.30` when the number of metrics that worsened after the step exceeds the number of metrics the agent directly changed. This penalizes actions that cause broader damage than their stated scope.

`ROLLBACK_USED` fires with `-0.1` penalty in `compute_task_reward()` — once per episode. Rollback is available as a recovery mechanism but costs. The `used_rollback` flag on `LifeStackState` prevents using it twice.

`action_type` is validated against `VALID_ACTION_TYPES` (a `frozenset` in `core/reward.py`). `metric_changes` in `step()` are filtered through `allowed_keys = set(self._internal_state.current_metrics.flatten().keys())` — the model cannot invent new metric paths.

---

## Longitudinal memory and personalization

Every episode writes a decision to ChromaDB via `store_decision()`, tagged with `personality_type` (the persona label from the Personality Lab dropdown). On the next request for the same personality, `retrieve_similar()` runs a two-pass query:

1. **Same-personality pass** — `where={"personality_type": {"$eq": person_label}}` retrieves semantically similar decisions from that profile's own history.
2. **Global fill-up pass** — if fewer than 3 results are returned (e.g. first run, Space restart), the remainder come from the shared pool of all past decisions and the pre-seeded wisdom pool.

The result is that Alex's past high-stress `communicate` decisions surface as few-shot context for Alex's next crisis, not Sam's low-stress `delegate` decisions. Each entry in the injected prompt block is labeled `[personality_type]` so the model can weight matched-profile precedents appropriately.

On startup the `decisions` collection auto-hydrates from `data/preseeded_memory*.json`, giving the agent a baseline of curated high-reward decisions before any user has interacted. User-generated decisions accumulate on top of this throughout the Space's uptime.

The `memory_compare` and `memory_ablation` endpoints expose this directly: run the same conflict with memory disabled vs. enabled to see the few-shot context change the chosen action and reward score.

---

## Deployment

LifeStack runs as two services inside a single Docker container. `start.sh` (the Docker `CMD`) starts `server.py` in the background on port 8000, waits 3 seconds for it to bind, then runs `app_flask.py` in the foreground on port 7860.

HuggingFace health-checks port 7860, so the Flask process must stay alive. `server.py` is written to return silently on any failure (missing `openenv-core`, `create_app` crash, etc.) rather than calling `raise SystemExit` — if the OpenEnv server crashes, the Flask UI keeps the Space alive and the demo remains functional. All `server.py` error paths use `print()` + `return`.

```bash
# Docker
docker build -t lifestack .
docker run -p 7860:7860 -p 8000:8000 lifestack

# Local dev — run services separately
python server.py                  # OpenEnv on :8000
python app_flask.py               # Flask UI on :7860

# Connect via OpenEnv client
from openenv import EnvClient
client = EnvClient("http://localhost:8000")
obs = client.reset()
```

---

## Project map

```
Meta-r2/
├── core/
│   ├── lifestack_env.py      # LifeStackEnv, WorldEngine, PartialObsFilter
│   ├── life_state.py         # LifeMetrics (23 values), DependencyGraph (32 edges)
│   ├── reward.py             # compute_reward, compute_task_reward, 4 scoring fns
│   ├── cascade_utils.py      # animate_cascade() — frame-by-frame cascade replay
│   ├── action_space.py       # AgentAction, PrimaryAction, apply_action
│   ├── task.py               # Task, ExoEvent, Route, Milestone, FlightCrisisTask
│   ├── verifier.py           # LifeStackVerifier (success/failure/milestone checks)
│   ├── feedback.py           # OutcomeFeedback, compute_human_feedback_reward
│   └── metric_schema.py      # VALID_METRIC_PATHS, normalize_metric_path
├── agent/
│   ├── agent.py              # LifeStackAgent (GRPO model + Groq fallback)
│   ├── memory.py             # LifeStackMemory (ChromaDB, 3 collections)
│   ├── conflict_generator.py # TaskGenerator (8 domains), TEMPLATES, ExoEvent injection
│   ├── conflict_predictor.py # ConflictPredictor (pattern matching on episode history)
│   └── counterfactuals.py    # What-if reasoning over metric snapshots
├── intake/
│   ├── simperson.py          # SimPerson (Big Five personality → action uptake scaling)
│   ├── intake.py             # Data ingestion pipeline
│   ├── gmail_intake.py       # Gmail OAuth source
│   └── calendar_intake.py    # Google Calendar OAuth source
├── server/
│   └── app.py                # OpenEnv FastAPI server (create_app wrapper)
├── scripts/
│   ├── train_trl.py          # Full GRPO training — single-step + episodic curriculum
│   ├── eval.py               # Evaluation script, random baseline
│   ├── plot_training.py      # Parse training logs → reward/loss/component plots
│   └── smoke_test.py         # Fast pipeline check (no GPU needed)
├── notebooks/
│   ├── Colab_GRPO_Training.ipynb   # Re-runnable training notebook
│   └── Training_Evidence.ipynb     # Plot generation from logs
├── data/
│   ├── conflicts.json              # 20+ named conflict scenarios
│   ├── training_log.json           # Episode reward history
│   └── before_after_comparison.json # Memory ablation results
├── plots/                          # Training evidence plots (v4 real run)
├── docs/                           # All project documentation
├── app_flask.py                    # Flask UI — port 7860, 10 tabs, Chart.js
├── server.py                       # Crash-safe OpenEnv server entry point
├── start.sh                        # Docker CMD — dual service startup
├── Dockerfile                      # python:3.11-slim, exposes 7860 + 8000
└── openenv.yaml                    # OpenEnv manifest
```

---

## Training evidence

**Two separate improvements — don't conflate them:**

| What improved | How much | What caused it |
|---|---|---|
| Single-step eval reward | −0.07 (base) → **+0.856** (v4 peak) | GRPO fine-tuning across 4 runs |
| Episode reward with memory | 1.13 → **2.45** (+116%) | RAG augmentation (ChromaDB few-shot) on the same trained model |

The GRPO improvement is the primary training result. The 116% memory gain is an ablation showing what the ChromaDB RAG layer adds on top of the trained model — it is not a trained-vs-untrained comparison.

![4-panel training summary — reward curve (top-left), loss curve (top-right), per-reward-function components (bottom-left), and run-over-run eval progression vs the −0.07 untrained baseline (bottom-right)](plots/training_summary.png)

*4-panel training summary: reward curve, loss curve, per-component breakdown, and run-over-run eval progression against the −0.07 untrained baseline.*

![v1 → v4 model progression — each column is one HF adapter checkpoint on the same scenario; action type and reward shift left-to-right as training matures](plots/model_progression_v1_to_v4.png)

*Cross-version model progression: all four LoRA checkpoints (v1→v4) on the same scenario. Policy shift is visible — v1 defaults to rest/delegate, v4 names resource costs and cascade targets.*

---

## Quick start

```bash
# Clone and install
git clone https://github.com/oki-dokii/Meta-R2.git
cd Meta-R2
bash setup.sh        # creates .venv, installs requirements.txt, runs smoke test

# Activate and run
source .venv/bin/activate
python app_flask.py  # Flask UI → http://localhost:7860
python server.py     # OpenEnv server → http://localhost:8000

# Smoke test (no GPU, no downloads)
python scripts/smoke_test.py

# Full training (requires GPU, ~2h on A100 80GB)
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py

# Dry run (CPU, validates pipeline end-to-end in ~60s)
python scripts/train_trl.py --dry-run
```

The trained checkpoints for v1, v2, v3, and v4 are on HuggingFace Hub at `jdsb06/lifestack-grpo`, `jdsb06/lifestack-grpo-v2`, `jdsb06/lifestack-grpo-v3`, and `jdsb06/lifestack-grpo-v4`. Load them with standard `PeftModel.from_pretrained()` against `Qwen/Qwen2.5-1.5B-Instruct` as the base.
