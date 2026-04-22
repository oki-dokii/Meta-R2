# LifeStack Long-Horizon Upgrade Plan

## Context

LifeStack is a hackathon RL project that simulates life-decision tasks as a gym-style environment. Currently episodes are 5 steps long, use a single linear conflict path, have no hidden state or exogenous events, and reward only step-level metric improvements. Judges expect a proper long-horizon environment with 20+ steps, branching routes, dynamic world changes, partial observability, and task-completion rewards. This plan covers the full upgrade across pre-hackathon, Day 1, and Day 2.

**Key discoveries from reading the repo:**
- `app.py` is a **Gradio app** (not FastAPI). New "endpoints" = new Gradio tabs/functions.
- `max_steps = 5` is hardcoded in **two places**: `core/lifestack_env.py:93` AND `core/lifestack_gym_env.py:62`.
- The current reward is step-local only (no task-completion bonus exists anywhere).
- `memory.py` stores single decisions keyed by conflict title — no trajectory concept exists.
- `run_episode.py` orchestrates the loop outside the env (agent loop + env.step in separate code).
- ChromaDB is already persistent (`./lifestack_memory/`).
- `train_trl.py` already has a working GRPO loop with Unsloth — just needs new env interface.
- `app.py` imports `LongitudinalDemo` (not in the file listing — likely missing or in a data file).

---

## Proposed `core/task.py` Schema (SHARED CONTRACT — agree before writing any logic)

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class HiddenStateField:
    key: str               # e.g. "boss_mood"
    initial_value: Any     # e.g. "neutral"
    inspect_target: str    # e.g. "call_boss" — which inspect action type reveals this
    description: str       # shown to agent after reveal

@dataclass
class ExoEvent:
    step: int              # inject at this step (inclusive); -1 = probabilistic
    probability: float     # 1.0 = deterministic; <1.0 = random at each step
    id: str                # e.g. "ticket_price_spike"
    description: str       # what agent sees in next observation
    world_mutation: dict   # e.g. {"ticket_price": 450, "seats_remaining": 1}
    hidden_state_mutation: dict  # e.g. {"boss_mood": "angry"}
    closes_routes: list[str] = field(default_factory=list)  # route IDs this event blocks

@dataclass
class Milestone:
    id: str                # e.g. "flight_rebooked"
    description: str
    condition_key: str     # world/hidden key to check, e.g. "flight_rebooked"
    condition_value: Any   # e.g. True
    reward: float          # milestone reward added to episode total

@dataclass
class Route:
    id: str                # e.g. "rebook_premium"
    name: str
    description: str
    required_action_types: list[str]  # must use these tool actions to complete
    preconditions: dict    # world/hidden state checks, e.g. {"card_available": True}
    consequences: dict     # world mutations on route completion, e.g. {"flight_rebooked": True}
    closes_routes: list[str]  # route IDs this blocks
    milestones_unlocked: list[str]  # milestone IDs this route can hit
    final_reward: float    # bonus on route completion

@dataclass
class Task:
    id: str
    domain: str            # "flight_crisis" | "code_merge_crisis"
    goal: str
    constraints: dict      # e.g. {"budget_max": 400, "deadline_step": 18}
    hidden_state: dict     # full truth, agent never sees directly
    mutable_world: dict    # partial truth, some fields revealed by inspect
    visible_world: dict    # agent sees this at each step (subset of mutable_world)
    success_conditions: list[dict]  # e.g. [{"key": "flight_rebooked", "value": True}]
    failure_conditions: list[dict]  # e.g. [{"key": "missed_deadline", "value": True}]
    event_schedule: list[ExoEvent]
    viable_routes: list[Route]
    milestones: list[Milestone]
    horizon: int           # max steps (20–50)
    difficulty: int        # 1–5
    domain_metadata: dict  # domain-specific extra data (story text, etc.)
```

**Agreement required:** All three team members must freeze this schema before writing any logic.

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Cascade runaway over 30 steps** — DependencyGraph with 0.6 dampening can collapse metrics to 0 after repeated disruptions | HIGH | Add `metric_floor = 10.0` in `life_state.py`; cascade clamps to `max(floor, result)` not `max(0, result)`. Also add per-step cascade cap: max 3 metrics affected per step. |
| **Resource exhaustion on longer episodes** — Default 20h/500$/100e depletes in ~5 steps of aggressive action | HIGH | Scale budgets proportionally in `reset()`: `time=20*max_steps/5`, etc. Make configurable per-Task via `constraints`. |
| **Reward hacking: inspect spam** — Agent learns to `inspect` repeatedly for reward | HIGH | Anti-cheat: same hidden_state key cannot be inspected twice. Inspect has no intrinsic reward. |
| **Reward hacking: wait loops** — Agent waits forever | MEDIUM | Cap: max 3 consecutive `wait` actions; 4th `wait` triggers forced `escalate`. |
| **Reward hacking: rollback loops** — Rollback-execute-rollback cycle | MEDIUM | Rollback is only available once per route; marks action as `used_rollback=True` in state. |
| **Colab T4 session timeout** — Free Colab sessions timeout at ~12h | MEDIUM | Save checkpoint every 50 steps in `train_trl.py`. Use `trainer.save_checkpoint()` not just `save_pretrained_merged()` at end. |
| **ChromaDB trajectory bloat** — 30 steps × 23 metrics = ~700 floats per trajectory; 100 trajectories = 70k floats | LOW | Store trajectory summary (start/end state diff + route taken + total reward), not full step-by-step. |
| **OpenEnv API version** — `openenv-core>=0.2.3` in requirements; `_EnvBase`, `Action`, `Observation`, `State`, `Rubric` are OpenEnv abstractions. Need to confirm `create_app()` signature matches. | MEDIUM | Do not change `LifeStackAction`/`LifeStackObservation`/`LifeStackState` class names or fields. Add new fields as `Optional` to maintain backward compat. |
| **Two hardcoded `max_steps=5`** — Will break if only one is updated | HIGH | Fix both in Phase 0. Make `max_steps` a constructor param defaulting to `task.horizon` or 30. |
| **`app.py` imports `LongitudinalDemo`** — Not in file listing; may be missing class | MEDIUM | Check if it's defined inline or in a missing file. If missing, stub it for Day 1. |
| **`run_episode.py` duplicates env loop** — Agent loop lives outside env. New long-horizon logic must work in both env.step() and the external runner | MEDIUM | Keep `run_episode.py` working; it calls `env.step()` which now handles world mutation/events internally. |
| **TRL GRPO reward function parses prompt** — `lifestack_reward_fn` in `train_trl.py` reconstructs state from prompt text | MEDIUM | After env upgrade, update `build_prompt_for_conflict()` to include Task fields and update reward function accordingly. |

---

## File-by-File Change Plan

### NEW: `core/task.py`
- All dataclasses from schema above
- `FlightCrisisTask()` factory function returning a hardcoded Task instance (used for testing)
- `CodeMergeCrisisTask()` factory (stubbed Day 1, complete Day 2)
- No imports from other project files (pure data)

### MODIFIED: `core/lifestack_env.py`
**Existing:** `max_steps=5`, flat step logic, no hidden state, no events
**Changes:**
- Add `WorldEngine` inner class:
  - `__init__(task: Task)` — stores event schedule
  - `inject_events(step: int, world: dict, hidden: dict) -> list[ExoEvent]` — returns events fired this step, mutates world/hidden in-place
  - `get_closed_routes() -> set[str]` — routes blocked by events
- Add `PartialObsFilter`:
  - `filter(world: dict, revealed_keys: set[str]) -> dict` — returns only visible_world + revealed fields
- Change `__init__` signature: `__init__(task: Task = None, max_steps: int = 30)`
- In `reset()`: initialize `world_state`, `hidden_state`, `revealed_hidden_keys`, `current_task`, `active_route`, `milestones_achieved`, `used_rollback`
- In `step()`:
  1. Run `world_engine.inject_events(step)` → get fired events
  2. Apply ToolAction logic (inspect/plan/execute/wait/rollback/escalate)
  3. Check route preconditions; mark routes closed if violated
  4. Compute reward via updated `compute_reward()` 
  5. Check success/failure conditions from task
  6. Build observation with `partial_obs_filter`
- Add `render()` update: show task goal, active route, milestones achieved, events log
- **Preserve:** `LifeStackAction`, `LifeStackObservation`, `LifeStackState` class names and core fields (add Optional new fields)

### MODIFIED: `core/action_space.py`
**Add** `ToolAction` enum:
```python
class ToolActionType(str, Enum):
    INSPECT = "inspect"
    PLAN = "plan"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"
    WAIT = "wait"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
```
**Add** `ToolAction` dataclass:
```python
@dataclass
class ToolAction:
    action_type: ToolActionType
    target: str          # inspect target, execute target, communicate recipient, etc.
    parameters: dict     # action-specific params
    reasoning: str
```
**Add** `validate_tool_action(action: ToolAction, env_state: dict) -> tuple[bool, str]`
- Checks: inspect not repeated for same key, wait count ≤ 3, rollback only if not used
**Keep:** `AgentAction`, `PrimaryAction`, `CommunicationAction`, `EXAMPLE_ACTIONS` unchanged

### MODIFIED: `core/reward.py`
**Add** functions (do NOT remove `compute_reward`):
```python
def compute_milestone_reward(milestones_achieved: list[str], task: Task) -> float
def compute_task_completion_reward(success_conditions_met: list[bool], task: Task) -> float
def compute_replan_bonus(exo_events_seen: int, milestones_after_event: int) -> float
def compute_dead_end_penalty(routes_remaining: int) -> float
```
**Add** `compute_task_reward(...)` — orchestrates all components:
- 10% local metric delta (old `compute_reward`)
- 40% milestone rewards
- 30% task completion
- 10% replan bonus
- 10% efficiency
- Penalties: dead end (-0.5), rollback used (-0.1), cascade collapse (-0.3)

### MODIFIED: `core/life_state.py`
- Add `METRIC_FLOOR = 10.0` constant
- In `DependencyGraph.cascade()`: change `max(0, ...)` to `max(METRIC_FLOOR, ...)` for cascade-induced changes (not direct actions)
- Add `per_step_cascade_cap = 3` — BFS stops after affecting 3 nodes per step call

### MODIFIED: `agent/conflict_generator.py`
**Add** `TaskGenerator` class:
```python
class TaskGenerator:
    def generate(self, domain: str = None, difficulty: int = None) -> Task
    def generate_flight_crisis(self, difficulty: int) -> Task
    def generate_code_merge_crisis(self, difficulty: int) -> Task
```
**Keep:** `ConflictEvent`, `TEMPLATES`, `generate_conflict()`, `escalate_conflict()` fully intact

### MODIFIED: `agent/memory.py`
**Add** to `store_decision()`: optional `trajectory: list[dict] = None` and `route_outcome: str = None` params
**Add** `store_trajectory(task_id, route_taken, total_reward, trajectory_summary)` method:
- `trajectory_summary` = `{start_state_diff, end_state_diff, milestones_hit, events_seen, route_id, total_reward}`
- Store in separate ChromaDB collection `'trajectories'`
**Add** `retrieve_similar_trajectories(task_domain, current_world) -> list[dict]`
**Keep:** all existing methods unchanged

### MODIFIED: `app.py` (Gradio)
**Add** Tab 5: "Task Explorer":
- Shows current Task object (goal, constraints, visible routes, milestones)
- Shows event log for current episode
- Shows route lock status

**Add** helper functions:
- `task_html(task: Task) -> str` — renders goal, routes, milestones
- `event_log_html(events: list[ExoEvent]) -> str`
- `route_status_html(routes: list[Route], closed: set[str]) -> str`

**Keep:** All existing tabs and functions unchanged.

### MODIFIED: `openenv.yaml`
```yaml
metadata:
  max_episode_steps: 50
  task_domains: [flight_crisis, code_merge_crisis]
  # existing fields unchanged
```

### MODIFIED: `notebooks/LifeStack_Training.ipynb`
- Update env init cell to use `Task` objects
- Add Colab-ready GRPO cell with pinned versions:
  - `unsloth==2024.12.4`, `trl>=0.9`, `transformers>=4.45`
  - Model: `Qwen2.5-1.5B-Instruct` (fits T4 with 4-bit)
- Add reward breakdown visualization cell
- Checkpoint every 50 steps cell

---

## Task Domain Specs

### Domain 1: Flight Crisis
```
goal: "Catch the rescheduled flight and submit expense report by Sunday"
constraints: {budget_max: 400, deadline_step: 18, report_deadline_step: 22}
hidden_state:
  boss_mood: "neutral"      # revealed by inspect("call_boss")
  card_limit: 350           # revealed by inspect("check_card")
  partner_flexibility: 0.7  # revealed by inspect("text_partner")
mutable_world:
  ticket_price: 280         # changes at step 5 (spike to 450)
  seats_remaining: 3        # decreases each step probabilistically
  flight_rebooked: false
  report_submitted: false
event_schedule:
  step 5: {ticket_price: 450, seats_remaining: 1} (closes route "rebook_premium" if budget_max=400)
  step 8: {boss_mood: "annoyed"} (hidden_state mutation via msg)
  step 12: {card_blocked: true} (closes routes "rebook_premium", "hotel_stay")
routes:
  A: rebook_premium (precond: card_available=True, budget>=ticket_price)
  B: bus_and_remote (always open; slower, lower reward)
  C: hotel_next_day (precond: card_available=True; closed at step 12)
  D: family_loan (precond: partner_flexibility>=0.5; revealed after inspect)
  E: negotiate_deadline (precond: boss_mood != "furious"; closed if boss_mood="furious")
milestones:
  - inspect_boss: reward=0.05 (inspected boss_mood)
  - flight_rebooked: reward=0.20
  - report_submitted: reward=0.15
  - under_budget: reward=0.10 (total spend < budget_max)
horizon: 25
```

### Domain 2: Code Merge Crisis
```
goal: "Merge feature branch without breaking main; deploy by Friday"
constraints: {deploy_deadline_step: 30, max_conflicts: 5}
hidden_state:
  reviewer_strictness: "medium"  # revealed by inspect("check_pr_history")
  ci_flakiness_score: 0.3       # revealed by inspect("check_ci_logs")
  teammate_available: true       # revealed by inspect("ping_teammate")
mutable_world:
  conflicts_remaining: 4
  ci_passing: false
  pr_approved: false
  deploy_done: false
event_schedule:
  step 3: new commits land (conflicts_remaining += 2)
  step 7: CI fails (ci_passing: false, closes "direct_merge" route)
  step 10: reviewer blocks PR (pr_approved: false, mutates reviewer_strictness based on history)
routes:
  A: rebase (always open; risk of conflict if new commits land)
  B: cherry_pick (precond: conflicts_remaining <= 3)
  C: manual_merge (always open; slower, high reward if careful)
  D: rollback_split_pr (precond: used_rollback=False)
milestones:
  - conflicts_resolved: reward=0.15
  - ci_passing: reward=0.15
  - pr_approved: reward=0.15
  - deployed: reward=0.25
horizon: 30
```

---

## Hour-by-Hour Task Board

### Phase 0 — Pre-hackathon (Now → Apr 25 8 AM)

| Time | Person A (Env) | Person B (Task+Reward) | Person C (Training) |
|------|----------------|------------------------|---------------------|
| Now | Define `core/task.py` together — ALL THREE agree on schema | Same | Same |
| +1h | Add `ToolActionType` enum to `action_space.py` | Add `TaskGenerator` stub returning 1 hardcoded FlightCrisis Task | Colab smoke test: TRL+Unsloth GRPO on 5-step env. Confirm GPU, pin versions. |
| +2h | Stub `WorldEngine` in `lifestack_env.py` (inject_events returns []) | Define full FlightCrisis `mutable_world` and `hidden_state` dicts | Confirm training loop runs 100 steps with non-zero reward |
| +3h | Bump `max_steps=30` in both files + openenv.yaml. Run `run_episode.py`. | Build all 5 Route objects for Flight Crisis | Save Colab checkpoint; verify Unsloth merge path works |
| +4h | Confirm existing tests pass with max_steps=30 | Stub Code Merge task (fields only, no events yet) | Update `train_trl.py` to accept Task object from env |
| +4h | Sleep | Sleep | Sleep |

### Day 1 — Apr 25 (8 AM → Midnight)

| Time | Person A (Env) | Person B (Task+Reward) | Person C (Training) |
|------|----------------|------------------------|---------------------|
| 8–10 AM | Full WorldEngine: inject_events fires at correct steps, mutates world/hidden dicts | Complete event_schedule for Flight Crisis (3 events) | Trajectory memory: add store_trajectory() to memory.py |
| 10 AM–1 PM | PartialObsFilter: filter() hides hidden_state fields until revealed. inspect action reveals one field per call. | Milestone reward: compute_milestone_reward() fires when condition_key/value matches. Test manually. | /task and /routes Gradio tab (task_html, route_status_html) |
| 1–3 PM | **Integration test**: run_episode.py on 25-step Flight Crisis. Events inject at steps 5/8/12. inspect reveals boss_mood. Milestone fires on flight_rebooked. | **Integration test**: reward breakdown shows milestone + completion components. Fix any component that returns NaN or 0 always. | **Integration test**: training loop runs on new env, reward curve non-trivially non-zero |
| 3–5 PM | Fix cascade runaway: add METRIC_FLOOR=10, per-step cascade cap=3 | Code Merge task: full event_schedule (steps 3/7/10) + all 4 routes | Start Colab training on FlightCrisis. Qwen2.5-1.5B. Log every 50 steps. |
| 5–7 PM | Reward hacking audit: can inspect spam score high? Can wait=30 score? Can rollback-loop? Fix each exploit. | Reward hacking audit: same. Anti-cheat: inspect blocks on repeated key, wait cap=3 consecutive | Monitor training. If reward flats at 0, check reward_fn in train_trl.py. |
| 7–9 PM | Smoke test: both task domains, 5 episodes each, no crashes | Smoke test all milestones + failure conditions fire correctly | Save checkpoint. Run before/after comparison: baseline vs trained on FlightCrisis. |
| 9–11 PM | render() update: show task goal, active route, milestone log, event log | Efficiency penalty tuning: make it punish but not dominate | Push notebook to Colab. Test from cold start. |
| 11 PM | Commit stable checkpoint | Commit | Commit |

### Day 2 — Apr 26 (8 AM → 8 PM)

| Time | Person A (Env) | Person B (Task+Reward) | Person C (Training) |
|------|----------------|------------------------|---------------------|
| 8–10 AM | Curriculum variants: easy Flight Crisis (deadline_step=25, no card block event) | Easy/medium/hard difficulty scaling for both tasks | Longer Kaggle (P100) training run. Curriculum: easy → hard. |
| 10 AM–12 PM | Render polish: episode timeline readable by judges | Reward breakdown display in Gradio | Inference test: load merged model, run 5 episodes, compare reward vs baseline |
| 12–2 PM | HF Space setup: test Space endpoint with $200 credits | Code Merge fully working end-to-end | Demo script: baseline → reward output → trained → measurable gain |
| 2–4 PM | README architecture diagram | Reward breakdown chart (matplotlib, per episode) | Record 2-min demo |
| 4–6 PM | Final smoke test of both domains | Final reward hacking audit pass | BLOG.md update |
| 6–8 PM | Submit | Submit | Submit |

---

## Verification Plan

1. **Unit test `core/task.py`**: instantiate both Task objects, check all fields present and typed correctly
2. **Unit test `WorldEngine`**: inject step 5 event on FlightCrisis, verify `ticket_price` updates from 280 to 450
3. **Unit test `PartialObsFilter`**: hidden field not in output before inspect; in output after inspect("call_boss")
4. **Unit test `compute_milestone_reward`**: set `flight_rebooked=True` in world, verify milestone fires with reward=0.20
5. **Integration test (run_episode.py)**: 25-step FlightCrisis episode with LifeStackAgent. Check: (a) reward > 0, (b) events fired at correct steps, (c) route closed after card_blocked event, (d) milestones logged in obs.metadata
6. **Reward hacking test**: manually set actions to pure inspect for 25 steps — verify total_reward < 0.1. Pure wait for 25 steps — verify truncation fires and penalty applied.
7. **Training test**: run `train_trl.py` for 50 steps on Colab. Verify reward_curve shows non-flat trend.
8. **Backward compat test**: run `run_episode.py` with the old `conflict_generator.generate_conflict()` (no Task object). Should not crash.

---

## Critical Files

| File | Status | Owner |
|------|--------|-------|
| `core/task.py` | NEW | A+B together first |
| `core/lifestack_env.py` | MAJOR CHANGE | A |
| `core/action_space.py` | ADD ToolAction enum | B |
| `core/reward.py` | ADD task-level functions | B |
| `core/life_state.py` | ADD floor + cap | A |
| `agent/conflict_generator.py` | ADD TaskGenerator | B |
| `agent/memory.py` | ADD trajectory storage | C |
| `app.py` | ADD Task Explorer tab | C |
| `openenv.yaml` | UPDATE max_episode_steps | A |
| `notebooks/LifeStack_Training.ipynb` | UPDATE for new env | C |
| `scripts/train_trl.py` | UPDATE reward_fn + prompt | C |
