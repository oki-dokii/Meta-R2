import copy
from typing import Any, Optional, Dict, List
from pydantic import Field

from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
from core.metric_schema import normalize_metric_path
from core.reward import compute_reward, compute_task_reward
from core.task import Task, ExoEvent, Route, Milestone, FlightCrisisTask
from core.verifier import LifeStackVerifier

try:
    from openenv.core import Environment, Action, Observation, State
    from openenv.core.env_server.types import EnvironmentMetadata
    from openenv.core.rubrics import Rubric
    USING_MODERN_API = True
except ImportError:
    try:
        from openenv.env import Env as Environment
        from pydantic import BaseModel
        # Shims for missing classes in older/alternative openenv
        class Action(BaseModel): pass
        class Observation(BaseModel): pass
        class State(BaseModel): pass
        class Rubric:
            def __init__(self, *a, **k): pass
            def compute(self, *a, **k): return 0.0
        EnvironmentMetadata = None
        USING_MODERN_API = False
    except ImportError:
        # Final fallback — must use BaseModel so Pydantic subclasses work
        from pydantic import BaseModel
        class Environment:
            def __init__(self, rubric=None): self.rubric = rubric
            def reset(self, *a, **k): pass
            def step(self, *a, **k): pass
        class Action(BaseModel): pass
        class Observation(BaseModel): pass
        class State(BaseModel): pass
        class Rubric:
            def __init__(self, *a, **k): pass
            def compute(self, *a, **k): return 0.0
        EnvironmentMetadata = None
        USING_MODERN_API = False

class LifeStackAction(Action):
    """Structured action for LifeStack."""
    metric_changes: Dict[str, float] = Field(default_factory=dict, description="Metric adjustment deltas")
    resource_cost: Dict[str, float] = Field(default_factory=dict, description="Time, money, and energy costs")
    actions_taken: int = Field(default=0, description="Number of atomic actions taken")
    
    # ToolAction fields (Long-horizon)
    action_type: Optional[str] = Field(default=None, description="inspect, plan, execute, etc.")
    target: Optional[str] = Field(default=None, description="e.g. route_id or hidden_key")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = Field(default=None)
    completion: Optional[str] = Field(default=None)

    inspect_target: Optional[str] = Field(default=None, description="Optional hidden state key to inspect")
    is_rollback: bool = Field(default=False, description="Set true to rollback the previous action.")

    @classmethod
    def from_agent_action(cls, agent_action: Any) -> "LifeStackAction":
        """Unified converter from legacy AgentAction to LifeStackAction."""
        primary = agent_action.primary
        return cls(
            action_type=primary.action_type,
            target=primary.target_domain, # Mapping target_domain to target
            metric_changes=primary.metric_changes,
            resource_cost=primary.resource_cost,
            reasoning=agent_action.reasoning,
            completion=getattr(agent_action, 'raw_completion', ""),
            actions_taken=1
        )

class LifeStackObservation(Observation):
    """Observation returned by LifeStack."""
    metrics: Dict[str, float] = Field(default_factory=dict, description="Flattened 23-domain life metrics")
    resources: Dict[str, float] = Field(default_factory=dict, description="Current budget remaining")
    step: int = Field(default=0, description="Current episode step")
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LifeStackState(State):
    """Internal state of the LifeStack environment."""
    current_metrics: LifeMetrics = Field(default_factory=LifeMetrics)
    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    episode_id: Optional[str] = None
    step_count: int = 0
    inspected_keys: list = Field(default_factory=list) # revealed keys
    consecutive_waits: int = 0
    used_rollback: bool = Field(default=False)
    rollback_penalty_charged: bool = Field(default=False)
    previous_metrics: Optional[LifeMetrics] = None
    previous_budget: Optional[ResourceBudget] = None

    # New task fields
    current_task: Optional[Task] = None
    active_route_id: Optional[str] = None
    milestones_achieved: list = Field(default_factory=list)
    world_state: dict = Field(default_factory=dict)
    hidden_state: dict = Field(default_factory=dict)
    fired_event_ids: list = Field(default_factory=list)
    exo_events_seen: int = 0
    milestones_after_event: int = 0
    closed_route_ids: set = Field(default_factory=set)
    # Legacy / Personality fields
    person: Optional[Any] = None
    agent_history: List[tuple] = Field(default_factory=list)
    current_conflict: Optional[Any] = None
    cumulative_rel_delta: float = Field(default=0.0)
class LifeStackRubric(Rubric):
    """Standard reward rubric for LifeStack."""
    def forward(self, action: LifeStackAction, observation: LifeStackObservation) -> float:
        # In LifeStack, reward is usually computed inside step() for state-transition access.
        # This rubric provides a hook for external reward evaluation if needed.
        return observation.reward if observation.reward is not None else 0.0

class PartialObsFilter:
    @staticmethod
    def filter(task: Task, revealed_keys: list) -> dict:
        """Returns visible_world plus any keys the agent has explicitly inspected.

        Revealed keys are checked against mutable_world first, then hidden_state.
        Keys sourced from hidden_state are wrapped as
        ``{"value": <val>, "source": "inspect"}`` so the agent knows they were
        obtained via an inspect action rather than being freely observable.
        """
        obs_world = copy.deepcopy(task.visible_world)
        for k in revealed_keys:
            if k in task.mutable_world:
                obs_world[k] = task.mutable_world[k]
            elif k in task.hidden_state:
                obs_world[k] = {"value": task.hidden_state[k], "source": "inspect"}
        return obs_world

class WorldEngine:
    def __init__(self, task: Task):
        self.task = task
        self.closed_routes = set()

    def inject_events(self, step: int, world: dict, hidden: dict) -> list[ExoEvent]:
        import random
        fired = []
        for event in self.task.event_schedule:
            fire = False
            if event.step == step:
                fire = True
            elif event.step == -1:
                if random.random() < event.probability:
                    fire = True
            
            if fire:
                fired.append(event)
                # Apply mutations
                world.update(event.world_mutation)
                hidden.update(event.hidden_state_mutation)
                for rid in event.closes_routes:
                    self.closed_routes.add(rid)
        return fired

    def get_closed_routes(self) -> set[str]:
        return self.closed_routes

_EnvBase = Environment[LifeStackAction, LifeStackObservation, LifeStackState] if USING_MODERN_API else Environment

class LifeStackEnv(_EnvBase):
    """
    LifeStack Environment v1.1 — Refactored for OpenEnv 0.2.3 compliance.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self, seed: Optional[int] = None, task=None, max_steps: int = 30):
        if USING_MODERN_API:
            super().__init__(rubric=LifeStackRubric())
        else:
            super().__init__()
            
        self.max_steps = getattr(task, 'horizon', max_steps) if task else max_steps
        
        self.metadata_internal = {
            'name': 'LifeStack-v1',
            'version': '1.1.0',
            'description': 'Premium multi-domain life conflict resolution simulation',
            'max_episode_steps': self.max_steps
        }
        
        self.graph = DependencyGraph()
        self._internal_state = LifeStackState()

    def get_metadata(self):
        if not USING_MODERN_API:
            return self.metadata_internal
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name=self.metadata_internal['name'],
            version=self.metadata_internal['version'],
            description=self.metadata_internal['description']
        )

    @property
    def state(self) -> LifeStackState:
        return self._internal_state

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, 
              task: Optional[Task] = None, conflict: Optional[Any] = None, 
              budget: Optional[dict] = None, person: Optional[Any] = None,
              agent_history: Optional[List[tuple]] = None, **kwargs) -> LifeStackObservation:
        """Resets the environment. Seed and task/conflict can be provided."""
        if USING_MODERN_API and getattr(self, 'rubric', None):
            self.rubric.reset()
        
        if seed is not None:
            import random
            random.seed(seed)

        # 1. Initialize Task
        self._internal_state.current_task = task or FlightCrisisTask()
        self.max_steps = getattr(self._internal_state.current_task, 'horizon', 30)
        
        # 2. Reset State
        self._internal_state.episode_id = episode_id
        self._internal_state.step_count = 0
        self._internal_state.current_metrics = LifeMetrics()
        self._internal_state.inspected_keys = []
        self._internal_state.consecutive_waits = 0
        self._internal_state.used_rollback = False
        self._internal_state.rollback_penalty_charged = False
        self._internal_state.previous_metrics = None
        self._internal_state.previous_budget = None
        self._internal_state.rollback_penalty_charged = False
        self._internal_state.cumulative_rel_delta = 0.0
        
        # Task state
        self._internal_state.world_state = copy.deepcopy(self._internal_state.current_task.mutable_world)
        self._internal_state.hidden_state = copy.deepcopy(self._internal_state.current_task.hidden_state)
        self._internal_state.milestones_achieved = []
        self._internal_state.active_route_id = None
        self._internal_state.fired_event_ids = []
        self._internal_state.exo_events_seen = 0
        self._internal_state.milestones_after_event = 0
        self._internal_state.closed_route_ids = set()
        
        self._internal_state.person = person
        self._internal_state.agent_history = agent_history or []
        self._internal_state.current_conflict = conflict
        
        self.world_engine = WorldEngine(self._internal_state.current_task)

        # 3. Budget Scaling
        scale = max(1.0, self.max_steps / 5.0)
        constraints = self._internal_state.current_task.constraints
        self._internal_state.budget = ResourceBudget(
            time_hours=budget.get("time", constraints.get("time", 20.0 * scale)) if budget else constraints.get("time", 20.0 * scale),
            money_dollars=budget.get("money", constraints.get("money", 500.0 * scale)) if budget else constraints.get("money", 500.0 * scale),
            energy_units=budget.get("energy", constraints.get("energy", 100.0 * scale)) if budget else constraints.get("energy", 100.0 * scale)
        )

        if conflict:
            # Legacy disruption support
            disruption = conflict.primary_disruption if hasattr(conflict, 'primary_disruption') else conflict
            self._internal_state.current_metrics = self.graph.cascade(self._internal_state.current_metrics, disruption)
            if budget is None and hasattr(conflict, 'resource_budget'):
                 rb = conflict.resource_budget
                 self._internal_state.budget = ResourceBudget(
                     time_hours=rb.get("time", 20.0),
                     money_dollars=rb.get("money", 500.0),
                     energy_units=rb.get("energy", 100.0)
                 )

        return self._get_obs()

    def _get_obs(self, done: bool = False, reward: Optional[float] = None,
                 success: bool = False, failure: bool = False,
                 failure_reason: str = "", routes_remaining: int = 0) -> LifeStackObservation:
        revealed_world = PartialObsFilter.filter(
            self._internal_state.current_task,
            self._internal_state.inspected_keys
        )
        
        return LifeStackObservation(
            metrics=self._internal_state.current_metrics.flatten(),
            resources={
                "time": self._internal_state.budget.time_hours,
                "money": self._internal_state.budget.money_dollars,
                "energy": self._internal_state.budget.energy_units
            },
            step=self._internal_state.step_count,
            done=done,
            reward=reward,
            metadata={
                "world_state": revealed_world,
                "goal": self._internal_state.current_task.goal,
                "active_route": self._internal_state.active_route_id,
                "milestones": self._internal_state.milestones_achieved,
                "events": self._internal_state.fired_event_ids,
                "success": success,
                "failure": failure,
                "failure_reason": failure_reason,
                "routes_remaining": routes_remaining,
                "conflict_title": self._internal_state.current_conflict.title if hasattr(self._internal_state.current_conflict, 'title') else "Custom Task",
                "person": self._internal_state.person.name if hasattr(self._internal_state.person, 'name') else "Unknown"
            }
        )

    def _update_metric(self, path: str, delta: float):
        """Internal helper for non-cascading updates."""
        path = normalize_metric_path(path)
        if '.' not in path:
            return
        domain_name, sub_name = path.split('.', 1)
        domain = getattr(self._internal_state.current_metrics, domain_name, None)
        if domain and hasattr(domain, sub_name):
            val = getattr(domain, sub_name)
            setattr(domain, sub_name, max(0.0, min(100.0, val + delta)))

    def step(self, action: LifeStackAction, timeout_s: Optional[float] = None, **kwargs) -> LifeStackObservation:
        """Executes one step in the environment using LifeStackAction logic."""
        if isinstance(action, dict):
            action = LifeStackAction(**action)

        task = self._internal_state.current_task
        state_before = copy.deepcopy(self._internal_state.current_metrics)
        info_msgs = []
        
        # 0. Personality Drift & Legacy Escalation
        if self._internal_state.person:
            drift_event = self._internal_state.person.drift(self._internal_state.step_count)
            if drift_event:
                path = drift_event.get('metric', '')
                delta = drift_event.get('delta', 0)
                if path and '.' in path:
                    self._update_metric(path, delta)
                info_msgs.append(f"DRIFT: {drift_event['reason']}")

        if self._internal_state.current_conflict and self._internal_state.step_count == 2:
            from agent.conflict_generator import adaptive_escalate
            conflict = self._internal_state.current_conflict
            if hasattr(conflict, 'difficulty') and conflict.difficulty < 5:
                new_conflict, reason = adaptive_escalate(conflict, self._internal_state.agent_history)
                if new_conflict.id != conflict.id:
                    self._internal_state.current_conflict = new_conflict
                    info_msgs.append(f"ESCALATION: {reason} -> {new_conflict.title}")
        fired_events = self.world_engine.inject_events(
            self._internal_state.step_count, 
            self._internal_state.world_state,
            self._internal_state.hidden_state
        )
        if fired_events:
            self._internal_state.exo_events_seen += len(fired_events)
            for e in fired_events:
                self._internal_state.fired_event_ids.append(e.id)
                info_msgs.append(f"EVENT_FIRED: {e.description}")
        
        self._internal_state.closed_route_ids.update(self.world_engine.get_closed_routes())

        # 2. Tool Logic & Metric Changes
        tool_type = action.action_type or (
            "rollback" if action.is_rollback else 
            "inspect" if action.inspect_target else 
            "execute"
        )
        
        allowed_keys = set(self._internal_state.current_metrics.flatten().keys())
        metric_changes = {k: v for k, v in action.metric_changes.items() if k in allowed_keys}
        resource_cost = copy.deepcopy(action.resource_cost)
        
        # Handle Rollback
        if tool_type == "rollback":
            self._internal_state.step_count += 1
            if self._internal_state.used_rollback:
                info_msgs.append("ROLLBACK_DENIED: Already used once.")
                return self._get_obs(reward=-0.1)
            if not self._internal_state.previous_metrics:
                return self._get_obs(reward=0.0)
            self._internal_state.current_metrics = copy.deepcopy(self._internal_state.previous_metrics)
            self._internal_state.budget = copy.deepcopy(self._internal_state.previous_budget)
            self._internal_state.used_rollback = True
            self._internal_state.rollback_penalty_charged = True  # Penalty baked into the -0.1 return above
            return self._get_obs(reward=-0.1)

        # Save state for future rollback
        self._internal_state.previous_metrics = copy.deepcopy(self._internal_state.current_metrics)
        self._internal_state.previous_budget = copy.deepcopy(self._internal_state.budget)

        # Handle Inspect
        if tool_type == "inspect":
            target = action.target or action.inspect_target
            if target:
                if target in self._internal_state.inspected_keys:
                    info_msgs.append(f"INSPECT_REDUNDANT: {target}")
                else:
                    self._internal_state.inspected_keys.append(target)
                    info_msgs.append(f"INSPECT_REVEALED: {target}")
                    # Emit an explicit signal when a hidden-state value is uncovered.
                    if target in task.hidden_state:
                        info_msgs.append(
                            f"INSPECT_REVEALED_HIDDEN: {target} = {task.hidden_state[target]}"
                        )
        
        # Handle Wait
        if tool_type == "wait":
            self._internal_state.consecutive_waits += 1
            if self._internal_state.consecutive_waits >= 4:
                metric_changes["mental_wellbeing.stress_level"] = metric_changes.get("mental_wellbeing.stress_level", 0) + 15.0
                info_msgs.append("WAIT_CAP_EXCEEDED: Forced stress applied.")
        else:
            self._internal_state.consecutive_waits = 0

        # Handle Route Execution
        if tool_type == "execute" and action.target:
            route = next((r for r in task.viable_routes if r.id == action.target), None)
            if route:
                # Check closed
                if route.id in self._internal_state.closed_route_ids:
                    info_msgs.append(f"ROUTE_BLOCKED: {route.name}")
                else:
                    # Check preconditions
                    pre_ok = True
                    for k, v in route.preconditions.items():
                        current_v = self._internal_state.hidden_state.get(k, self._internal_state.world_state.get(k))
                        if current_v != v:
                            pre_ok = False
                            break
                    
                    if not pre_ok:
                        info_msgs.append(f"PRECONDITIONS_FAILED for {route.name}")
                    else:
                        # Success: Apply route
                        self._internal_state.active_route_id = route.id
                        self._internal_state.world_state.update(route.consequences)
                        info_msgs.append(f"ROUTE_SUCCESS: {route.name}")

        # 3. Resource Deduction (must happen BEFORE metric changes to prevent budget-bypass exploit)
        deduct_ok = self._internal_state.budget.deduct(
            time=resource_cost.get('time', 0.0),
            money=resource_cost.get('money', 0.0),
            energy=resource_cost.get('energy', 0.0)
        )
        if not deduct_ok:
            info_msgs.append("RESOURCE_DEPLETED_ACTION_BLOCKED")
            metric_changes = {}  # Discard changes — agent can't afford this action

        # 4. Apply Metric and Cascade
        sig_changes = {k: v for k, v in metric_changes.items() if abs(v) > 5.0}
        for k, v in metric_changes.items():
            if k not in sig_changes:
                self._update_metric(k, v)

        if sig_changes:
            self._internal_state.current_metrics = self.graph.cascade(self._internal_state.current_metrics, sig_changes)

        # 5. Task Progression Check
        success_mets = LifeStackVerifier.check_success(task, self._internal_state.world_state, self._internal_state.hidden_state)
        failure_mets = LifeStackVerifier.check_failure(task, self._internal_state.world_state, self._internal_state.hidden_state, self._internal_state.current_metrics.flatten())
        
        # Check milestones dynamically
        newly_met = LifeStackVerifier.check_new_milestones(task, self._internal_state.world_state, self._internal_state.hidden_state, self._internal_state.milestones_achieved)
        for mid in newly_met:
            self._internal_state.milestones_achieved.append(mid)
            if self._internal_state.exo_events_seen > 0:
                self._internal_state.milestones_after_event += 1
            info_msgs.append(f"MILESTONE_UNLOCKED: {mid}")

        # 6. Reward Calculation (Task-Aware)
        routes_rem, _ = LifeStackVerifier.get_route_status(task, self._internal_state.closed_route_ids, self._internal_state.world_state, self._internal_state.hidden_state)

        # Determine cascade collapse
        metrics_after = self._internal_state.current_metrics.flatten()
        metrics_before = state_before.flatten()
        collapse = any(metrics_after[k] < 20 and metrics_before[k] >= 20 for k in metrics_after)

        # Track cumulative relationship erosion across steps
        rel_keys_cum = [k for k in metrics_after if k.startswith('relationships.')]
        if rel_keys_cum:
            step_rel_delta = sum(metrics_after[k] - metrics_before[k] for k in rel_keys_cum) / len(rel_keys_cum)
            self._internal_state.cumulative_rel_delta += step_rel_delta

        # Increment step_count BEFORE reward so timeout_check fires correctly
        self._internal_state.step_count += 1

        # Rollback penalty fires only once per episode
        rollback_this_step = self._internal_state.used_rollback and not self._internal_state.rollback_penalty_charged
        if rollback_this_step:
            self._internal_state.rollback_penalty_charged = True

        # conflict_domain from task.domain (not conflict.title) to prevent empty-string bypass
        conflict_domain = task.domain if task and hasattr(task, 'domain') else ""

        if task:
            reward, breakdown = compute_task_reward(
                state_before=state_before,
                state_after=self._internal_state.current_metrics,
                resources_used=resource_cost,
                actions_taken=action.actions_taken,
                milestones_achieved=self._internal_state.milestones_achieved,
                success_conditions_met=success_mets,
                exo_events_seen=self._internal_state.exo_events_seen,
                milestones_after_event=self._internal_state.milestones_after_event,
                routes_remaining=routes_rem,
                rollback_used=rollback_this_step,
                cascade_collapse=collapse,
                task=task,
                reasoning=getattr(action, 'reasoning', ""),
                completion=getattr(action, 'completion', ""),
                conflict_domain=conflict_domain,
                step_count=self._internal_state.step_count,
                max_steps=self.max_steps,
                metric_changes=metric_changes,
                cumulative_rel_delta=self._internal_state.cumulative_rel_delta,
                action_type=tool_type
            )
            # Charge the rollback penalty only once per episode
            if self._internal_state.used_rollback and not self._internal_state.rollback_penalty_charged:
                self._internal_state.rollback_penalty_charged = True
        else:
            reward, breakdown = compute_reward(
                state_before=state_before,
                state_after=self._internal_state.current_metrics,
                resources_used=resource_cost,
                actions_taken=action.actions_taken,
                metric_changes=metric_changes,
                completion=getattr(action, 'completion', ""),
                action_type=tool_type
            )
        
        # 7. End Conditions
        # Check if ANY success condition is met. 
        # For multi-goal tasks with mutually exclusive routes, any() allows termination.
        is_success = any(success_mets) if (success_mets and len(task.success_conditions) > 0) else False
        is_task_failure = any(val == True for val in failure_mets)
        metric_death = any(v <= 10 for v in metrics_after.values())
        
        failure_reason = ""
        if is_task_failure:
            reasons = [cond['key'] for i, cond in enumerate(task.failure_conditions) if failure_mets[i]]
            failure_reason = f"Condition failed: {', '.join(reasons)}"
        elif metric_death:
            dead_metrics = [k for k, v in metrics_after.items() if v <= 0]
            failure_reason = f"Metrics hit zero: {', '.join(dead_metrics)}"
        elif routes_rem == 0 and not is_success:
            failure_reason = "Dead end: No reachable routes left."

        terminated = is_task_failure or metric_death
        truncated = self._internal_state.step_count >= self.max_steps
        if is_success:
            truncated = True
        done = terminated or truncated

        observation = self._get_obs(
            done, 
            reward, 
            success=is_success, 
            failure=terminated, 
            failure_reason=failure_reason, 
            routes_remaining=routes_rem
        )
        observation.metadata["breakdown"] = breakdown
        observation.metadata["info"] = info_msgs
        return observation

    def rollout(self, n_steps: int = 7, gamma: float = 0.9) -> dict:
        """
        Simulate n_steps null/rest actions starting from the current env state.

        Intended to be called immediately AFTER env.step(model_action) so it
        models "what happens to your life over the next N days if nothing
        extraordinary occurs."

        The env state is fully restored after the rollout — calling this is
        side-effect-free from the caller's perspective.

        Returns:
            {
              "discounted_reward": float,          # γ-discounted cumulative
              "immediate_r0": float,               # reward from the action (caller supplies)
              "trajectory": [                      # one entry per simulated day
                  {
                      "step": int,                 # 1-indexed future day
                      "reward": float,
                      "metrics": Dict[str, float], # flattened snapshot
                      "discounted_contribution": float,
                  },
                  ...
              ],
              "n_steps_completed": int,
            }
        """
        saved_state = copy.deepcopy(self._internal_state)

        null_action = LifeStackAction(
            action_type="rest",
            target="time",
            metric_changes={},
            resource_cost={},
            actions_taken=0,
        )

        trajectory = []
        cumulative = 0.0

        for t in range(n_steps):
            obs = self.step(null_action)
            disc = (gamma ** (t + 1)) * float(obs.reward)
            cumulative += disc
            trajectory.append({
                "step": t + 1,
                "reward": float(obs.reward),
                "metrics": dict(obs.metrics),
                "discounted_contribution": round(disc, 5),
            })
            if obs.done:
                break

        # Restore — rollout must not mutate the env visible to the caller
        self._internal_state = saved_state

        return {
            "discounted_reward": round(cumulative, 5),
            "trajectory": trajectory,
            "n_steps_completed": len(trajectory),
        }

    def render(self):
        """Vibrant status report of the current state and task progress."""
        task = self._internal_state.current_task
        print("\n" + "═"*70)
        print(f"🎯 GOAL: {task.goal} | Horizon: {self._internal_state.step_count}/{self.max_steps}")
        print(f"⌛ TIME: {self._internal_state.budget.time_hours:.1f}h | 💵 MONEY: ${self._internal_state.budget.money_dollars:.1f} | ⚡ ENERGY: {self._internal_state.budget.energy_units:.1f}")
        
        if self._internal_state.active_route_id:
            print(f"🛣️ ACTIVE ROUTE: {self._internal_state.active_route_id}")
        
        print(f"⭐ MILESTONES: {', '.join(self._internal_state.milestones_achieved) or 'None'}")
        
        if self._internal_state.fired_event_ids:
            print(f"🚨 EVENTS: {', '.join(self._internal_state.fired_event_ids)}")

        flat = self._internal_state.current_metrics.flatten()
        domain_labels = {
            "career": "💼 CAREER",
            "finances": "💰 FINANCES",
            "relationships": "❤️ RELATIONSHIPS",
            "physical_health": "💪 PHYSICAL",
            "mental_wellbeing": "🧠 MENTAL",
            "time": "📅 TIME"
        }
        
        for dom, label in domain_labels.items():
            print(f"\n{label}")
            submetrics = {k: v for k, v in flat.items() if k.startswith(dom + ".")}
            inverted = {"stress_level", "debt_pressure", "workload", "commute_burden", "admin_overhead"}
            for name, val in submetrics.items():
                short = name.split('.')[1]
                icon = ("🔴" if val > 70 else "🟢") if short in inverted else ("🟢" if val > 70 else "🔴")
                if 40 <= val <= 70: icon = "🟡"
                print(f"  {icon} {short:20} : {val:5.2f}")
        print("═"*70)


def env_render_compact(env, obs):
    """Compact printer for testing."""
    print(f"STEP: {obs.step} | REWARD: {obs.reward:.3f} | DONE: {obs.done}")
    if obs.metadata.get("breakdown", {}).get("penalties_fired"):
        print(f"  ⚠️ PENALTIES: {obs.metadata['breakdown']['penalties_fired']}")


def main():
    env = LifeStackEnv()
    
    # 1. Reset with Friday 6PM Conflict
    conflict = {
        "career.workload": 30.0,
        "finances.liquidity": -40.0
    }
    print("Initializing environment with Friday 6PM conflict...")
    env.reset(conflict=conflict)
    env.render()
    
    total_reward = 0
    metrics_history = []
    
    # 2. Sequential Actions
    scenarios = [
        {
            "name": "GOOD ACTION: Delegating and budget review",
            "action": {
                "metric_changes": {"career.workload": -15.0, "finances.liquidity": 10.0, "mental_wellbeing.stress_level": -5.0},
                "resource_cost": {"time": 4.0, "money": 100.0, "energy": 20.0},
                "actions_taken": 2
            }
        },
        {
            "name": "MEDIUM ACTION: Small self-care rest",
            "action": {
                "metric_changes": {"physical_health.sleep_quality": 6.0, "mental_wellbeing.clarity": 3.0},
                "resource_cost": {"time": 2.0, "energy": -20.0}, # Rest recovers energy
                "actions_taken": 1
            }
        },
        {
            "name": "INACTION: Let the cascade run",
            "action": {
                "metric_changes": {},
                "resource_cost": {},
                "actions_taken": 0
            }
        }
    ]
    
    for sce in scenarios:
        print(f"\nTaking Action: {sce['name']}...")
        action_obj = LifeStackAction(**sce['action'])
        obs = env.step(action_obj)
        env_render_compact(env, obs)
        total_reward += (obs.reward or 0.0)

    # 3. Final Summary
    final_flat = env.state.current_metrics.flatten()
    critical = [k for k, v in final_flat.items() if v < 20]
    
    print("\n" + "█"*60)
    print("EPISODE SUMMARY")
    print(f"Steps Taken      : {env.state.step_count}")
    print(f"Total Cumulative Reward : {total_reward:.4f}")
    if critical:
        print(f"Critical Floor Violations: {', '.join(critical)}")
    else:
        print("Critical Violations: NONE")
    print("█"*60)

if __name__ == "__main__":
    main()
