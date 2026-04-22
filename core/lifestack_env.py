import copy
from typing import Any, Optional, Dict
from pydantic import Field

from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
from core.metric_schema import normalize_metric_path
from core.reward import compute_reward

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
        # Final fallback to nominal shim
        class Environment:
            def __init__(self, rubric=None): self.rubric = rubric
            def reset(self, *a, **k): pass
            def step(self, *a, **k): pass
        class Action: pass
        class Observation: pass
        class State: pass
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

class LifeStackRubric(Rubric):
    """Standard reward rubric for LifeStack."""
    def forward(self, action: LifeStackAction, observation: LifeStackObservation) -> float:
        # In LifeStack, reward is usually computed inside step() for state-transition access.
        # This rubric provides a hook for external reward evaluation if needed.
        return observation.reward if observation.reward is not None else 0.0

_EnvBase = Environment[LifeStackAction, LifeStackObservation, LifeStackState] if USING_MODERN_API else Environment

class LifeStackEnv(_EnvBase):
    """
    LifeStack Environment v1.1 — Refactored for OpenEnv 0.2.3 compliance.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        if USING_MODERN_API:
            super().__init__(rubric=LifeStackRubric())
        else:
            super().__init__()
        
        self.metadata_internal = {
            'name': 'LifeStack-v1',
            'version': '1.1.0',
            'description': 'Premium multi-domain life conflict resolution simulation',
            'max_episode_steps': 5
        }
        
        self.graph = DependencyGraph()
        self.max_steps = 5
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
              conflict: Optional[dict] = None, budget: Optional[dict] = None, **kwargs) -> LifeStackObservation:
        """Resets the environment. Seed and conflict can be provided."""
        if USING_MODERN_API and getattr(self, 'rubric', None):
            self.rubric.reset()
        
        if seed is not None:
            import random
            random.seed(seed)

        # Reset state
        self._internal_state.episode_id = episode_id
        self._internal_state.step_count = 0
        self._internal_state.current_metrics = LifeMetrics()
        
        if budget:
            self._internal_state.budget = ResourceBudget(
                time_hours=budget.get("time", 20.0),
                money_dollars=budget.get("money", 500.0),
                energy_units=budget.get("energy", 100.0)
            )
        else:
            self._internal_state.budget = ResourceBudget(time_hours=20.0, money_dollars=500.0, energy_units=100.0)

        if conflict:
            # Apply initial disruption via cascade
            self._internal_state.current_metrics = self.graph.cascade(self._internal_state.current_metrics, conflict)

        return self._get_obs()

    def _get_obs(self, done: bool = False, reward: Optional[float] = None) -> LifeStackObservation:
        return LifeStackObservation(
            metrics=self._internal_state.current_metrics.flatten(),
            resources={
                "time": self._internal_state.budget.time_hours,
                "money": self._internal_state.budget.money_dollars,
                "energy": self._internal_state.budget.energy_units
            },
            step=self._internal_state.step_count,
            done=done,
            reward=reward
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
        """Executes one step in the environment using LifeStackAction."""
        if isinstance(action, dict): # Backward compatibility for old calls
            action = LifeStackAction(**action)

        state_before = copy.deepcopy(self._internal_state.current_metrics)
        metric_changes = action.metric_changes
        resource_cost = action.resource_cost
        
        info_msgs = []

        # 1. Cascade logic
        sig_changes = {}
        for path, delta in metric_changes.items():
            path = normalize_metric_path(path)
            if abs(delta) > 5:
                sig_changes[path] = delta
            else:
                self._update_metric(path, delta)
        
        if sig_changes:
            self._internal_state.current_metrics = self.graph.cascade(self._internal_state.current_metrics, sig_changes)

        # 2. Resource deduction
        deduct_success = self._internal_state.budget.deduct(
            time=resource_cost.get('time', 0.0),
            money=resource_cost.get('money', 0.0),
            energy=resource_cost.get('energy', 0.0)
        )
        
        budget_penalty = 0.0
        if not deduct_success:
            budget_penalty = -0.20
            info_msgs.append("INSUFFICIENT_RESOURCES")

        # 3. Reward calculation
        reward, breakdown = compute_reward(state_before, self._internal_state.current_metrics, resource_cost, action.actions_taken)
        reward += budget_penalty
        
        self._internal_state.step_count += 1

        # 4. End conditions
        metrics_flat = self._internal_state.current_metrics.flatten()
        any_hit_zero = any(v <= 0.0 for v in metrics_flat.values())
        resources_dead = (self._internal_state.budget.time_hours <= 0 and 
                          self._internal_state.budget.money_dollars <= 0 and 
                          self._internal_state.budget.energy_units <= 0)
        
        terminated = any_hit_zero or resources_dead
        truncated = self._internal_state.step_count >= self.max_steps
        done = terminated or truncated
        
        observation = self._get_obs(done, reward)
        observation.metadata["breakdown"] = breakdown
        observation.metadata["info"] = info_msgs
        
        return observation

    def render(self):
        """Vibrant status report of the current LifeMetrics state."""
        print("\n" + "═"*60)
        print(f"STEP: {self._internal_state.step_count} | ⏳ TIME: {self._internal_state.budget.time_hours:.1f}h | 💵 MONEY: ${self._internal_state.budget.money_dollars:.1f} | ⚡ ENERGY: {self._internal_state.budget.energy_units:.1f}")
        
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
        print("═"*60)


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

def env_render_compact(env, obs):
    """Compact printer for testing."""
    print(f"STEP: {obs.step} | REWARD: {obs.reward:.3f} | DONE: {obs.done}")
    if obs.metadata.get("breakdown", {}).get("penalties_fired"):
        print(f"  ⚠️ PENALTIES: {obs.metadata['breakdown']['penalties_fired']}")

        
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
