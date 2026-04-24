"""
lifestack_gym_env.py — Gymnasium-compatible wrapper for LifeStack

Exposes the LifeStack environment as a standard gym.Env with:
- observation_space: Box(0, 100, shape=(26,)) — 23 sub-metrics + 3 resources
- action_space: Discrete(7) — 7 action types mapped to template actions
- Standard reset() / step() / render() API
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random, copy
from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
from core.metric_schema import normalize_metric_path
from core.reward import compute_reward, compute_task_reward
from agent.conflict_generator import generate_conflict, ConflictEvent
from intake.simperson import SimPerson


# Map discrete action IDs to action types
ACTION_TYPE_MAP = {
    0: "negotiate",
    1: "communicate",
    2: "delegate",
    3: "spend",
    4: "reschedule",
    5: "rest",
    6: "execute",
}


class LifeStackGymEnv(gym.Env):
    """
    LifeStack as a Gymnasium environment.
    
    Observation: 26-dim vector (23 life sub-metrics + 3 resource values)
    Action: Discrete(7) — one of 7 action types
    Reward: float in [-1, 1]
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task=None, difficulty: int = None, render_mode: str = None, max_steps: int = 30):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)
        self.render_mode = render_mode
        self.task = task
        self.difficulty = difficulty
        self.max_steps = max_steps
        
        from core.lifestack_env import LifeStackEnv
        self.env = LifeStackEnv()
        self._metric_keys = list(LifeMetrics().flatten().keys())

    def _obs_vector(self) -> np.ndarray:
        flat = self.env.state.current_metrics.flatten()
        metric_vals = [flat[k] for k in self._metric_keys]
        budget = self.env.state.budget
        resource_vals = [
            budget.time_hours,
            budget.money_dollars,
            budget.energy_units,
        ]
        return np.array(metric_vals + resource_vals, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        conflict = None
        if self.task is None:
            from agent.conflict_generator import generate_conflict
            conflict = generate_conflict(self.difficulty)
            
        obs_obj = self.env.reset(task=self.task, conflict=conflict)
        return self._obs_vector(), obs_obj.metadata

    def step(self, action: int):
        from core.lifestack_env import LifeStackAction
        action_type = ACTION_TYPE_MAP[action]
        
        # Build logical action from template
        metric_changes, resource_cost = self._action_to_changes(action_type)
        
        # In this wrapper, we pick a reasonable target if needed
        target = ""
        if action_type == "execute" and self.env.task:
            # Pick first available route
            for r in self.env.task.viable_routes:
                if r.id not in self.env.closed_route_ids:
                    target = r.id
                    break
        
        ls_action = LifeStackAction(
            action_type=action_type,
            target=target,
            reasoning=f"Agent chose {action_type} for discrete action {action}.",
            metric_changes=metric_changes,
            resource_cost=resource_cost,
            actions_taken=1
        )
        
        obs_obj = self.env.step(ls_action)
        
        terminated = obs_obj.done
        # Truncated only if not naturally terminated
        truncated = (not terminated) and (self.env.state.step_count >= (self.task.horizon if self.task else self.max_steps))
        
        return self._obs_vector(), obs_obj.reward, terminated, truncated, {"breakdown": obs_obj.metadata.get("breakdown", {})}

    def _action_to_changes(self, action_type: str):
        """Maps an action type string to (metric_changes, resource_cost)."""
        templates = {
            "negotiate": (
                {"career.workload": -15.0, "mental_wellbeing.stress_level": -5.0},
                {"time": 1.5, "energy": 20.0},
            ),
            "communicate": (
                {"relationships.romantic": 10.0, "mental_wellbeing.stress_level": -5.0},
                {"time": 0.5, "energy": 10.0},
            ),
            "delegate": (
                {"career.workload": -10.0, "relationships.professional_network": -5.0},
                {"time": 1.0, "energy": 15.0},
            ),
            "spend": (
                {"finances.liquidity": -20.0, "mental_wellbeing.stress_level": -10.0},
                {"time": 1.0, "energy": 15.0},
            ),
            "reschedule": (
                {"career.workload": -10.0, "time.free_hours_per_week": 5.0},
                {"time": 2.0, "energy": 15.0},
            ),
            "rest": (
                {"mental_wellbeing.stress_level": -12.0, "physical_health.energy": 10.0},
                {"time": 1.0},
            ),
            "execute": (
                {}, # executes a route target
                {"time": 1.0, "energy": 10.0},
            ),
        }
        return templates.get(action_type, ({}, {}))

    def render(self):
        if self.render_mode == "human":
            # Delegate to the internal env's render
            self.env.render()


# ── Quick smoke test ──
if __name__ == "__main__":
    env = LifeStackGymEnv(difficulty=3, render_mode="human")
    obs, info = env.reset()
    print(f"Conflict: {info['conflict_title']} | Person: {info['person']}")
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    env.render()

    total = 0.0
    done = False
    while not done:
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        total += rew
        done = term or trunc
        print(f"  Action {act} → reward {rew:.3f}")

    env.render()
    print(f"\nTotal reward: {total:.3f}")
