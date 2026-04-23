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
from core.reward import compute_reward
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
    6: "deprioritize",
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
        self.difficulty = difficulty
        self.render_mode = render_mode

        # 23 sub-metrics + 3 resources = 26
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        # Internal state
        self.graph = DependencyGraph()
        self.state: LifeMetrics = None
        self.budget: ResourceBudget = None
        self.conflict: ConflictEvent = None
        self.person: SimPerson = None
        self.step_count = 0
        self.max_steps = getattr(task, 'horizon', max_steps) if task else max_steps
        self.last_reward = None
        self.last_breakdown = None

        # Precompute sorted metric keys for consistent ordering
        self._metric_keys = list(LifeMetrics().flatten().keys())

    def _obs_vector(self) -> np.ndarray:
        flat = self.state.flatten()
        metric_vals = [flat[k] for k in self._metric_keys]
        resource_vals = [
            self.budget.time_hours,
            self.budget.money_dollars,
            self.budget.energy_units,
        ]
        return np.array(metric_vals + resource_vals, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random person
        person_pool = [
            SimPerson(name="Alex", openness=0.4, conscientiousness=0.9,
                      extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
            SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2,
                      extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
            SimPerson(name="Sam", openness=0.5, conscientiousness=0.6,
                      extraversion=0.1, agreeableness=0.65, neuroticism=0.9),
            SimPerson(name="Maya", openness=0.5, conscientiousness=0.7,
                      extraversion=0.5, agreeableness=0.95, neuroticism=0.3),
            SimPerson(name="Leo", openness=0.85, conscientiousness=0.8,
                      extraversion=0.4, agreeableness=0.4, neuroticism=0.55),
        ]
        self.person = self.np_random.choice(person_pool) if seed else random.choice(person_pool)

        # Generate conflict
        self.conflict = generate_conflict(self.difficulty)

        # Fresh state + budget
        self.state = LifeMetrics()
        self.budget = ResourceBudget(
            time_hours=self.conflict.resource_budget.get("time", 20.0),
            money_dollars=self.conflict.resource_budget.get("money", 500.0),
            energy_units=self.conflict.resource_budget.get("energy", 100.0),
        )
        self.step_count = 0
        self.last_reward = None
        self.last_breakdown = None

        # Apply initial disruption
        self.state = self.graph.cascade(self.state, self.conflict.primary_disruption)

        obs = self._obs_vector()
        info = {
            "conflict_title": self.conflict.title,
            "conflict_story": self.conflict.story,
            "person": self.person.name,
            "difficulty": self.conflict.difficulty,
        }
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        state_before = copy.deepcopy(self.state)
        action_type = ACTION_TYPE_MAP[action]

        # Build a template action based on action_type
        # Find a matching example action, or generate a default
        metric_changes, resource_cost = self._action_to_changes(action_type)

        # Apply personality uptake scaling
        stress = self.state.mental_wellbeing.stress_level
        uptake = self.person.respond_to_action(action_type, resource_cost, stress)
        scaled_changes = {k: v * uptake for k, v in metric_changes.items()}

        # Apply significant changes via cascade
        for path, delta in scaled_changes.items():
            path = normalize_metric_path(path)
            if '.' not in path:
                continue
            if abs(delta) > 5:
                self.state = self.graph.cascade(self.state, {path: delta})
            else:
                dom, sub = path.split('.', 1)
                d = getattr(self.state, dom, None)
                if d and hasattr(d, sub):
                    cur = getattr(d, sub)
                    setattr(d, sub, max(0.0, min(100.0, cur + delta)))

        # Deduct resources
        self.budget.deduct(
            time=resource_cost.get("time", 0.0),
            money=resource_cost.get("money", 0.0),
            energy=resource_cost.get("energy", 0.0),
        )

        # Reward
        reward, breakdown = compute_reward(
            state_before, self.state, resource_cost, actions_taken=1
        )
        self.last_reward = reward
        self.last_breakdown = breakdown
        self.step_count += 1

        # Termination
        flat = self.state.flatten()
        any_zero = any(v <= 0.0 for v in flat.values())
        no_resources = (
            self.budget.time_hours <= 0
            and self.budget.money_dollars <= 0
            and self.budget.energy_units <= 0
        )
        terminated = any_zero or no_resources
        truncated = self.step_count >= self.max_steps

        obs = self._obs_vector()
        info = {
            "breakdown": breakdown,
            "step": self.step_count,
        }
        return obs, reward, terminated, truncated, info

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
            "deprioritize": (
                {"time.free_hours_per_week": 8.0, "relationships.social": -10.0},
                {"time": 0.5, "energy": 5.0},
            ),
        }
        return templates.get(action_type, ({}, {}))

    def render(self):
        if self.render_mode == "human":
            # Reuse the rich render from the original env
            from core.lifestack_env import LifeStackEnv as _Orig
            tmp = _Orig()
            tmp.state = self.state
            tmp.budget = self.budget
            tmp.step_count = self.step_count
            tmp.last_reward = self.last_reward
            tmp.last_breakdown = self.last_breakdown
            tmp.render()


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
