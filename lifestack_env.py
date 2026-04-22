import copy
from life_state import LifeMetrics, ResourceBudget, DependencyGraph
from metric_schema import normalize_metric_path
from reward import compute_reward

try:
    from openenv.env import Env
except ImportError:
    try:
        from openenv.core import Environment as Env
    except ImportError:
        class Env:
            """Fallback shim so the local environment still runs without openenv installed."""

            def __init__(self, *args, **kwargs):
                pass

class LifeStackEnv(Env):
    def __init__(self):
        super().__init__()
        
        self.observation_space = {
            'metrics': {'type': 'Box', 'low': 0.0, 'high': 100.0, 'shape': (23,)},
            'resources': {'type': 'Box', 'low': 0.0, 'high': 500.0, 'shape': (3,)},
            'step': {'type': 'Discrete', 'n': 6}
        }
        self.action_space = {
            'action_type': {'type': 'Discrete', 'n': 7},
            'target_domain': {'type': 'Discrete', 'n': 6},
            'metric_changes': {'type': 'Box', 'low': -50.0, 'high': 50.0, 'shape': (23,)},
            'resource_cost': {'type': 'Box', 'low': 0.0, 'high': 100.0, 'shape': (3,)}
        }
        self.metadata = {
            'name': 'LifeStack-v1',
            'version': '1.0.0',
            'description': 'Multi-domain life conflict resolution environment',
            'reward_range': (-1.0, 1.0),
            'max_episode_steps': 5
        }
        
        self.graph = DependencyGraph()
        self.max_steps = 5
        self._state = None
        self.budget = None
        self.step_count = 0
        self.last_reward = None
        self.last_breakdown = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def seed(self, s: int):
        import random
        random.seed(s)

    def observation_to_vector(self, obs: dict) -> list:
        metrics = list(obs['metrics'].values())
        resources = [obs['resources']['time'], obs['resources']['money'], obs['resources']['energy']]
        step = [obs['step']]
        import numpy as np
        return np.array(metrics + resources + step).tolist()

    def reset(self, seed: int = None, episode_id: str = None, conflict: dict = None, budget: dict = None, **kwargs) -> dict:
        """Resets the environment to initial state (70s) or apply a conflict."""
        self.state = LifeMetrics()  # All metrics at 70
        if budget:
            self.budget = ResourceBudget(
                time_hours=budget.get("time", 20.0),
                money_dollars=budget.get("money", 500.0),
                energy_units=budget.get("energy", 100.0)
            )
        else:
            self.budget = ResourceBudget(time_hours=20.0, money_dollars=500.0, energy_units=100.0)
        self.step_count = 0
        self.last_reward = None
        self.last_breakdown = None

        if conflict:
            # Apply initial disruption via cascade
            self.state = self.graph.cascade(self.state, conflict)

        return self._get_obs(), {}

    def _get_obs(self, done: bool = False) -> dict:
        return {
            "metrics": self.state.flatten(),
            "resources": {
                "time": self.budget.time_hours,
                "money": self.budget.money_dollars,
                "energy": self.budget.energy_units
            },
            "step": self.step_count,
            "done": done
        }

    def _update_metric(self, path: str, delta: float):
        """Manually update a metric without a full cascade."""
        path = normalize_metric_path(path)
        if '.' not in path:
            return
        domain_name, sub_name = path.split('.', 1)
        domain = getattr(self.state, domain_name, None)
        if domain is None or not hasattr(domain, sub_name):
            return
        current = getattr(domain, sub_name)
        setattr(domain, sub_name, max(0.0, min(100.0, current + delta)))

    def step(self, action: dict, timeout_s: float = None, **kwargs) -> dict:
        """
        Executes one step in the environment.
        
        action: {
            'metric_changes': {path: delta},
            'resource_cost': {'time': T, 'money': M, 'energy': E},
            'actions_taken': int
        }
        """
        state_before = copy.deepcopy(self.state)
        metric_changes = action.get('metric_changes', {})
        resource_cost = action.get('resource_cost', {})
        actions_taken = action.get('actions_taken', 0)
        
        info = []

        # 1. Separate changes into significant (trigger cascade) and insignificant
        # Skip malformed keys (LLM sometimes omits the domain prefix)
        sig_changes = {}
        for path, delta in metric_changes.items():
            path = normalize_metric_path(path)
            if '.' not in path:
                continue
            if abs(delta) > 5:
                sig_changes[path] = delta
            else:
                self._update_metric(path, delta)
        
        # 2. Run cascade on significant changes
        if sig_changes:
            self.state = self.graph.cascade(self.state, sig_changes)

        # 3. Handle resource deduction
        deduct_success = self.budget.deduct(
            time=resource_cost.get('time', 0.0),
            money=resource_cost.get('money', 0.0),
            energy=resource_cost.get('energy', 0.0)
        )
        
        budget_penalty = 0.0
        if not deduct_success:
            budget_penalty = -0.20
            info.append("INSUFFICIENT_RESOURCES")

        # 4. Calculate reward
        reward, breakdown = compute_reward(state_before, self.state, resource_cost, actions_taken)
        reward += budget_penalty
        
        self.last_reward = reward
        self.last_breakdown = breakdown
        self.step_count += 1

        # 5. Ending conditions
        metrics_flat = self.state.flatten()
        any_hit_zero = any(v <= 0.0 for v in metrics_flat.values())
        resources_dead = (self.budget.time_hours <= 0 and 
                          self.budget.money_dollars <= 0 and 
                          self.budget.energy_units <= 0)
        
        terminated = any_hit_zero or resources_dead
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated
        
        obs = self._get_obs(done)
        env_info = {
            "breakdown": breakdown,
            "info_msgs": info
        }
        return obs, reward, terminated, truncated, env_info

    def render(self):
        """Vibrant status report of the current LifeMetrics state."""
        print("\n" + "═"*60)
        print(f"STEP: {self.step_count} | ⏳ TIME: {self.budget.time_hours:.1f}h | 💵 MONEY: ${self.budget.money_dollars:.1f} | ⚡ ENERGY: {self.budget.energy_units:.1f}")
        
        flat = self.state.flatten()
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
                if short in inverted:
                    icon = "🔴" if val > 70 else ("🟡" if val >= 40 else "🟢")
                else:
                    icon = "🟢" if val > 70 else ("🟡" if val >= 40 else "🔴")
                print(f"  {icon} {short:20} : {val:5.2f}")

        if self.last_reward is not None:
            print(f"\nLAST STEP REWARD: {self.last_reward:.4f}")
            if self.last_breakdown and self.last_breakdown['penalties_fired']:
                print(f"CRITICAL WARNINGS: {', '.join(self.last_breakdown['penalties_fired'])}")
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
        obs, reward, terminated, truncated, _ = env.step(sce['action'])
        total_reward += reward
        env.render()
        
    # 3. Final Summary
    final_flat = env.state.flatten()
    critical = [k for k, v in final_flat.items() if v < 20]
    
    print("\n" + "█"*60)
    print("EPISODE SUMMARY")
    print(f"Steps Taken      : {env.step_count}")
    print(f"Total Cumulative Reward : {total_reward:.4f}")
    if critical:
        print(f"Critical Floor Violations: {', '.join(critical)}")
    else:
        print("Critical Violations: NONE")
    print("█"*60)

if __name__ == "__main__":
    main()
