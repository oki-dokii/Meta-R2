import copy
from life_state import LifeMetrics, ResourceBudget, DependencyGraph
from reward import compute_reward

class LifeStackEnv:
    def __init__(self):
        self.graph = DependencyGraph()
        self.max_steps = 5
        self.state = None
        self.budget = None
        self.step_count = 0
        self.last_reward = None
        self.last_breakdown = None

    def reset(self, conflict: dict = None) -> dict:
        """Resets the environment to initial state (70s) or apply a conflict."""
        self.state = LifeMetrics()  # All metrics at 70
        self.budget = ResourceBudget(time_hours=20.0, money_dollars=500.0, energy_units=100.0)
        self.step_count = 0
        self.last_reward = None
        self.last_breakdown = None

        if conflict:
            # Apply initial disruption via cascade
            self.state = self.graph.cascade(self.state, conflict)

        return self._get_obs()

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
        if '.' not in path:
            return
        domain_name, sub_name = path.split('.', 1)
        domain = getattr(self.state, domain_name, None)
        if domain is None or not hasattr(domain, sub_name):
            return
        current = getattr(domain, sub_name)
        setattr(domain, sub_name, max(0.0, min(100.0, current + delta)))

    def step(self, action: dict) -> dict:
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
        
        done = (self.step_count >= self.max_steps or any_hit_zero or resources_dead)
        
        obs = self._get_obs(done)
        obs.update({
            "reward": reward,
            "breakdown": breakdown,
            "info": info
        })
        return obs

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
            for name, val in submetrics.items():
                short = name.split('.')[1]
                # Indicators: 🟢 > 70, 🟡 40-70, 🔴 < 40
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
    env.reset(conflict)
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
        obs = env.step(sce['action'])
        total_reward += obs['reward']
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
