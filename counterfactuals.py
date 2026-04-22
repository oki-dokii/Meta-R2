"""
counterfactuals.py — Generates alternative "What If" scenarios for LifeStack agent decisions.
"""

import copy
import random
from reward import compute_reward
from life_state import DependencyGraph

def generate_counterfactuals(agent, metrics, budget, conflict, person, chosen_action):
    """
    Simulates 3 alternative action types and compares them to the agent's choice.
    Returns a list of dicts with alternative outcomes.
    """
    action_types = ["communicate", "rest", "delegate", "negotiate", "spend", "reschedule", "deprioritize"]
    chosen_type = chosen_action.primary.action_type
    
    # Filter and pick 3 different types
    alternatives = [t for t in action_types if t != chosen_type]
    random.shuffle(alternatives)
    target_types = alternatives[:3]
    
    results = []
    graph = DependencyGraph()
    
    for action_type in target_types:
        try:
            # 1. Generate alternative action
            # We use the special forced-type method we added to the agent
            alt_action = agent.get_action_for_type(metrics, budget, conflict, person, action_type)
            
            # 2. Simulate applying it
            current_stress = metrics.mental_wellbeing.stress_level
            uptake = person.respond_to_action(
                alt_action.primary.action_type, 
                alt_action.primary.resource_cost, 
                current_stress
            )
            
            state_after = copy.deepcopy(metrics)
            for path, delta in alt_action.primary.metric_changes.items():
                if "." not in path: continue
                try:
                    scaled_delta = float(delta) * uptake
                except (ValueError, TypeError):
                    continue
                    
                if abs(scaled_delta) > 5:
                    state_after = graph.cascade(state_after, {path: scaled_delta})
                else:
                    dom, sub = path.split('.')
                    d = getattr(state_after, dom, None)
                    if d:
                        cur = getattr(d, sub, 70.0)
                        setattr(d, sub, max(0.0, min(100.0, cur + scaled_delta)))
            
            # 3. Compute Reward
            reward, breakdown = compute_reward(metrics, state_after, alt_action.primary.resource_cost, 1)
            
            # 4. Analysis deltas
            flat_before = metrics.flatten()
            flat_after = state_after.flatten()
            deltas = {k: flat_after[k] - flat_before[k] for k in flat_after}
            
            # Filter for meaningful changes (>1.0)
            significant = {k: v for k, v in deltas.items() if abs(v) > 1.0}
            
            trade_off = ""
            if significant:
                best = max(significant.items(), key=lambda x: x[1])
                worst = min(significant.items(), key=lambda x: x[1])
                
                b_name = best[0].split('.')[-1].replace('_', ' ')
                if best[1] > 2:
                    trade_off = f"Better {b_name} (+{best[1]:.0f})"
                else:
                    trade_off = f"Stability in {b_name}"
                    
                if worst[1] < -2:
                    w_name = worst[0].split('.')[-1].replace('_', ' ')
                    trade_off += f" but drops {w_name} ({worst[1]:.0f})"
                else:
                    trade_off += " but mission impact is lower than optimal."
            else:
                trade_off = "Minimal impact on core life metrics."

            # Incorporate resource commentary
            cost = alt_action.primary.resource_cost
            if cost.get('money', 0) > 100:
                trade_off += f" (${cost['money']:.0f} cost)"
            elif cost.get('time', 0) > 4:
                trade_off += f" ({cost['time']:.1f}h time drain)"

            results.append({
                "action_type": action_type,
                "description": alt_action.primary.description,
                "reward": reward,
                "trade_off": trade_off,
                "uptake": uptake
            })
            
        except Exception as e:
            print(f"Error in counterfactual generation for {action_type}: {e}")
            
    return results
