"""
counterfactuals.py — Generates alternative "What If" scenarios for LifeStack agent decisions.
"""

import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.reward import compute_reward
from core.life_state import DependencyGraph

# Domain-aware action preferences: for a given conflict domain, these are the
# semantically valid alternatives (in priority order). Prevents DELEGATE showing
# up for relationship conflicts, REST showing up for financial ones, etc.
_DOMAIN_ACTIONS = {
    "relationships":    ["communicate", "spend", "reschedule", "negotiate"],
    "career":           ["negotiate", "delegate", "deprioritize", "reschedule"],
    "finances":         ["spend", "negotiate", "deprioritize", "reschedule"],
    "mental_wellbeing": ["rest", "communicate", "deprioritize", "reschedule"],
    "physical_health":  ["rest", "spend", "reschedule", "communicate"],
    "time":             ["deprioritize", "delegate", "reschedule", "negotiate"],
}
_ALL_ACTIONS = ["communicate", "rest", "delegate", "negotiate", "spend", "reschedule", "deprioritize"]


def generate_counterfactuals(agent, metrics, budget, conflict, person, chosen_action):
    """
    Simulates 3 domain-relevant alternative actions in parallel.
    Returns a list of dicts with alternative outcomes, sorted by reward descending.
    """
    chosen_type = chosen_action.primary.action_type

    # Infer the primary conflict domain from the disruption keys
    disruption = getattr(conflict, 'primary_disruption', {}) or {}
    domain_key = ""
    if disruption:
        # Find the domain with the largest absolute disruption magnitude
        domain_totals = {}
        for path, val in disruption.items():
            dom = path.split('.')[0] if '.' in path else path
            domain_totals[dom] = domain_totals.get(dom, 0) + abs(val)
        domain_key = max(domain_totals, key=domain_totals.get) if domain_totals else ""
    preferred = _DOMAIN_ACTIONS.get(domain_key, _ALL_ACTIONS)
    alternatives = [t for t in preferred if t != chosen_type]
    for t in _ALL_ACTIONS:
        if t not in alternatives and t != chosen_type:
            alternatives.append(t)
    target_types = alternatives[:3]

    graph = DependencyGraph()

    def _run_one(action_type):
        try:
            alt_action = agent.get_action_for_type(metrics, budget, conflict, person, action_type)

            current_stress = metrics.mental_wellbeing.stress_level
            uptake = person.respond_to_action(
                alt_action.primary.action_type,
                alt_action.primary.resource_cost,
                current_stress
            )

            state_after = copy.deepcopy(metrics)
            total_abs_delta = 0.0
            for path, delta in alt_action.primary.metric_changes.items():
                if "." not in path:
                    continue
                try:
                    scaled_delta = float(delta) * uptake
                except (ValueError, TypeError):
                    continue

                total_abs_delta += abs(scaled_delta)
                if abs(scaled_delta) > 5:
                    state_after = graph.cascade(state_after, {path: scaled_delta})
                else:
                    dom, sub = path.split('.')
                    d = getattr(state_after, dom, None)
                    if d:
                        cur = getattr(d, sub, 70.0)
                        setattr(d, sub, max(0.0, min(100.0, cur + scaled_delta)))

            reward, _ = compute_reward(metrics, state_after, alt_action.primary.resource_cost, 1)

            # Penalise near-zero impact: an alternative that changes nothing shouldn't
            # score ~0.52 from resource_efficiency + cascade_containment alone.
            if total_abs_delta < 3.0:
                reward = max(-0.5, reward - 0.30)

            flat_before = metrics.flatten()
            flat_after = state_after.flatten()
            deltas = {k: flat_after[k] - flat_before[k] for k in flat_after}
            significant = {k: v for k, v in deltas.items() if abs(v) > 1.0}

            trade_off = ""
            if significant:
                best = max(significant.items(), key=lambda x: x[1])
                worst = min(significant.items(), key=lambda x: x[1])
                b_name = best[0].split('.')[-1].replace('_', ' ')
                trade_off = f"Better {b_name} (+{best[1]:.0f})" if best[1] > 2 else f"Stability in {b_name}"
                if worst[1] < -2:
                    w_name = worst[0].split('.')[-1].replace('_', ' ')
                    trade_off += f" but drops {w_name} ({worst[1]:.0f})"
                else:
                    trade_off += " but mission impact is lower than optimal."
            else:
                trade_off = "Minimal impact on core life metrics."

            cost = alt_action.primary.resource_cost
            if cost.get('money', 0) > 100:
                trade_off += f" (${cost['money']:.0f} cost)"
            elif cost.get('time', 0) > 4:
                trade_off += f" ({cost['time']:.1f}h time drain)"

            return {
                "action_type": action_type,
                "description": alt_action.primary.description,
                "reward": reward,
                "trade_off": trade_off,
                "uptake": uptake,
                "metrics": state_after.flatten(),
            }

        except Exception as e:
            print(f"Error in counterfactual for {action_type}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_run_one, t): t for t in target_types}
        for future in as_completed(futures, timeout=65):
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda x: x["reward"], reverse=True)
    return results
