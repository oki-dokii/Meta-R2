import math
import copy
import json
import re
from core.life_state import LifeMetrics
from core.task import Task

def compute_reward(
    state_before: LifeMetrics, 
    state_after: LifeMetrics, 
    resources_used: dict, 
    actions_taken: int,
    metric_changes: dict = None,
    completion: str = None,
    disruption_baseline: int = None
) -> tuple[float, dict]:
    """
    Computes the reward for a life step based on changes in LifeMetrics and resource usage.
    
    Args:
        state_before: The state at the start of the step.
        state_after: The state after actions and cascades.
        resources_used: Dict with keys 'time', 'money', 'energy'.
        actions_taken: Integer count of intentional actions performed.
        disruption_baseline: Expected number of metrics affected by an action.
        
    Returns:
        tuple[float, dict]: (final_reward, breakdown_dict)
    """
    before_flat = state_before.flatten()
    after_flat = state_after.flatten()
    
    # 1. OUTCOME SCORE (Weighted average of positive deltas)
    domain_weights = {
        "career": 1/6,
        "finances": 1/6,
        "relationships": 1/6,
        "physical_health": 1/6,
        "mental_wellbeing": 1/6,
        "time": 1/6
    }
    
    # Map sub-metrics to their domains
    submetrics_per_domain = {}
    for k in before_flat.keys():
        domain = k.split('.')[0]
        submetrics_per_domain[domain] = submetrics_per_domain.get(domain, 0) + 1
    
    outcome_score = 0.0
    for k in before_flat.keys():
        domain = k.split('.')[0]
        delta = after_flat[k] - before_flat[k]
        if delta > 0:
            # Each domain is 1/6. Each sub-metric within a domain gets its equal share of that 1/6.
            # Normalize delta by 100 (max possible increase is 100).
            weight = domain_weights[domain] / submetrics_per_domain[domain]
            outcome_score += (delta / 100.0) * weight
            
    # 2. CASCADE CONTAINMENT SCORE
    worsened_count = sum(1 for k in before_flat.keys() if after_flat[k] < before_flat[k])
    total_metrics = len(before_flat)
    cascade_containment_score = 1.0 - (worsened_count / total_metrics)
    
    # 3. RESOURCE EFFICIENCY SCORE
    # Available: time 20, money 500, energy 100
    m_time = resources_used.get('time', 0.0) / 20.0
    m_money = resources_used.get('money', 0.0) / 500.0
    m_energy = resources_used.get('energy', 0.0) / 100.0
    
    # Normalize by total slots (3 resources)
    resource_efficiency_score = 1.0 - ((m_time + m_money + m_energy) / 3.0)
    resource_efficiency_score = max(0.0, min(1.0, resource_efficiency_score))
    
    # 4. RELATIONSHIP PRESERVATION SCORE (Sigmoid applied to average delta)
    rel_keys = [k for k in before_flat.keys() if k.startswith('relationships.')]
    avg_rel_before = sum(before_flat[k] for k in rel_keys) / len(rel_keys)
    avg_rel_after = sum(after_flat[k] for k in rel_keys) / len(rel_keys)
    delta_rel = avg_rel_after - avg_rel_before
    
    # score = 1 / (1 + exp(-delta/10))
    relationship_preservation_score = 1.0 / (1.0 + math.exp(-delta_rel / 10.0))
    
    # FINAL REWARD FORMULA
    base_reward = (
        (0.40 * outcome_score) + 
        (0.25 * cascade_containment_score) + 
        (0.20 * resource_efficiency_score) + 
        (0.15 * relationship_preservation_score)
    )
    
    # PENALTIES
    penalties = 0.0
    fired = []
    
    # -0.50 if ANY metric is below 20 after the step
    if any(v < 20 for v in after_flat.values()):
        penalties -= 0.50
        fired.append("CRITICAL_FLOOR_VIOLATION")
        
    # -0.30 if cascade spread wider than the number of metrics the agent directly changed
    # Scaled baseline from task metadata preferred over hardcoded default
    if disruption_baseline is None:
        disruption_baseline = len(metric_changes) if metric_changes else 2
        
    if worsened_count > disruption_baseline:
        penalties -= 0.30
        fired.append("CASCADE_SPREAD_WIDER")
        
    # -0.40 if actions_taken == 0
    if actions_taken == 0:
        penalties -= 0.40
        fired.append("INACTION_PENALTY")
        
    # -0.15 if relationships domain average dropped more than 20 points
    if delta_rel < -20:
        penalties -= 0.15
        fired.append("RELATIONSHIP_COLLAPSE")

    # [NEW] Plausibility Penalty
    if metric_changes:
        plaus = reward_plausibility_check(metric_changes, resources_used)
        if plaus < 0:
            penalties += plaus
            fired.append("PLAUSIBILITY_VIOLATION")

    # [NEW] Format Compliance (Only if raw completion provided)
    comp_reward = 0.0
    if completion:
        comp_reward = reward_format_compliance(completion)
        # Note: reward_format_compliance returns high magnitude values, 
        # we can cap it or use it as-is.
        
    final_reward = max(-1.0, min(1.0, base_reward + penalties))
    
    breakdown = {
        "components": {
            "outcome": outcome_score,
            "containment": cascade_containment_score,
            "efficiency": resource_efficiency_score,
            "preservation": relationship_preservation_score,
            "format_compliance": comp_reward,
            "plausibility": reward_plausibility_check(metric_changes, resources_used) if metric_changes else 0.0
        },
        "base_reward": base_reward,
        "penalties_total": penalties,
        "penalties_fired": fired,
        "metrics_worsened": worsened_count,
        "rel_delta": delta_rel
    }
    
    return final_reward, breakdown

def compute_milestone_reward(milestones_achieved: list[str], task: Task) -> float:
    if not task.milestones:
        return 0.0
    total_possible = sum(m.reward for m in task.milestones)
    if total_possible == 0:
        return 0.0
    achieved = sum(m.reward for m in task.milestones if m.id in milestones_achieved)
    return min(1.0, achieved / total_possible)

def compute_task_completion_reward(success_conditions_met: list[bool], task: Task) -> float:
    if not success_conditions_met:
        return 0.0
    return sum(success_conditions_met) / len(success_conditions_met)

def compute_replan_bonus(exo_events_seen: int, milestones_after_event: int) -> float:
    # Scale bonus based on ability to bounce back after exogenous events
    if exo_events_seen == 0:
        return 0.0
    return min(1.0, (milestones_after_event / exo_events_seen) * 0.5)

def compute_dead_end_penalty(routes_remaining: int) -> float:
    return -0.5 if routes_remaining <= 0 else 0.0

def compute_task_reward(
    state_before: LifeMetrics,
    state_after: LifeMetrics,
    resources_used: dict,
    actions_taken: int,
    milestones_achieved: list[str],
    success_conditions_met: list[bool],
    exo_events_seen: int,
    milestones_after_event: int,
    routes_remaining: int,
    rollback_used: bool,
    cascade_collapse: bool,
    task: Task,
    reasoning: str = "",
    completion: str = "",
    conflict_domain: str = "",
    step_count: int = 0,
    max_steps: int = 0,
    metric_changes: dict = None,
    cumulative_rel_delta: float = 0.0
) -> tuple[float, dict]:
    # 1. Base local components (with scaled disruption baseline from task metadata)
    d_baseline = len(task.mutable_world) if task and hasattr(task, 'mutable_world') else None
    local_reward, local_breakdown = compute_reward(state_before, state_after, resources_used, actions_taken,
                                                   metric_changes=metric_changes, completion=completion,
                                                   disruption_baseline=d_baseline)

    # 2. Orchestrator components
    # Use only the raw outcome component from local_breakdown to avoid double-counting 
    # efficiency, containment, or preservation which are added separately below.
    outcome_score_local = local_breakdown["components"].get("outcome", 0.0)
    milestone_score = compute_milestone_reward(milestones_achieved, task)
    completion_score = compute_task_completion_reward(success_conditions_met, task)
    replan_score = compute_replan_bonus(exo_events_seen, milestones_after_event)
    efficiency_score = local_breakdown["components"].get("efficiency", 0.0)
    preservation_score = local_breakdown["components"].get("preservation", 0.0)
    reasoning_score = reward_reasoning_coherence(reasoning, conflict_domain)
    
    # Check for specific failure cases
    timeout_pen = reward_timeout_check(step_count, max_steps, any(success_met for success_met in success_conditions_met) if success_conditions_met else False)
    dead_end_pen = compute_dead_end_penalty(routes_remaining)
    
    # 3. Final weighting (all components are now unique/non-overlapping)
    base_reward = (
        (0.35 * milestone_score) + 
        (0.25 * completion_score) + 
        (0.15 * outcome_score_local) + 
        (0.10 * replan_score) + 
        (0.10 * efficiency_score) + 
        (0.05 * reasoning_score)
    )

    # 4. Penalties
    penalties = timeout_pen
    fired = []

    dead_end_pen = compute_dead_end_penalty(routes_remaining)
    if dead_end_pen < 0:
        penalties += dead_end_pen
        fired.append("DEAD_END")

    if rollback_used:
        penalties += -0.1
        fired.append("ROLLBACK_USED")

    if cascade_collapse:
        penalties += -0.3
        fired.append("CASCADE_COLLAPSE")

    # Direct inaction penalty — not diluted by the 0.05 local weight
    if actions_taken == 0:
        penalties += -0.20
        fired.append("TASK_INACTION_PENALTY")

    # Cumulative relationship erosion across the episode
    if cumulative_rel_delta < -20:
        penalties += -0.15
        fired.append("CUMULATIVE_RELATIONSHIP_EROSION")

    final_reward = max(-1.0, min(1.0, base_reward + penalties))

    breakdown = {
        "components": {
            "local_metric_delta": local_metric_delta_score,
            "milestone": milestone_score,
            "completion": completion_score,
            "replan": replan_score,
            "efficiency": efficiency_score,
            "reasoning": reasoning_score,
            "format_compliance": format_score,
            "plausibility": local_breakdown["components"].get("plausibility", 0.0),
            "timeout_penalty": timeout_pen
        },
        "base_reward": base_reward,
        "penalties_total": penalties,
        "penalties_fired": fired,
        "local_breakdown": local_breakdown
    }

    return final_reward, breakdown

def reward_format_compliance(completion: str) -> float:
    """
    Scores the completion based on its format (JSON validity and required fields).
    
    Returns:
        +1.0: Valid JSON with all required fields (action_type, metric_changes, resource_cost, reasoning)
        +0.5: Valid JSON but missing one or more required fields
        -0.5: Invalid JSON / unparseable
        -1.0: Empty strings or refusal content
    """
    if not completion or len(completion.strip()) < 10:
        return -1.0
        
    # Potential refusal indicators
    if any(x in completion.lower() for x in ["i cannot", "i'm sorry", "as an ai"]):
        return -1.0

    # Extract JSON content from markdown code blocks if present
    json_str = completion.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[-1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[-1].split("```")[0].strip()
        
    try:
        data = json.loads(json_str)
        required = ["action_type", "metric_changes", "resource_cost", "reasoning"]
        if all(k in data for k in required):
            return 1.0
        return 0.5
    except json.JSONDecodeError:
        # Final attempt: try to find anything between { and }
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                required = ["action_type", "metric_changes", "resource_cost", "reasoning"]
                if all(k in data for k in required):
                    return 1.0
                return 0.5
            except:
                pass
        return -0.5

def reward_plausibility_check(metric_changes: dict, resource_cost: dict) -> float:
    """
    Anti-gaming check. Prevents the model from claiming massive metric changes while spending 0 resources.
    Resource cost is normalized to comparable units (time/20h, money/$500, energy/100pts).
    """
    total_delta = sum(abs(v) for v in metric_changes.values())

    # Zero-cost shortcut: any non-trivial claim with no cost at all is implausible
    # Also handles empty resource_cost.
    if not resource_cost or all(v == 0 for v in resource_cost.values()):
        if total_delta > 3.0:
            return -0.30
        return 0.0

    # Normalize each resource dimension to [0,1] before summing
    norm_time   = resource_cost.get('time', 0.0)   / 20.0
    norm_money  = resource_cost.get('money', 0.0)  / 500.0
    norm_energy = resource_cost.get('energy', 0.0) / 100.0
    total_cost  = norm_time + norm_money + norm_energy

    ratio = total_delta / max(0.01, total_cost)

    if ratio > 150:
        return -0.30   # Claiming massive change for virtually free
    if ratio > 80:
        return -0.10   # Highly suspicious efficiency
    return 0.0         # Plausible ratio

def reward_timeout_check(step_count: int, max_steps: int, done: bool) -> float:
    """
    Penalizes episodes that end by reaching the step limit without being resolved.
    """
    if step_count >= max_steps and not done:
        return -0.20
    return 0.0

def reward_reasoning_coherence(reasoning: str, conflict_domain: str) -> float:
    """
    Process-aware intermediate reward.
    Requires a non-empty domain AND non-empty reasoning — empty domain would match every string.
    """
    if not conflict_domain or not reasoning:
        return -0.10

    reasoning_lower = reasoning.lower()

    if conflict_domain.lower() in reasoning_lower:
        return 0.10

    if len(reasoning.strip()) < 20:
        return -0.10

    return 0.0

def main():
    # Scenario setup
    print("--- TESTING REWARD SYSTEM ---")
    
    # 1. PERFECT ACTION: All metrics improve by 10 points
    state_start = LifeMetrics() # Defaults at 70
    state_perfect = copy.deepcopy(state_start)
    for k in state_perfect.flatten().keys():
        domain, sub = k.split('.')
        current = getattr(getattr(state_perfect, domain), sub)
        setattr(getattr(state_perfect, domain), sub, current + 10)
    
    res_perfect = {"time": 2, "money": 50, "energy": 10}
    reward_p, break_p = compute_reward(state_start, state_perfect, res_perfect, actions_taken=5)
    
    print("\n[SCENARIO 1: PERFECT ACTION]")
    print(f"Reward: {reward_p:.4f}")
    print(f"Breakdown: {break_p}")

    # 2. BAD ACTION: Relationships tank by 30 points, everything else stays same
    state_bad = copy.deepcopy(state_start)
    for k in state_bad.flatten().keys():
        if k.startswith('relationships.'):
            domain, sub = k.split('.')
            current = getattr(getattr(state_bad, domain), sub)
            setattr(getattr(state_bad, domain), sub, current - 30)
            
    res_bad = {"time": 10, "money": 300, "energy": 80}
    reward_b, break_b = compute_reward(state_start, state_bad, res_bad, actions_taken=1)
    
    print("\n[SCENARIO 2: BAD ACTION (Relationships Tank)]")
    print(f"Reward: {reward_b:.4f}")
    print(f"Breakdown: {break_b}")

    # 3. INACTION: Nothing changes
    state_nothing = copy.deepcopy(state_start)
    res_none = {}
    reward_n, break_n = compute_reward(state_start, state_nothing, res_none, actions_taken=0)
    
    print("\n[SCENARIO 3: INACTION]")
    print(f"Reward: {reward_n:.4f}")
    print(f"Breakdown: {break_n}")

if __name__ == "__main__":
    main()
