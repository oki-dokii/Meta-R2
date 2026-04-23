from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from core.lifestack_env import LifeStackObservation

@dataclass
class OutcomeFeedback:
    episode_id: str
    submitted_at: datetime = field(default_factory=datetime.now)
    # Did the advice work overall? 0-10 scale
    overall_effectiveness: int = 5
    # Which domains actually changed (user-reported)
    domains_improved: List[str] = field(default_factory=list)
    domains_worsened: List[str] = field(default_factory=list)
    # Free text: what unexpected effects happened?
    unexpected_effects: str = ""
    # Time to resolution (hours)
    resolution_time_hours: float = 0.0

def compute_human_feedback_reward(initial_metrics: dict, predicted_obs: LifeStackObservation, feedback: OutcomeFeedback) -> float:
    """
    Computes a reward score (0.0 to 1.0) based on how well the environment's 
    predicted outcomes match the human's reported reality.
    """
    # Metrics where a decrease is an improvement
    inverted = {"stress_level", "debt_pressure", "workload", "commute_burden", "admin_overhead"}
    
    predicted_improved = set()
    for key, final_val in predicted_obs.metrics.items():
        if key not in initial_metrics:
            continue
            
        initial_val = initial_metrics[key]
        delta = final_val - initial_val
        submetric = key.split('.')[-1]
        domain = key.split('.')[0]
        
        # Determine if this specific change is an "improvement"
        is_improvement = False
        if submetric in inverted:
            if delta < -1.0: # Significant decrease in negative metric
                is_improvement = True
        else:
            if delta > 1.0:  # Significant increase in positive metric
                is_improvement = True
                
        if is_improvement:
            predicted_improved.add(domain)

    actual_improved = set(feedback.domains_improved)

    union = predicted_improved | actual_improved
    if not union:
        overlap = 1.0 # Both agreed nothing improved
    else:
        intersection = predicted_improved & actual_improved
        overlap = len(intersection) / len(union)
        
    # 2. Effectiveness Score (0.0 - 1.0)
    effectiveness_score = max(0.0, min(1.0, feedback.overall_effectiveness / 10.0))

    # Weighted Average
    return 0.5 * overlap + 0.5 * effectiveness_score
