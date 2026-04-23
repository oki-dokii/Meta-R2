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

def compute_human_feedback_reward(predicted_obs: LifeStackObservation, feedback: OutcomeFeedback) -> float:
    """
    Computes a reward score (0.0 to 1.0) based on how well the environment's 
    predicted outcomes match the human's reported reality.
    """
    # 1. Domain Alignment Score
    # We look at which domains improved in the predicted metrics
    # Note: predicted_obs.metrics is a flat dict of "domain.submetric": value (delta or final?)
    # Usually in Step rewards, we look at the DELTA. However, compute_human_feedback_reward 
    # likely compares the END OF EPISODE predicted state vs reality.
    
    # If predicted_obs represents the final state, we should compare it to the initial state.
    # For now, we assume metrics here are deltas or final levels > threshold.
    # Let's assume positive values in predicted_obs.metrics represent improvements if they are deltas.
    # Actually, in LifeStackEnv, obs.metrics is the final value. 
    # To find improvement, we'd need the initial state.
    
    # But wait, the user's snippet says:
    # predicted_improved = set(k.split('.')[0] for k, v in predicted_obs.metrics.items() if v > 0)
    # This implies v > 0 means improvement (delta logic).
    
    predicted_improved = set(k.split('.')[0] for k, v in predicted_obs.metrics.items() if v > 0)
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
