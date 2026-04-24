import sys
import os
sys.path.append(os.getcwd())

from core.reward import compute_task_reward
from core.life_state import LifeMetrics
from core.task import Task

def test_reasoning_alignment_pass():
    """Verify that reasoning mentioning the action category gets higher score."""
    state = LifeMetrics()
    task = Task(
        "test", 
        "career",
        "test", 
        {}, {}, {}, {}, # constraints, hidden, mutable, visible
        [], [], # success, failure
        [], [], [], # events, routes, milestones
        10, 1, {} # horizon, diff, metadata
    )
    
    # Action: SPEND, Reasoning: mentions cost
    _, res_match = compute_task_reward(
        state_before=state, state_after=state, resources_used={}, actions_taken=1,
        milestones_achieved=[], success_conditions_met=[], exo_events_seen=0,
        milestones_after_event=0, routes_remaining=1, rollback_used=False,
        cascade_collapse=False, task=task, action_type="spend",
        reasoning="I am doing this because the cost is low."
    )
    
    # Action: SPEND, Reasoning: mentions nothing relevant
    _, res_mismatch = compute_task_reward(
        state_before=state, state_after=state, resources_used={}, actions_taken=1,
        milestones_achieved=[], success_conditions_met=[], exo_events_seen=0,
        milestones_after_event=0, routes_remaining=1, rollback_used=False,
        cascade_collapse=False, task=task, action_type="spend",
        reasoning="I am doing this because I am happy."
    )
    
    score_match = res_match["components"]["reasoning"]
    score_mismatch = res_mismatch["components"]["reasoning"]
    
    print(f"Match Score: {score_match}")
    print(f"Mismatch Score: {score_mismatch}")
    
    assert score_match > score_mismatch, f"Match {score_match} should be > Mismatch {score_mismatch}"
    print("✅ Reasoning alignment test passed!")

if __name__ == "__main__":
    try:
        test_reasoning_alignment_pass()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
