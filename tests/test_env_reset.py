import pytest
from core.lifestack_env import LifeStackEnv

def test_env_reset_consistency():
    """Verify that calling reset() multiple times produces a consistent stable state."""
    env = LifeStackEnv()
    
    # Reset 1
    obs1 = env.reset()
    metrics1 = env.state.current_metrics.flatten()
    
    # Reset 2
    obs2 = env.reset()
    metrics2 = env.state.current_metrics.flatten()
    
    for key in metrics1:
        assert metrics1[key] == metrics2[key], f"Metric {key} inconsistent after multiple resets"

def test_env_reset_custom_conflict():
    """Verify that custom primary disruptions are applied correctly during reset."""
    env = LifeStackEnv()
    custom_conflict = {"career.workload": 20.0, "mental_wellbeing.stress_level": 15.0}
    
    env.reset(conflict=custom_conflict)
    flat = env.state.current_metrics.flatten()
    
    assert flat["career.workload"] > 55.0  # Base is 45.0; +20.0 disruption → expect > 55.0
    assert flat["mental_wellbeing.stress_level"] > 70.0
