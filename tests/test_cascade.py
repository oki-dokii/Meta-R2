import pytest
from core.cascade_utils import animate_cascade
from core.life_state import LifeMetrics

def test_cascade_frame_count():
    """Ensure the cascade animation produces the standard 4 frames (Stable, Disruption, Cascade 1, Cascade 2)."""
    metrics = LifeMetrics()
    disruption = {"mental_wellbeing.stress_level": 30.0}
    
    frames = animate_cascade(disruption, metrics)
    
    assert len(frames) == 4
    assert frames[1]["status"]["mental_wellbeing.stress_level"] == "primary"
    assert "first" in frames[2]["status"].values()
    assert "second" in frames[3]["status"].values()

def test_cascade_value_propagation():
    """Verify that a disruption in stress propagates to sleep_quality via the dependency graph."""
    metrics = LifeMetrics()
    base_sleep = metrics.physical_health.sleep_quality
    
    # Stress (90) -> Sleep Quality (-0.4 weight in graph)
    disruption = {"mental_wellbeing.stress_level": 50.0}
    frames = animate_cascade(disruption, metrics)
    
    final_sleep = frames[-1]["flat"]["physical_health.sleep_quality"]
    assert final_sleep < base_sleep, "Cascade failed to propagate stress to sleep quality"
