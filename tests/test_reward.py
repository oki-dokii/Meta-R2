import pytest
import copy
from core.reward import compute_task_reward
from core.action_space import AgentAction, PrimaryAction
from core.task import TaskGenerator
from core.life_state import LifeMetrics

def test_reward_milestone_bonus():
    """Verify that hitting a milestone results in a positive reward component."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    milestone = task.milestones[0]
    
    # State before: empty world
    state_before = LifeMetrics()
    
    # State after: satisfied the milestone condition
    state_after = copy.deepcopy(state_before)
    # Most milestones in TaskGenerator are boolean flags in mutable_world
    # We must simulate the world mutation that matches the task logic
    # Note: Task metrics are actually in the flat 'mutable_world' or specific LifeMetrics domains
    # For smoke test, we simulate the 'milestones_achieved' list directly as returned by Env
    
    reward, breakdown = compute_task_reward(
        state_before=state_before,
        state_after=state_after,
        resources_used={"time": 1.0, "energy": 10.0},
        actions_taken=1,
        milestones_achieved=[milestone.id],
        success_conditions_met=[False],
        exo_events_seen=0,
        milestones_after_event=0,
        routes_remaining=1,
        rollback_used=False,
        cascade_collapse=False,
        task=task
    )
    assert breakdown["components"]["milestone"] > 0
    assert reward >= 0

def test_reward_scaling_with_impact():
    """Verify that improving metrics results in higher outcome reward than stationary state."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    
    state_before = LifeMetrics()
    
    # Positive case: metrics improve
    state_good = copy.deepcopy(state_before)
    state_good.career.stability = 90.0 # Started at 70
    
    # Neutral case: no change
    state_neutral = copy.deepcopy(state_before)
    
    reward_good, break_good = compute_task_reward(
        state_before=state_before, state_after=state_good,
        resources_used={"time": 1.0}, actions_taken=1, milestones_achieved=[],
        success_conditions_met=[False], exo_events_seen=0, milestones_after_event=0,
        routes_remaining=1, rollback_used=False, cascade_collapse=False, task=task
    )
    
    reward_neutral, break_neutral = compute_task_reward(
        state_before=state_before, state_after=state_neutral,
        resources_used={"time": 1.0}, actions_taken=1, milestones_achieved=[],
        success_conditions_met=[False], exo_events_seen=0, milestones_after_event=0,
        routes_remaining=1, rollback_used=False, cascade_collapse=False, task=task
    )
    
    assert break_good["components"]["local_metric_delta"] > break_neutral["components"]["local_metric_delta"]
