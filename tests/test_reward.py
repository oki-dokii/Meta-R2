import pytest
from core.reward import compute_task_reward
from core.action_space import AgentAction, PrimaryAction
from core.task import TaskGenerator
from core.life_state import LifeMetrics

def test_reward_milestone_bonus():
    """Verify that hitting a milestone results in a positive reward component."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    milestone = task.milestones[0]
    
    # Mock an action that matches the milestone
    action = AgentAction(
        primary=PrimaryAction(
            action_type="execute",
            target_domain="career",
            metric_changes={},
            resource_cost={},
            description="Matching milestone action"
        ),
        reasoning="Testing reward"
    )
    
    reward, breakdown = compute_task_reward(
        state_before=LifeMetrics(),
        state_after=LifeMetrics(),
        resources_used={},
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

def test_reward_scaling():
    """Ensure that the total reward is bounded reasonably."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    action = AgentAction(
        primary=PrimaryAction(
            action_type="rest", 
            target_domain="time",
            metric_changes={},
            resource_cost={},
            description="Short breather"
        ),
        reasoning="Testing reward"
    )
    
    reward, breakdown = compute_task_reward(
        state_before=LifeMetrics(),
        state_after=LifeMetrics(),
        resources_used={},
        actions_taken=1,
        milestones_achieved=[],
        success_conditions_met=[False],
        exo_events_seen=0,
        milestones_after_event=0,
        routes_remaining=1,
        rollback_used=False,
        cascade_collapse=False,
        task=task
    )
    assert -1.0 <= reward <= 2.0  # Basic bounds
