import pytest
from core.task import TaskGenerator

def test_task_solvability():
    """Verify that the task goal is achievable through the provided routes."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    
    # Check that at least one success condition key appears in some route consequence
    success_keys = [cond["key"] for cond in task.success_conditions]
    consequence_keys = []
    for route in task.viable_routes:
        consequence_keys.extend(route.consequences.keys())
        
    reachable = any(sk in consequence_keys for sk in success_keys)
    assert reachable, f"Task {task.id} success conditions {success_keys} are not reachable by any route consequences"

def test_task_generation_validity():
    """Verify that the TaskGenerator produces tasks with valid structures (routes, milestones)."""
    gen = TaskGenerator()
    task = gen.get_random_task()
    
    assert task.goal is not None
    assert len(task.viable_routes) > 0
    assert len(task.milestones) > 0
    
    # Check that at least one route has valid action types
    sample_route = task.viable_routes[0]
    assert len(sample_route.required_action_types) > 0

def test_task_diversity():
    """Verify that the task pool contains at least 2 distinct task types (deterministic)."""
    gen = TaskGenerator()
    # Instantiate every task factory directly — no random luck needed
    all_ids = set(factory().id for factory in gen.tasks)
    assert len(all_ids) > 1, "TaskGenerator pool must contain at least 2 distinct task types"
