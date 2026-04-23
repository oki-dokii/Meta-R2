from typing import Dict, List, Set, Any, Tuple
from core.task import Task, Milestone, Route

class LifeStackVerifier:
    """Standalone verifier for Task success, failure, and progression."""
    
    @staticmethod
    def check_success(task: Task, world_state: dict, hidden_state: dict) -> list[bool]:
        """Checks if task-specific success conditions are met."""
        results = []
        for cond in task.success_conditions:
            val = hidden_state.get(cond['key'], world_state.get(cond['key']))
            results.append(val == cond['value'])
        return results

    @staticmethod
    def check_failure(task: Task, world_state: dict, hidden_state: dict, metrics_flat: dict) -> list[bool]:
        """Checks if task-specific or global failure conditions (metric death) are met."""
        results = []
        # 1. Task failures
        for cond in task.failure_conditions:
            val = hidden_state.get(cond['key'], world_state.get(cond['key']))
            results.append(val == cond['value'])
        # 2. Metric death
        if any(v <= 0 for v in metrics_flat.values()):
            results.append(True)
        return results

    @staticmethod
    def check_new_milestones(task: Task, world_state: dict, hidden_state: dict, achieved_ids: list) -> list[str]:
        """Identifies any milestones that have just been met by current state."""
        newly_met = []
        for m in task.milestones:
            if m.id not in achieved_ids:
                val = hidden_state.get(m.condition_key, world_state.get(m.condition_key))
                if val == m.condition_value:
                    newly_met.append(m.id)
        return newly_met

    @staticmethod
    def get_route_status(task: Task, closed_ids: set, world_state: dict, hidden_state: dict) -> Tuple[int, bool]:
        """Returns (remaining_routes_count, is_dead_end)."""
        remaining = 0
        for route in task.viable_routes:
            if route.id in closed_ids:
                continue
            
            # Check if reachable via preconditions
            pre_ok = True
            for k, v in route.preconditions.items():
                current_v = hidden_state.get(k, world_state.get(k))
                if current_v != v:
                    pre_ok = False
                    break
            
            if pre_ok:
                remaining += 1
        
        return remaining, remaining == 0
