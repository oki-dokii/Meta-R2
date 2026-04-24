from typing import Dict, List, Set, Any, Tuple
from core.task import Task, Milestone, Route

class LifeStackVerifier:
    """Standalone verifier for Task success, failure, and progression."""
    
    @staticmethod
    def _check_cond(cond: dict, world_state: dict, hidden_state: dict, metrics_flat: dict = None) -> bool:
        key = cond['key']
        target = cond['value']
        op = cond.get('op', 'eq')
        
        # Priority: Metrics > Hidden > World
        val = None
        if metrics_flat and key in metrics_flat:
            val = metrics_flat[key]
        else:
            val = hidden_state.get(key, world_state.get(key))
            
        if val is None:
            return False
            
        if op == 'eq': return val == target
        if op == 'ne': return val != target
        if op == 'gt': return val > target
        if op == 'lt': return val < target
        if op == 'ge': return val >= target
        if op == 'le': return val <= target
        return False

    @staticmethod
    def check_success(task: Task, world_state: dict, hidden_state: dict) -> list[bool]:
        """Checks if task-specific success conditions are met."""
        return [LifeStackVerifier._check_cond(c, world_state, hidden_state) for c in task.success_conditions]

    @staticmethod
    def check_failure(task: Task, world_state: dict, hidden_state: dict, metrics_flat: dict) -> list[bool]:
        """Checks if task-specific or global failure conditions (metric death) are met."""
        results = [LifeStackVerifier._check_cond(c, world_state, hidden_state, metrics_flat) for c in task.failure_conditions]
        # 2. Metric death
        if any(v <= 10 for v in metrics_flat.values()):
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
