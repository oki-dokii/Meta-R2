from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class HiddenStateField:
    key: str               # e.g. "boss_mood"
    initial_value: Any     # e.g. "neutral"
    inspect_target: str    # e.g. "call_boss" — which inspect action type reveals this
    description: str       # shown to agent after reveal

@dataclass
class ExoEvent:
    step: int              # inject at this step (inclusive); -1 = probabilistic
    probability: float     # 1.0 = deterministic; <1.0 = random at each step
    id: str                # e.g. "ticket_price_spike"
    description: str       # what agent sees in next observation
    world_mutation: Dict   # e.g. {"ticket_price": 450, "seats_remaining": 1}
    hidden_state_mutation: Dict  # e.g. {"boss_mood": "angry"}
    closes_routes: List[str] = field(default_factory=list)  # route IDs this event blocks

@dataclass
class Milestone:
    id: str                # e.g. "flight_rebooked"
    description: str
    condition_key: str     # world/hidden key to check, e.g. "flight_rebooked"
    condition_value: Any   # e.g. True
    reward: float          # milestone reward added to episode total

@dataclass
class Route:
    id: str                # e.g. "rebook_premium"
    name: str
    description: str
    required_action_types: List[str]  # must use these tool actions to complete
    preconditions: Dict    # world/hidden state checks, e.g. {"card_available": True}
    consequences: Dict     # world mutations on route completion, e.g. {"flight_rebooked": True}
    closes_routes: List[str]  # route IDs this blocks
    milestones_unlocked: List[str]  # milestone IDs this route can hit
    final_reward: float    # bonus on route completion

@dataclass
class Task:
    id: str
    domain: str            # "flight_crisis" | "code_merge_crisis"
    goal: str
    constraints: Dict      # e.g. {"budget_max": 400, "deadline_step": 18}
    hidden_state: Dict     # full truth, agent never sees directly
    mutable_world: Dict    # partial truth, some fields revealed by inspect
    visible_world: Dict    # agent sees this at each step (subset of mutable_world)
    success_conditions: List[Dict]  # e.g. [{"key": "flight_rebooked", "value": True}]
    failure_conditions: List[Dict]  # e.g. [{"key": "missed_deadline", "value": True}]
    event_schedule: List[ExoEvent]
    viable_routes: List[Route]
    milestones: List[Milestone]
    horizon: int           # max steps (20–50)
    difficulty: int        # 1–5
    domain_metadata: Dict  # domain-specific extra data (story text, etc.)
