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
    world_mutation: dict   # e.g. {"ticket_price": 450, "seats_remaining": 1}
    hidden_state_mutation: dict  # e.g. {"boss_mood": "angry"}
    closes_routes: list[str] = field(default_factory=list)  # route IDs this event blocks

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
    required_action_types: list[str]  # must use these tool actions to complete
    preconditions: dict    # world/hidden state checks, e.g. {"card_available": True}
    consequences: dict     # world mutations on route completion, e.g. {"flight_rebooked": True}
    closes_routes: list[str]  # route IDs this blocks
    milestones_unlocked: list[str]  # milestone IDs this route can hit
    final_reward: float    # bonus on route completion

@dataclass
class Task:
    id: str
    domain: str            # "flight_crisis" | "code_merge_crisis"
    goal: str
    constraints: dict      # e.g. {"budget_max": 400, "deadline_step": 18}
    hidden_state: dict     # full truth, agent never sees directly
    mutable_world: dict    # partial truth, some fields revealed by inspect
    visible_world: dict    # agent sees this at each step (subset of mutable_world)
    success_conditions: list[dict]  # e.g. [{"key": "flight_rebooked", "value": True}]
    failure_conditions: list[dict]  # e.g. [{"key": "missed_deadline", "value": True}]
    event_schedule: list[ExoEvent]
    viable_routes: list[Route]
    milestones: list[Milestone]
    horizon: int           # max steps (20–50)
    difficulty: int        # 1–5
    domain_metadata: dict  # domain-specific extra data (story text, etc.)


def FlightCrisisTask() -> Task:
    routes = [
        Route(
            id="rebook_premium",
            name="Rebook Premium Option",
            description="Call agent and rebook on premium ticket",
            required_action_types=["communicate", "execute"],
            preconditions={"card_available": True},
            consequences={"flight_rebooked": True},
            closes_routes=["wait_lounge"],
            milestones_unlocked=["m1"],
            final_reward=2.5
        ),
        Route(
            id="wait_lounge",
            name="Accept Delay & Work",
            description="Stay at airport lounge and work on laptop",
            required_action_types=["wait", "plan"],
            preconditions={"lounge_access": True},
            consequences={"caught_up": True},
            closes_routes=["rebook_premium"],
            milestones_unlocked=["m2"],
            final_reward=1.8
        )
    ]
    milestones = [
        Milestone(id="m1", description="Successfully rebooked flight before deadline", condition_key="flight_rebooked", condition_value=True, reward=1.0),
        Milestone(id="m2", description="Caught up with all emergency slack messages", condition_key="caught_up", condition_value=True, reward=0.8),
    ]
    events = [
        ExoEvent(step=5, probability=1.0, id="price_surge", description="Ticket prices sharply increased by $300.", world_mutation={}, hidden_state_mutation={"card_available": False}, closes_routes=[]),
        ExoEvent(step=8, probability=1.0, id="lounge_full", description="The airport lounge is now at maximum capacity.", world_mutation={"lounge_access": False}, hidden_state_mutation={}, closes_routes=["wait_lounge"]),
    ]
    return Task(
        id="flight_crisis_task_main",
        domain="flight_crisis",
        goal="Survive Airport Cancellation",
        constraints={"budget_max": 800, "deadline_step": 20},
        hidden_state={
            "card_available": True
        },
        mutable_world={
            "lounge_access": True,
            "flight_rebooked": False,
            "caught_up": False
        },
        visible_world={
            "lounge_access": True
        },
        success_conditions=[{"key": "flight_rebooked", "value": True}],
        failure_conditions=[{"key": "missed_deadline", "value": True}],
        event_schedule=events,
        viable_routes=routes,
        milestones=milestones,
        horizon=30,
        difficulty=4,
        domain_metadata={"story": "A major storm grounded commercial flights."}
    )

def CodeMergeCrisisTask() -> Task:
    return Task(
        id="code_merge_crisis_stub",
        domain="code_merge_crisis",
        goal="Resolve Production Outage",
        constraints={"budget_max": 1000, "deadline_step": 15},
        hidden_state={}, mutable_world={}, visible_world={},
        success_conditions=[], failure_conditions=[],
        event_schedule=[], viable_routes=[], milestones=[],
        horizon=30, difficulty=3, domain_metadata={}
    )
