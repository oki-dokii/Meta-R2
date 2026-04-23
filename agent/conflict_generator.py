import json
import random
from dataclasses import dataclass, field, asdict

@dataclass
class ConflictEvent:
    id: str
    title: str
    story: str
    primary_disruption: dict
    decisions_required: list[str]
    resource_budget: dict
    difficulty: int

TEMPLATES = [
    # DIFFICULTY 1
    ConflictEvent(
        id="d1_gym",
        title="The Slump",
        story="You haven't seen the inside of a gym in ten days. Your energy is flagging and your favorite jeans feel tight.",
        primary_disruption={"physical_health.fitness": -15.0},
        decisions_required=["Wake up early for a run", "Join a weekend boot camp", "Ignore it and rest"],
        resource_budget={"time": 4.0, "money": 0.0, "energy": 20.0},
        difficulty=1
    ),
    ConflictEvent(
        id="d1_bill",
        title="Forgotten Invoice",
        story="A late notice arrived for your electricity bill. It's not a lot, but the late fee is annoying.",
        primary_disruption={"finances.liquidity": -20.0},
        decisions_required=["Pay it now", "Call to dispute the fee", "Set up autopay for next time"],
        resource_budget={"time": 1.0, "money": 100.0, "energy": 5.0},
        difficulty=1
    ),
    ConflictEvent(
        id="d1_argument",
        title="Heated Group Chat",
        story="A minor political disagreement in the group chat turned personal. Everyone is being quiet now.",
        primary_disruption={"relationships.social": -20.0},
        decisions_required=["Apologize to the group", "Message the friend privately", "Mute the chat for a week"],
        resource_budget={"time": 2.0, "money": 30.0, "energy": 15.0},
        difficulty=1
    ),

    # DIFFICULTY 2
    ConflictEvent(
        id="d2_project",
        title="The Surge",
        story="Your boss just walked by and dropped a 'small favor' on your desk. It looks like it'll take ten hours.",
        primary_disruption={"career.workload": 25.0, "time.free_hours_per_week": -20.0},
        decisions_required=["Work late all week", "Delegate parts to a junior", "Refuse the assignment"],
        resource_budget={"time": 10.0, "money": 0.0, "energy": 40.0},
        difficulty=2
    ),
    ConflictEvent(
        id="d2_car",
        title="Check Engine Light",
        story="Your car started making a rhythmic thumping sound on the highway. The mechanic says the repair isn't cheap.",
        primary_disruption={"finances.liquidity": -30.0, "time.commute_burden": 25.0},
        decisions_required=["Repair it immediately", "Take the bus for a week", "Borrow a car from a friend"],
        resource_budget={"time": 5.0, "money": 500.0, "energy": 10.0},
        difficulty=2
    ),
    ConflictEvent(
        id="d2_neglect",
        title="Cold Dinner",
        story="Your partner mentions they feel like 'roommates' lately. You realize you haven't had a real conversation in weeks.",
        primary_disruption={"relationships.romantic": -25.0, "mental_wellbeing.stress_level": 20.0},
        decisions_required=["Plan a surprise date", "Have a long talk tonight", "Buy a thoughtful gift"],
        resource_budget={"time": 6.0, "money": 150.0, "energy": 30.0},
        difficulty=2
    ),

    # DIFFICULTY 3
    ConflictEvent(
        id="d3_interview",
        title="The Opportunity",
        story="An old contact reached out for a dream job interview. You need to prep while keeping your current job afloat.",
        primary_disruption={"career.workload": 20.0, "time.free_hours_per_week": -15.0, "mental_wellbeing.stress_level": 20.0},
        decisions_required=["Intensive weekend prep", "Fake a sick day to interview", "Turn it down to stay stable"],
        resource_budget={"time": 12.0, "money": 50.0, "energy": 50.0},
        difficulty=3
    ),
    ConflictEvent(
        id="d3_family",
        title="Family SOS",
        story="Your sibling is going through a rough patch and needs help moving out and some financial support.",
        primary_disruption={"relationships.family": 20.0, "time.free_hours_per_week": -25.0, "finances.liquidity": -20.0},
        decisions_required=["Spend the weekend helping", "Send them money but stay home", "Help them find other movers"],
        resource_budget={"time": 15.0, "money": 400.0, "energy": 60.0},
        difficulty=3
    ),
    ConflictEvent(
        id="d3_health",
        title="The Warning Sign",
        story="You had a fainting spell at the office. Tests are expensive, and doctors say you need immediate change.",
        primary_disruption={"physical_health.energy": -30.0, "mental_wellbeing.stress_level": 30.0, "finances.liquidity": -40.0},
        decisions_required=["Take a week of medical leave", "Consult a high-end specialist", "Change diet and sleep habits"],
        resource_budget={"time": 20.0, "money": 800.0, "energy": 5.0},
        difficulty=3
    ),

    # DIFFICULTY 4
    ConflictEvent(
        id="d4_review",
        title="Judgment Day",
        story="A major performance review is in three days. Rumors of layoffs are circulating and the atmosphere is tense.",
        primary_disruption={"career.workload": 30.0, "mental_wellbeing.stress_level": 25.0, "relationships.romantic": -15.0, "time.free_hours_per_week": -20.0},
        decisions_required=["Pull all-nighters to prove worth", "Start networking for new roles", "Draft a defensive report"],
        resource_budget={"time": 18.0, "money": 0.0, "energy": 80.0},
        difficulty=4
    ),
    ConflictEvent(
        id="d4_move",
        title="The Big Relocation",
        story="You've decided to move across the country for growth. The logistics are a nightmare and friends are sad to see you go.",
        primary_disruption={"finances.liquidity": -50.0, "relationships.social": -30.0, "career.growth_trajectory": 20.0, "time.admin_overhead": 30.0},
        decisions_required=["Hire full-service movers", "Host a series of farewell dinners", "DIY pack everything"],
        resource_budget={"time": 30.0, "money": 1500.0, "energy": 100.0},
        difficulty=4
    ),
    ConflictEvent(
        id="d4_audit",
        title="Tax Audit",
        story="The IRS has flagged your last three years of returns. You need to dig through thousands of documents while paying a CPA.",
        primary_disruption={"finances.long_term_health": -20.0, "mental_wellbeing.stress_level": 30.0, "time.admin_overhead": 40.0, "finances.liquidity": -15.0},
        decisions_required=["Spend nights scanning receipts", "Hire a tax lawyer", "Try to settle immediately"],
        resource_budget={"time": 25.0, "money": 1000.0, "energy": 40.0},
        difficulty=4
    ),

    # DIFFICULTY 5
    ConflictEvent(
        id="d5_friday",
        title="Friday 6PM",
        story="Your flight just got cancelled. Your card declined trying to rebook. Your boss moved Monday deadline to Sunday.",
        primary_disruption={"career.workload": 35.0, "finances.liquidity": -40.0, "mental_wellbeing.stress_level": 30.0, "time.free_hours_per_week": -25.0},
        decisions_required=["Book a bus and work on it", "Call boss to negotiate", "Crash at a nearby friend's"],
        resource_budget={"time": 10.0, "money": 500.0, "energy": 60.0},
        difficulty=5
    ),
    ConflictEvent(
        id="d5_storm",
        title="The Perfect Storm",
        story="Your firm lost its biggest client, your partner moved out, and your car got towed—all on the same Tuesday.",
        primary_disruption={"career.stability": -30.0, "relationships.romantic": -25.0, "finances.debt_pressure": 35.0, "physical_health.energy": -25.0},
        decisions_required=["Find an emergency side hustle", "Beg partner for a second chance", "Take a mental health day"],
        resource_budget={"time": 8.0, "money": 200.0, "energy": 20.0},
        difficulty=5
    ),
    ConflictEvent(
        id="d5_burnout",
        title="The Total Collapse",
        story="You can't get out of bed. Your body has quit, your motivation is gone, and work emails are piling into the hundreds.",
        primary_disruption={"mental_wellbeing.motivation": -40.0, "physical_health.sleep_quality": -30.0, "career.satisfaction": -35.0, "relationships.family": -20.0},
        decisions_required=["Request indefinite medical leave", "Disconnect all electronics", "Let it all burn and sleep"],
        resource_budget={"time": 40.0, "money": 2000.0, "energy": 0.0},
        difficulty=5
    )
]

def generate_conflict(difficulty: int = None) -> ConflictEvent:
    if difficulty:
        pool = [t for t in TEMPLATES if t.difficulty == difficulty]
    else:
        pool = TEMPLATES
    return random.choice(pool)

def escalate_conflict(conflict: ConflictEvent) -> ConflictEvent:
    new_disruption = {k: v * 1.4 for k, v in conflict.primary_disruption.items()}
    new_budget = {k: v * 0.7 for k, v in conflict.resource_budget.items()}
    new_difficulty = min(5, conflict.difficulty + 1)
    
    return ConflictEvent(
        id=f"{conflict.id}_escalated",
        title=f"ESCALATED: {conflict.title}",
        story=f"Current situation just got much worse. {conflict.story}",
        primary_disruption=new_disruption,
        decisions_required=conflict.decisions_required,
        resource_budget=new_budget,
        difficulty=new_difficulty
    )

def adaptive_escalate(conflict: ConflictEvent, agent_history: list) -> tuple:
    """Decide whether to escalate, ease, or hold based on past performance.
    
    Args:
        conflict: Current conflict event.
        agent_history: List of (conflict_id, reward) tuples from past episodes.
    
    Returns:
        (new_conflict, reason): Updated conflict and a human-readable reason string.
    """
    # Group history by conflict id prefix (strip _escalated suffix)
    from collections import defaultdict
    by_type = defaultdict(list)
    for cid, reward in agent_history:
        base_id = cid.replace("_escalated", "")
        by_type[base_id].append(reward)
    
    base_id = conflict.id.replace("_escalated", "")
    past = by_type.get(base_id, [])
    
    if len(past) >= 3:
        avg = sum(past) / len(past)
        if avg > 0.7:
            # Agent is crushing this type — escalate
            escalated = escalate_conflict(conflict)
            return escalated, f"Agent averaged {avg:.2f} on {base_id} ({len(past)} runs) — escalating"
        elif avg < 0.4:
            # Agent is struggling — reduce difficulty
            new_diff = max(1, conflict.difficulty - 1)
            eased = generate_conflict(difficulty=new_diff)
            return eased, f"Agent averaged {avg:.2f} on {base_id} ({len(past)} runs) — easing to difficulty {new_diff}"
    
    # Not enough history — no change
    return conflict, "insufficient history — holding"

def save_templates():
    import os
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conflicts.json")
    with open(data_path, 'w') as f:
        json.dump([asdict(t) for t in TEMPLATES], f, indent=4)
    print(f"Saved 15 templates to {data_path}")

def main():
    save_templates()
    print("\n--- GENERATED CONFLICT SAMPLES ---")
    for d in range(1, 6):
        c = generate_conflict(d)
        print(f"\n[DIFFICULTY {d}] {c.title}")
        print(f"Story: {c.story}")
        print(f"Primary Disruption: {c.primary_disruption}")
        print(f"Resource Budget: {c.resource_budget}")

if __name__ == "__main__":
    main()

from core.task import Task, Route, ExoEvent, Milestone

class TaskGenerator:
    def generate(self, domain: str = None, difficulty: int = None) -> Task:
        diff = difficulty or 3
        if domain == "flight_crisis":
            return self.generate_flight_crisis(diff)
        elif domain == "code_merge_crisis":
            return self.generate_code_merge_crisis(diff)
        else:
            return self.generate_flight_crisis(diff)

    def generate_flight_crisis(self, difficulty: int) -> Task:
        routes = [
            Route(id="rebook_premium", name="Rebook Premium Option", description="Call agent and rebook on premium ticket", required_action_types=["communicate", "spend"], preconditions={}, consequences={}, closes_routes=["wait_lounge"], milestones_unlocked=["m1"], final_reward=2.5),
            Route(id="wait_lounge", name="Accept Delay & Work", description="Stay at airport lounge and work on laptop", required_action_types=["rest", "delegate"], preconditions={}, consequences={}, closes_routes=["rebook_premium"], milestones_unlocked=["m2"], final_reward=1.8),
        ]
        milestones = [
            Milestone(id="m1", description="Successfully rebooked flight before deadline", condition_key="", condition_value=True, reward=1.0),
            Milestone(id="m2", description="Caught up with all emergency slack messages", condition_key="", condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=2, probability=1.0, id="price_surge", description="Ticket prices sharply increased by $300.", world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=4, probability=1.0, id="lounge_full", description="The airport lounge is now at maximum capacity.", world_mutation={}, hidden_state_mutation={}, closes_routes=["wait_lounge"]),
        ]
        return Task(
            id="flight_crisis_task", domain="flight_crisis", goal="Survive Airport Cancellation",
            constraints={"budget_max": 800, "deadline_step": 10},
            hidden_state={"lounge_capacity": 100}, mutable_world={}, visible_world={},
            success_conditions=[], failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "A major storm grounded commercial flights."}
        )

    def generate_code_merge_crisis(self, difficulty: int) -> Task:
        routes = [
            Route(id="revert_commit", name="Revert Commit", description="Quickly revert the broken merge to unblock the team.", required_action_types=["delegate", "communicate"], preconditions={}, consequences={}, closes_routes=["hotfix"], milestones_unlocked=["unblocked"], final_reward=1.5),
            Route(id="hotfix", name="Patch Forward", description="Find the logic error and push a hotfix.", required_action_types=["communicate", "spend"], preconditions={}, consequences={}, closes_routes=["revert_commit"], milestones_unlocked=["fixed"], final_reward=3.0),
        ]
        milestones = [
            Milestone(id="unblocked", description="CI pipeline is green again", condition_key="", condition_value=True, reward=1.0),
            Milestone(id="fixed", description="Bug resolved without losing features", condition_key="", condition_value=True, reward=2.0),
        ]
        events = [
            ExoEvent(step=3, probability=0.8, id="cto_ping", description="CTO asks for an ETA on the fix.", world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
        ]
        return Task(
            id="code_merge_task", domain="code_merge_crisis", goal="Resolve Production Outage",
            constraints={"budget_max": 1000, "deadline_step": 8},
            hidden_state={}, mutable_world={}, visible_world={},
            success_conditions=[], failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=10 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "A botched merge just took down the staging environment."}
        )
