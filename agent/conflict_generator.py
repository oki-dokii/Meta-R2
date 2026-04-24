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
    ),

    # ── TRANSPORT SCENARIOS (difficulty 1–5, all modes) ──────────────────
    ConflictEvent(
        id="d1_flat_tyre",
        title="Flat Tyre",
        story="Your bike tyre went flat halfway to work. You're going to be late to a team standup.",
        primary_disruption={"time.commute_burden": 20.0, "mental_wellbeing.stress_level": 10.0},
        decisions_required=["Call a cab", "Lock the bike and walk", "Ask to dial into the standup"],
        resource_budget={"time": 2.0, "money": 30.0, "energy": 15.0},
        difficulty=1
    ),
    ConflictEvent(
        id="d2_train_delay",
        title="Train Delay",
        story="Your morning train is delayed 90 minutes due to a signal failure. You have a 9 AM client meeting.",
        primary_disruption={"time.commute_burden": 30.0, "career.workload": 15.0, "mental_wellbeing.stress_level": 15.0},
        decisions_required=["Dial in remotely", "Take a rideshare", "Reschedule the meeting"],
        resource_budget={"time": 3.0, "money": 80.0, "energy": 20.0},
        difficulty=2
    ),
    ConflictEvent(
        id="d3_car_breakdown",
        title="Breakdown on the Highway",
        story="Your car engine seized on the freeway during rush hour. Tow + rental = $400 minimum.",
        primary_disruption={"finances.liquidity": -35.0, "time.commute_burden": 40.0, "mental_wellbeing.stress_level": 20.0},
        decisions_required=["Rent a replacement car", "Rideshare all week", "Borrow from a friend"],
        resource_budget={"time": 6.0, "money": 500.0, "energy": 30.0},
        difficulty=3
    ),
    ConflictEvent(
        id="d4_rideshare_surge",
        title="Surge Pricing Nightmare",
        story="A major event cancelled all transit. Rideshares are 9x surge. You're presenting in 2 hours.",
        primary_disruption={"finances.liquidity": -50.0, "mental_wellbeing.stress_level": 30.0, "time.free_hours_per_week": -10.0},
        decisions_required=["Pay the surge", "Organise a carpool", "Present remotely"],
        resource_budget={"time": 4.0, "money": 200.0, "energy": 40.0},
        difficulty=4
    ),
    ConflictEvent(
        id="d5_transit_strike",
        title="City-Wide Transit Strike",
        story="All buses, trains, and rideshares are on indefinite strike. Your car is in the shop.",
        primary_disruption={"time.commute_burden": 50.0, "finances.liquidity": -30.0, "career.workload": 20.0, "mental_wellbeing.stress_level": 25.0},
        decisions_required=["Negotiate remote work for the week", "Rent an e-bike/scooter", "Crash at a colleague's place"],
        resource_budget={"time": 15.0, "money": 400.0, "energy": 50.0},
        difficulty=5
    ),
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
        if domain == "transport_crisis":
            return self.generate_transport_crisis(diff)
        elif domain == "flight_crisis":          # kept as explicit sub-type
            return self.generate_flight_crisis(diff)
        elif domain == "code_merge_crisis":
            return self.generate_code_merge_crisis(diff)
        elif domain == "career":
            return self.generate_career(diff)
        elif domain == "finances":
            return self.generate_finances(diff)
        elif domain == "relationships":
            return self.generate_relationships(diff)
        elif domain == "physical_health":
            return self.generate_physical_health(diff)
        elif domain == "mental_wellbeing":
            return self.generate_mental_wellbeing(diff)
        elif domain == "time":
            return self.generate_time(diff)
        else:
            return self.generate_transport_crisis(diff)

    # ── TRANSPORT CRISIS: master dispatcher ──────────────────────────────
    def generate_transport_crisis(self, difficulty: int) -> Task:
        """Randomly choose one of 5 real-world transport disruption modes."""
        return random.choice([
            self.generate_flight_crisis,
            self.generate_train_delay,
            self.generate_car_breakdown,
            self.generate_rideshare_surge,
            self.generate_transit_strike,
        ])(difficulty)

    def generate_train_delay(self, difficulty: int) -> Task:
        routes = [
            Route(id="dial_in",     name="Dial In Remotely",       description="Join the meeting via video call from the station.",          required_action_types=["communicate"],           preconditions={}, consequences={"meeting_attended": True},   closes_routes=["rideshare"],  milestones_unlocked=["m1"], final_reward=2.0),
            Route(id="rideshare",   name="Take a Rideshare",        description="Pay for a cab/rideshare and make it there in time.",         required_action_types=["spend", "communicate"],  preconditions={}, consequences={"arrived_on_time": True},  closes_routes=["dial_in"],    milestones_unlocked=["m2"], final_reward=2.5),
            Route(id="reschedule",  name="Reschedule the Meeting",  description="Negotiate a new meeting time with all parties.",            required_action_types=["communicate"],           preconditions={}, consequences={"meeting_rescheduled": True}, closes_routes=[],            milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="Meeting attended on time remotely.",             condition_key="meeting_attended",    condition_value=True, reward=1.0),
            Milestone(id="m2", description="Made it to the office despite the delay.",       condition_key="arrived_on_time",     condition_value=True, reward=1.5),
            Milestone(id="m3", description="Meeting rescheduled without relationship cost.", condition_key="meeting_rescheduled", condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=2, probability=0.8, id="delay_extended",  description="Train delay extended by another 45 minutes.",      world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=4, probability=0.6, id="rideshare_surge", description="Rideshares now showing 3x surge pricing.",          world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
        ]
        return Task(
            id="train_delay_task", domain="transport_crisis", goal="Navigate Train Delay Crisis",
            constraints={"budget_max": 150, "deadline_step": 8},
            hidden_state={"platform_reassigned": False}, 
            mutable_world={"time.commute_burden": 30.0, "mental_wellbeing.stress_level": 15.0}, 
            visible_world={"time.commute_burden": 30.0, "mental_wellbeing.stress_level": 15.0},
            success_conditions=[{"key": "meeting_attended", "value": True}, {"key": "arrived_on_time", "value": True}, {"key": "meeting_rescheduled", "value": True}], 
            failure_conditions=[{"key": "finances.liquidity", "value": 10.0, "op": "lt"}],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=12 + difficulty * 2, difficulty=difficulty,
            domain_metadata={"story": "Signal failure has brought the entire line to a halt.", "transport_mode": "train"}
        )

    def generate_car_breakdown(self, difficulty: int) -> Task:
        routes = [
            Route(id="rent_car",       name="Rent a Replacement Car",     description="Call a rental agency and get mobile again.",              required_action_types=["spend", "communicate"], preconditions={}, consequences={"mobile": True},            closes_routes=[],            milestones_unlocked=["m1"], final_reward=2.5),
            Route(id="rideshare_week", name="Rideshare for the Week",      description="Use rideshares until the car is repaired.",               required_action_types=["spend"],               preconditions={}, consequences={"transport_sorted": True}, closes_routes=["rent_car"],  milestones_unlocked=["m2"], final_reward=1.5),
            Route(id="borrow_car",     name="Borrow a Friend's Car",       description="Call around and borrow a vehicle.",                        required_action_types=["communicate"],          preconditions={}, consequences={"borrowed": True},         closes_routes=[],            milestones_unlocked=["m3"], final_reward=2.0),
        ]
        milestones = [
            Milestone(id="m1", description="Replacement vehicle secured.",               condition_key="mobile",            condition_value=True, reward=1.5),
            Milestone(id="m2", description="Transport plan for the week sorted.",        condition_key="transport_sorted",  condition_value=True, reward=1.0),
            Milestone(id="m3", description="Vehicle borrowed without relationship cost.", condition_key="borrowed",          condition_value=True, reward=1.2),
        ]
        events = [
            ExoEvent(step=2, probability=1.0, id="repair_estimate",   description="Mechanic confirms repair takes 3–5 days, not 1.",       world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=5, probability=0.7, id="rental_shortage",   description="Rental agencies report no compact cars available.",    world_mutation={}, hidden_state_mutation={}, closes_routes=["rent_car"]),
        ]
        return Task(
            id="car_breakdown_task", domain="transport_crisis", goal="Recover from Car Breakdown",
            constraints={"budget_max": 500, "deadline_step": 10},
            hidden_state={"tow_dispatched": False}, 
            mutable_world={"finances.liquidity": -35.0, "time.commute_burden": 40.0}, 
            visible_world={"finances.liquidity": -35.0, "time.commute_burden": 40.0},
            success_conditions=[{"key": "mobile", "value": True}, {"key": "transport_sorted", "value": True}, {"key": "borrowed", "value": True}], 
            failure_conditions=[{"key": "finances.liquidity", "value": 0.0, "op": "le"}],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=14 + difficulty * 2, difficulty=difficulty,
            domain_metadata={"story": "Engine seized on the highway. Car is in the shop for days.", "transport_mode": "car"}
        )

    def generate_rideshare_surge(self, difficulty: int) -> Task:
        routes = [
            Route(id="pay_surge",  name="Pay the Surge Price",       description="Absorb the cost and get there on time.",                     required_action_types=["spend"],                          preconditions={}, consequences={"arrived": True},         closes_routes=["remote"], milestones_unlocked=["m1"], final_reward=2.0),
            Route(id="carpool",    name="Organise a Carpool",         description="Find colleagues or strangers going the same way.",           required_action_types=["communicate", "negotiate"],       preconditions={}, consequences={"carpooled": True},        closes_routes=[],         milestones_unlocked=["m2"], final_reward=3.0),
            Route(id="remote",     name="Present Remotely",           description="Negotiate to dial in instead of attending in person.",       required_action_types=["communicate"],                    preconditions={}, consequences={"remote_approved": True},  closes_routes=["pay_surge"], milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="Arrived at venue on time.",            condition_key="arrived",         condition_value=True, reward=1.5),
            Milestone(id="m2", description="Carpool arranged — zero cost.",         condition_key="carpooled",       condition_value=True, reward=2.0),
            Milestone(id="m3", description="Remote attendance approved.",           condition_key="remote_approved", condition_value=True, reward=1.0),
        ]
        events = [
            ExoEvent(step=1, probability=1.0, id="surge_spike",        description="Surge jumped to 12x. All buses cancelled.",         world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=3, probability=0.9, id="meeting_reminder",   description="Organiser sends a 30-minute warning.",               world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
        ]
        return Task(
            id="rideshare_surge_task", domain="transport_crisis", goal="Get to the Presentation on Time",
            constraints={"budget_max": 200, "deadline_step": 6},
            hidden_state={}, 
            mutable_world={"finances.liquidity": -50.0, "mental_wellbeing.stress_level": 30.0}, 
            visible_world={"finances.liquidity": -50.0, "mental_wellbeing.stress_level": 30.0},
            success_conditions=[{"key": "arrived", "value": True}, {"key": "carpooled", "value": True}, {"key": "remote_approved", "value": True}], 
            failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=8 + difficulty * 2, difficulty=difficulty,
            domain_metadata={"story": "A major city event caused city-wide rideshare surge on your big presentation day.", "transport_mode": "rideshare"}
        )

    def generate_transit_strike(self, difficulty: int) -> Task:
        routes = [
            Route(id="wfh_negotiate",  name="Negotiate Full Remote Week",   description="Get manager approval to WFH for the strike duration.",   required_action_types=["communicate", "negotiate"], preconditions={}, consequences={"wfh_approved": True},        closes_routes=[],                  milestones_unlocked=["m1"], final_reward=3.0),
            Route(id="micromobility",  name="Rent E-Bike / Scooter",         description="Use micro-mobility for the week.",                        required_action_types=["spend"],                    preconditions={}, consequences={"transport_secured": True}, closes_routes=[],                  milestones_unlocked=["m2"], final_reward=2.0),
            Route(id="colleague_crash",name="Crash at a Colleague's Place",  description="Stay near the office temporarily.",                       required_action_types=["communicate"],              preconditions={}, consequences={"accommodation_sorted": True}, closes_routes=[],             milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="WFH approved for the strike period.",  condition_key="wfh_approved",        condition_value=True, reward=2.0),
            Milestone(id="m2", description="Micro-mobility solution in place.",     condition_key="transport_secured",   condition_value=True, reward=1.0),
            Milestone(id="m3", description="Temporary accommodation sorted.",       condition_key="accommodation_sorted",condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=2, probability=0.9, id="strike_extended",   description="Union announces the strike could last 2 weeks.",         world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=5, probability=0.7, id="scooter_shortage",  description="E-bike rental companies sold out in your area.",         world_mutation={}, hidden_state_mutation={}, closes_routes=["micromobility"]),
        ]
        return Task(
            id="transit_strike_task", domain="transport_crisis", goal="Survive City-Wide Transit Strike",
            constraints={"budget_max": 400, "deadline_step": 14},
            hidden_state={}, 
            mutable_world={"time.commute_burden": 50.0, "mental_wellbeing.stress_level": 25.0}, 
            visible_world={"time.commute_burden": 50.0, "mental_wellbeing.stress_level": 25.0},
            success_conditions=[{"key": "wfh_approved", "value": True}, {"key": "transport_secured", "value": True}, {"key": "accommodation_sorted", "value": True}], 
            failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=18 + difficulty * 2, difficulty=difficulty,
            domain_metadata={"story": "All public transport workers walked off the job. The city is gridlocked.", "transport_mode": "transit_strike"}
        )

    def generate_flight_crisis(self, difficulty: int) -> Task:
        routes = [
            Route(id="rebook_premium", name="Rebook Premium Option", description="Call agent and rebook on premium ticket", required_action_types=["communicate", "spend"], preconditions={}, consequences={"flight_rebooked": True}, closes_routes=["wait_lounge"], milestones_unlocked=["m1"], final_reward=2.5),
            Route(id="wait_lounge", name="Accept Delay & Work", description="Stay at airport lounge and work on laptop", required_action_types=["rest", "delegate"], preconditions={}, consequences={"caught_up": True}, closes_routes=["rebook_premium"], milestones_unlocked=["m2"], final_reward=1.8),
        ]
        milestones = [
            Milestone(id="m1", description="Successfully rebooked flight before deadline", condition_key="flight_rebooked", condition_value=True, reward=1.0),
            Milestone(id="m2", description="Caught up with all emergency slack messages", condition_key="caught_up", condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=2, probability=1.0, id="price_surge", description="Ticket prices sharply increased by $300.", world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
            ExoEvent(step=4, probability=1.0, id="lounge_full", description="The airport lounge is now at maximum capacity.", world_mutation={}, hidden_state_mutation={}, closes_routes=["wait_lounge"]),
        ]
        return Task(
            id="flight_crisis_task", domain="flight_crisis", goal="Survive Airport Cancellation",
            constraints={"budget_max": 800, "deadline_step": 10},
            hidden_state={"lounge_capacity": 100}, 
            mutable_world={"mental_wellbeing.stress_level": 25.0, "time.free_hours_per_week": -10.0}, 
            visible_world={"mental_wellbeing.stress_level": 25.0, "time.free_hours_per_week": -10.0},
            success_conditions=[{"key": "flight_rebooked", "value": True}, {"key": "caught_up", "value": True}],
            failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "A major storm grounded commercial flights."}
        )

    def generate_code_merge_crisis(self, difficulty: int) -> Task:
        routes = [
            Route(id="revert_commit", name="Revert Commit", description="Quickly revert the broken merge to unblock the team.", required_action_types=["delegate", "communicate"], preconditions={}, consequences={"pipeline_unblocked": True}, closes_routes=["hotfix"], milestones_unlocked=["unblocked"], final_reward=1.5),
            Route(id="hotfix", name="Patch Forward", description="Find the logic error and push a hotfix.", required_action_types=["communicate", "spend"], preconditions={}, consequences={"bug_resolved": True}, closes_routes=["revert_commit"], milestones_unlocked=["fixed"], final_reward=3.0),
        ]
        milestones = [
            Milestone(id="unblocked", description="CI pipeline is green again", condition_key="pipeline_unblocked", condition_value=True, reward=1.0),
            Milestone(id="fixed", description="Bug resolved without losing features", condition_key="bug_resolved", condition_value=True, reward=2.0),
        ]
        events = [
            ExoEvent(step=3, probability=0.8, id="cto_ping", description="CTO asks for an ETA on the fix.", world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
        ]
        return Task(
            id="code_merge_task", domain="code_merge_crisis", goal="Resolve Production Outage",
            constraints={"budget_max": 1000, "deadline_step": 8},
            hidden_state={}, 
            mutable_world={"career.stability": -20.0, "mental_wellbeing.stress_level": 30.0}, 
            visible_world={"career.stability": -20.0, "mental_wellbeing.stress_level": 30.0},
            success_conditions=[{"key": "pipeline_unblocked", "value": True}, {"key": "bug_resolved", "value": True}], 
            failure_conditions=[],
            event_schedule=events, viable_routes=routes, milestones=milestones,
            horizon=10 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "A botched merge just took down the staging environment."}
        )

    def generate_career(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Negotiate Workload", description="Discuss with manager to reduce workload.", required_action_types=["communicate"], preconditions={}, consequences={"workload_reduced": True}, closes_routes=["r2"], milestones_unlocked=["m1"], final_reward=2.0),
            Route(id="r2", name="Find New Job", description="Start applying for new roles.", required_action_types=["spend", "communicate"], preconditions={}, consequences={"job_found": True}, closes_routes=["r1", "r3"], milestones_unlocked=["m2"], final_reward=3.0),
            Route(id="r3", name="Delegate to Team", description="Push tasks to junior colleagues.", required_action_types=["delegate"], preconditions={}, consequences={"team_delegated": True}, closes_routes=["r2"], milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="Manager agreed to reduce tasks.", condition_key="workload_reduced", condition_value=True, reward=1.0),
            Milestone(id="m2", description="Interview secured.", condition_key="job_found", condition_value=True, reward=1.5),
            Milestone(id="m3", description="Tasks successfully delegated.", condition_key="team_delegated", condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=3, probability=0.7, id="boss_asks", description="Boss asks for progress on current tasks.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="career_crisis", domain="career", goal="Manage Career Overload", constraints={"budget_max": 500, "deadline_step": 12}, 
            hidden_state={}, 
            mutable_world={"career.workload": 30.0, "time.free_hours_per_week": -20.0}, 
            visible_world={"career.workload": 30.0, "time.free_hours_per_week": -20.0}, 
            success_conditions=[{"key": "workload_reduced", "value": True}, {"key": "job_found", "value": True}, {"key": "team_delegated", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "Severe workload is threatening your career stability."}
        )

    def generate_finances(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Emergency Fund", description="Dip into savings.", required_action_types=["spend"], preconditions={}, consequences={"used_emergency": True}, closes_routes=[], milestones_unlocked=["m1"], final_reward=1.0),
            Route(id="r2", name="Negotiate Payment Plan", description="Call the creditor to delay payments.", required_action_types=["communicate"], preconditions={}, consequences={"payment_plan": True}, closes_routes=["r1"], milestones_unlocked=["m2"], final_reward=2.5),
            Route(id="r3", name="Sell Asset", description="Liquidate an asset for quick cash.", required_action_types=["communicate", "spend"], preconditions={}, consequences={"asset_sold": True}, closes_routes=["r2"], milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="Emergency fund accessed.", condition_key="used_emergency", condition_value=True, reward=0.5),
            Milestone(id="m2", description="Favorable payment plan negotiated.", condition_key="payment_plan", condition_value=True, reward=1.0),
            Milestone(id="m3", description="Asset successfully sold.", condition_key="asset_sold", condition_value=True, reward=0.8),
        ]
        events = [
            ExoEvent(step=2, probability=0.9, id="late_fee", description="A late fee was applied to the balance.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="finance_crisis", domain="finances", goal="Resolve Financial Pressure", constraints={"budget_max": 1000, "deadline_step": 10}, 
            hidden_state={}, 
            mutable_world={"finances.liquidity": -40.0, "finances.debt_pressure": 20.0}, 
            visible_world={"finances.liquidity": -40.0, "finances.debt_pressure": 20.0}, 
            success_conditions=[{"key": "used_emergency", "value": True}, {"key": "payment_plan", "value": True}, {"key": "asset_sold", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "An unexpected expense has caused financial strain."}
        )

    def generate_relationships(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Couples Therapy", description="Book a session with a therapist.", required_action_types=["spend", "communicate"], preconditions={}, consequences={"therapy_scheduled": True}, closes_routes=["r3"], milestones_unlocked=["m1"], final_reward=3.0),
            Route(id="r2", name="Honest Conversation", description="Sit down and talk through issues.", required_action_types=["communicate"], preconditions={}, consequences={"had_conversation": True}, closes_routes=[], milestones_unlocked=["m2"], final_reward=2.0),
            Route(id="r3", name="Give Space", description="Take some time apart.", required_action_types=["rest"], preconditions={}, consequences={"giving_space": True}, closes_routes=["r1", "r2"], milestones_unlocked=["m3"], final_reward=1.0),
        ]
        milestones = [
            Milestone(id="m1", description="Therapy session completed.", condition_key="therapy_scheduled", condition_value=True, reward=1.5),
            Milestone(id="m2", description="A productive conversation occurred.", condition_key="had_conversation", condition_value=True, reward=1.0),
            Milestone(id="m3", description="Space given without escalation.", condition_key="giving_space", condition_value=True, reward=0.5),
        ]
        events = [
            ExoEvent(step=4, probability=0.6, id="partner_escalates", description="Partner sends an emotional text msg.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="relationship_crisis", domain="relationships", goal="Repair Relationship Friction", constraints={"budget_max": 800, "deadline_step": 14}, 
            hidden_state={}, 
            mutable_world={"relationships.romantic": -30.0, "mental_wellbeing.stress_level": 20.0}, 
            visible_world={"relationships.romantic": -30.0, "mental_wellbeing.stress_level": 20.0}, 
            success_conditions=[{"key": "therapy_scheduled", "value": True}, {"key": "had_conversation", "value": True}, {"key": "giving_space", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "Growing distance and recent conflicts demand attention."}
        )

    def generate_physical_health(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Medical Leave", description="Request time off to recover.", required_action_types=["communicate", "rest"], preconditions={}, consequences={"on_leave": True}, closes_routes=[], milestones_unlocked=["m1"], final_reward=2.5),
            Route(id="r2", name="See Specialist", description="Pay for a top-tier medical consultation.", required_action_types=["spend", "communicate"], preconditions={}, consequences={"saw_doctor": True}, closes_routes=[], milestones_unlocked=["m2"], final_reward=2.0),
            Route(id="r3", name="Lifestyle Change", description="Commit to better diet and sleep.", required_action_types=["rest"], preconditions={}, consequences={"lifestyle_changed": True}, closes_routes=["r1"], milestones_unlocked=["m3"], final_reward=1.5),
        ]
        milestones = [
            Milestone(id="m1", description="Leave approved.", condition_key="on_leave", condition_value=True, reward=1.0),
            Milestone(id="m2", description="Clear diagnosis received.", condition_key="saw_doctor", condition_value=True, reward=1.0),
            Milestone(id="m3", description="First week of new habits complete.", condition_key="lifestyle_changed", condition_value=True, reward=0.5),
        ]
        events = [
            ExoEvent(step=3, probability=0.8, id="doctor_call", description="The clinic calls with test results.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="health_crisis", domain="physical_health", goal="Address Health Warning", constraints={"budget_max": 1500, "deadline_step": 15}, 
            hidden_state={}, 
            mutable_world={"physical_health.energy": -30.0, "mental_wellbeing.stress_level": 30.0}, 
            visible_world={"physical_health.energy": -30.0, "mental_wellbeing.stress_level": 30.0}, 
            success_conditions=[{"key": "on_leave", "value": True}, {"key": "saw_doctor", "value": True}, {"key": "lifestyle_changed", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "Physical symptoms are becoming impossible to ignore."}
        )

    def generate_mental_wellbeing(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Professional Therapy", description="Start regular therapy sessions.", required_action_types=["spend", "communicate"], preconditions={}, consequences={"therapy_started": True}, closes_routes=[], milestones_unlocked=["m1"], final_reward=3.0),
            Route(id="r2", name="Disconnect", description="Take a full digital detox break.", required_action_types=["rest"], preconditions={}, consequences={"disconnected": True}, closes_routes=["r3"], milestones_unlocked=["m2"], final_reward=1.5),
            Route(id="r3", name="Medication Evaluation", description="See a psychiatrist for options.", required_action_types=["spend"], preconditions={}, consequences={"medication_taken": True}, closes_routes=["r2"], milestones_unlocked=["m3"], final_reward=2.0),
        ]
        milestones = [
            Milestone(id="m1", description="Meaningful breakthrough in therapy.", condition_key="therapy_started", condition_value=True, reward=1.5),
            Milestone(id="m2", description="Successfully unplugged for 48 hours.", condition_key="disconnected", condition_value=True, reward=0.8),
            Milestone(id="m3", description="Prescription acquired.", condition_key="medication_taken", condition_value=True, reward=1.0),
        ]
        events = [
            ExoEvent(step=2, probability=0.5, id="panic_attack", description="A sudden wave of severe anxiety hits.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="mental_crisis", domain="mental_wellbeing", goal="Avert Total Burnout", constraints={"budget_max": 600, "deadline_step": 12}, 
            hidden_state={}, 
            mutable_world={"mental_wellbeing.motivation": -35.0, "mental_wellbeing.stress_level": 40.0}, 
            visible_world={"mental_wellbeing.motivation": -35.0, "mental_wellbeing.stress_level": 40.0}, 
            success_conditions=[{"key": "therapy_started", "value": True}, {"key": "disconnected", "value": True}, {"key": "medication_taken", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "Complete exhaustion and loss of motivation."}
        )

    def generate_time(self, difficulty: int) -> Task:
        routes = [
            Route(id="r1", name="Reprioritize", description="Restructure calendar and say 'no'.", required_action_types=["communicate"], preconditions={}, consequences={"priorities_reset": True}, closes_routes=[], milestones_unlocked=["m1"], final_reward=2.0),
            Route(id="r2", name="Delegate", description="Pay someone or ask for help with chores.", required_action_types=["spend", "delegate"], preconditions={}, consequences={"tasks_delegated": True}, closes_routes=[], milestones_unlocked=["m2"], final_reward=1.5),
            Route(id="r3", name="Cancel Commitments", description="Drop out of major upcoming events.", required_action_types=["communicate"], preconditions={}, consequences={"commitments_cancelled": True}, closes_routes=["r1"], milestones_unlocked=["m3"], final_reward=1.0),
        ]
        milestones = [
            Milestone(id="m1", description="Calendar cleared of non-essentials.", condition_key="priorities_reset", condition_value=True, reward=1.0),
            Milestone(id="m2", description="Help secured for daily tasks.", condition_key="tasks_delegated", condition_value=True, reward=0.8),
            Milestone(id="m3", description="Social obligations cancelled.", condition_key="commitments_cancelled", condition_value=True, reward=0.5),
        ]
        events = [
            ExoEvent(step=3, probability=0.9, id="new_request", description="A friend asks for an 'urgent' favor.", world_mutation={}, hidden_state_mutation={}, closes_routes=[])
        ]
        return Task(
            id="time_crisis", domain="time", goal="Regain Time Control", constraints={"budget_max": 300, "deadline_step": 10}, 
            hidden_state={}, 
            mutable_world={"time.free_hours_per_week": -25.0, "time.admin_overhead": 20.0}, 
            visible_world={"time.free_hours_per_week": -25.0, "time.admin_overhead": 20.0}, 
            success_conditions=[{"key": "priorities_reset", "value": True}, {"key": "tasks_delegated", "value": True}, {"key": "commitments_cancelled", "value": True}],
            failure_conditions=[], event_schedule=events, viable_routes=routes, milestones=milestones, horizon=15 + difficulty * 2, difficulty=difficulty, domain_metadata={"story": "You are double-booked and drowning in obligations."}
        )
