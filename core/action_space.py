import copy
from dataclasses import dataclass, field
from core.life_state import LifeMetrics, ResourceBudget
from enum import Enum
from intake.simperson import SimPerson

class ToolActionType(str, Enum):
    INSPECT = "inspect"
    PLAN = "plan"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"
    WAIT = "wait"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"

@dataclass
class ToolAction:
    action_type: ToolActionType
    target: str = ""          # inspect target, execute target, communicate recipient, etc.
    parameters: dict = field(default_factory=dict)
    reasoning: str = ""

@dataclass
class PrimaryAction:
    action_type: str  # reschedule, delegate, negotiate, spend, communicate, rest, deprioritize
    target_domain: str
    metric_changes: dict
    resource_cost: dict
    description: str

@dataclass
class CommunicationAction:
    recipient: str  # boss, partner, family, friend, colleague
    message_type: str  # apologize, negotiate, inform, request, reassure
    tone: str  # formal, warm, urgent, calm, assertive
    content: str

@dataclass
class AgentAction:
    primary: PrimaryAction
    communication: CommunicationAction = None
    reasoning: str = ""

def validate_action(action: AgentAction, budget: ResourceBudget) -> tuple[bool, str]:
    cost = action.primary.resource_cost
    if budget.time_hours < cost.get('time', 0.0):
        return False, f"Not enough time (Needs {cost.get('time')}h, has {budget.time_hours:.1f}h)"
    if budget.money_dollars < cost.get('money', 0.0):
        return False, f"Not enough money (Needs ${cost.get('money')}, has ${budget.money_dollars:.1f})"
    if budget.energy_units < cost.get('energy', 0.0):
        return False, f"Not enough energy (Needs {cost.get('energy')}u, has {budget.energy_units:.1f}u)"
    return True, ""

def validate_tool_action(action: ToolAction, env_state: dict) -> tuple[bool, str]:
    """
    Checks logic for tool actions.
    env_state should contain: inspected_keys (list), consecutive_waits (int), used_rollback (bool)
    """
    atype = action.action_type
    if atype == ToolActionType.INSPECT:
        if action.target in env_state.get('inspected_keys', []):
            return False, f"Already inspected {action.target}."
    
    if atype == ToolActionType.WAIT:
        if env_state.get('consecutive_waits', 0) >= 3:
            return False, "Max consecutive waits (3) reached. Must act or escalate."
            
    if atype == ToolActionType.ROLLBACK:
        if env_state.get('used_rollback', False):
            return False, "Rollback already used in this episode."
            
    return True, ""

def apply_action(action: AgentAction, metrics: LifeMetrics, budget: ResourceBudget, person: SimPerson) -> tuple[LifeMetrics, ResourceBudget, float]:
    """Validates, scales by personality uptake, and applies the action to the state."""
    
    # 1. Validation
    is_valid, reason = validate_action(action, budget)
    if not is_valid:
        # If invalid, the action fails but we return current state with 0 uptake
        return metrics, budget, 0.0
    
    # 2. Personality Scaling (Uptake)
    current_stress = metrics.mental_wellbeing.stress_level
    uptake_score = person.respond_to_action(
        action.primary.action_type, 
        action.primary.resource_cost, 
        current_stress
    )
    
    # 3. Apply changes (Scaled by uptake)
    new_metrics = copy.deepcopy(metrics)
    for path, delta in action.primary.metric_changes.items():
        # Guard: skip malformed keys without a domain prefix (e.g. LLM returns "stress_level" instead of "mental_wellbeing.stress_level")
        if '.' not in path:
            print(f"  ⚠️  Skipping malformed metric key: '{path}' (expected 'domain.submetric')")
            continue
        parts = path.split('.', 1)
        domain_name, sub_name = parts[0], parts[1]
        domain = getattr(new_metrics, domain_name, None)
        if domain is None or not hasattr(domain, sub_name):
            print(f"  ⚠️  Skipping unknown metric: '{path}'")
            continue
        current = getattr(domain, sub_name)
        
        # Scale the benefit/cost by the person's receptiveness
        try:
            scaled_delta = float(delta) * uptake_score
            setattr(domain, sub_name, max(0.0, min(100.0, current + scaled_delta)))
        except ValueError:
            print(f"  ⚠️  Skipping metric change due to invalid delta value: '{delta}'")
        
    # 4. Deduct resources (Fixed cost, doesn't scale with uptake)
    new_budget = copy.deepcopy(budget)
    new_budget.deduct(
        time=action.primary.resource_cost.get('time', 0.0),
        money=action.primary.resource_cost.get('money', 0.0),
        energy=action.primary.resource_cost.get('energy', 0.0)
    )
    
    return new_metrics, new_budget, uptake_score

# 10 EXAMPLE ACTIONS for Friday 6PM Conflict
EXAMPLE_ACTIONS = [
    AgentAction(
        primary=PrimaryAction(
            action_type="negotiate", target_domain="career",
            metric_changes={"career.workload": -15.0, "mental_wellbeing.stress_level": -5.0},
            resource_cost={"time": 1.5, "energy": 20.0},
            description="Negotiate a Sunday deadline extension with my boss."
        ),
        communication=CommunicationAction("boss", "negotiate", "formal", "Due to flight issues, I need until Sunday PM for the report."),
        reasoning="Relieving the immediate workload pressure is critical to reduce cascade spread."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="spend", target_domain="finances",
            metric_changes={"finances.liquidity": -350.0, "mental_wellbeing.stress_level": -10.0},
            resource_cost={"time": 1.0, "energy": 15.0},
            description="Rebook the canceled flight using a premium fare."
        ),
        reasoning="Immediate resolution of logistics fixes the source of the crisis."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="communicate", target_domain="relationships",
            metric_changes={"relationships.romantic": 12.0, "mental_wellbeing.stress_level": -5.0},
            resource_cost={"time": 0.5, "energy": 10.0},
            description="Call my partner to explain the situation and reassure them."
        ),
        communication=CommunicationAction("partner", "reassure", "warm", "Hey, I'm stuck but I'll be home soon. Miss you."),
        reasoning="Prevents relationship decay while stress is high."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="communicate", target_domain="finances",
            metric_changes={"finances.liquidity": 200.0, "relationships.family": -5.0},
            resource_cost={"time": 1.5, "energy": 25.0},
            description="Ask my sibling for a temporary loan to cover rebooking."
        ),
        communication=CommunicationAction("family", "request", "urgent", "My card declined, can you Venmo me $200 for the flight?"),
        reasoning="Fixes the liquidity block at a small social cost."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="reschedule", target_domain="time",
            metric_changes={"career.workload": -10.0, "time.free_hours_per_week": 5.0},
            resource_cost={"time": 2.0, "energy": 15.0},
            description="Cancel non-essential meetings to create a deep-work block."
        ),
        reasoning="Regaining time allows for better problem solving later."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="rest", target_domain="physical_health",
            metric_changes={"mental_wellbeing.stress_level": -12.0, "physical_health.energy": 10.0},
            resource_cost={"time": 1.0, "energy": -10.0},
            description="Take a 60-minute power nap in the airport lounge."
        ),
        reasoning="Restores energy to tackle the remaining Sunday deadline."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="delegate", target_domain="career",
            metric_changes={"career.workload": -10.0, "relationships.professional_network": -5.0},
            resource_cost={"time": 1.0, "energy": 15.0},
            description="Ask a colleague to handle the final formatting of the slides."
        ),
        communication=CommunicationAction("colleague", "request", "assertive", "I'm stuck at airport, can you finish the formatting?"),
        reasoning="Reduces workload by leaning on the professional network."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="deprioritize", target_domain="time",
            metric_changes={"time.free_hours_per_week": 8.0, "relationships.social": -10.0},
            resource_cost={"time": 0.5, "energy": 5.0},
            description="Tell friends I can't attend the weekend gathering."
        ),
        communication=CommunicationAction("friend", "inform", "calm", "Hey, work crisis. Won't make it this weekend. Sorry!"),
        reasoning="Aggressively reclaims time for high-value tasks."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="communicate", target_domain="career",
            metric_changes={"career.stability": 8.0, "mental_wellbeing.stress_level": -5.0},
            resource_cost={"time": 0.5, "energy": 10.0},
            description="Send an apology note to boss for the delay."
        ),
        communication=CommunicationAction("boss", "apologize", "formal", "Apologies for the delay caused by travel disruptions. On it now."),
        reasoning="Maintains career stability during an active crisis."
    ),
    AgentAction(
        primary=PrimaryAction(
            action_type="reschedule", target_domain="finances",
            metric_changes={"finances.debt_pressure": -10.0, "time.admin_overhead": 10.0},
            resource_cost={"time": 2.0, "energy": 15.0},
            description="Call the bank to unlock the declined card."
        ),
        communication=CommunicationAction("colleague", "request", "assertive", "Unlock my credit card immediately."),
        reasoning="Removes the liquidity barrier by handling admin overhead."
    )
]

def main():
    # 1. Setup Personalities
    # Sam (Anxious Introvert): Neuroticism 0.9, Extraversion 0.1
    sam = SimPerson(name="Sam (Introvert)", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.9)
    
    # 2. Setup initial state (Friday 6PM Conflict)
    from core.life_state import DependencyGraph
    graph = DependencyGraph()
    metrics = LifeMetrics() # starts at 70s
    metrics = graph.cascade(metrics, {"career.workload": 35.0, "finances.liquidity": -40.0})
    budget = ResourceBudget(time_hours=20.0, money_dollars=500.0, energy_units=100.0)
    
    print("--- SIMULATING ACTIONS FOR SAM (ANXIOUS INTROVERT) ---")
    print(f"Initial Stress: {metrics.mental_wellbeing.stress_level:.2f}")
    print(f"Initial Metrics Health (Avg): {sum(metrics.flatten().values())/23:.2f}")
    
    # 3. Apply each action
    for i, action in enumerate(EXAMPLE_ACTIONS, 1):
        print(f"\nACTION {i}: {action.primary.description}")
        
        is_valid, reason = validate_action(action, budget)
        if not is_valid:
            print(f"  ❌ FAILED: {reason}")
            continue
            
        m_after, b_after, uptake = apply_action(action, metrics, budget, sam)
        
        print(f"  ✅ SUCCESS | Uptake: {uptake:.2f}")
        print(f"  Cost: {action.primary.resource_cost}")
        
        # Show specific improvements
        for path, delta in action.primary.metric_changes.items():
            domain_name, sub_name = path.split('.')
            val_before = getattr(getattr(metrics, domain_name), sub_name)
            val_after = getattr(getattr(m_after, domain_name), sub_name)
            real_delta = val_after - val_before
            print(f"  - {path:25}: {val_before:.2f} -> {val_after:.2f} (Actual Change: {real_delta:+.2f})")

if __name__ == "__main__":
    main()
