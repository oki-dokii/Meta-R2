"""
run_episode.py — LifeStack Full Episode Runner

Orchestrates a complete episode:
  1. Generate a conflict
  2. Initialize environment, agent, person, and memory
  3. Loop up to 5 steps: agent decides → action applied → reward computed → memory updated
  4. Print a rich episode summary at the end
"""

import random
from life_state import LifeMetrics, ResourceBudget
from lifestack_env import LifeStackEnv
from agent import LifeStackAgent
from simperson import SimPerson
from conflict_generator import generate_conflict, escalate_conflict
from action_space import apply_action, validate_action
from memory import LifeStackMemory
from reward import compute_reward
import copy


def run_episode(
    difficulty: int = None,
    verbose: bool = True,
    memory: "LifeStackMemory" = None,
    agent: "LifeStackAgent" = None,
) -> dict:
    """
    Runs one full LifeStack episode.

    Args:
        memory: Optional shared LifeStackMemory instance (avoids re-loading the
                sentence-transformer model on every episode).
        agent:  Optional shared LifeStackAgent instance (avoids re-creating the
                Groq client on every episode).

    Returns:
        summary dict with total_reward, steps, final_metrics, conflicts_seen
    """
    # --------------------------------------------------
    # 1. SETUP
    # --------------------------------------------------
    env = LifeStackEnv()
    if agent is None:
        agent = LifeStackAgent()
    if memory is None:
        memory = LifeStackMemory()

    # Pick a SimPerson from a diverse pool
    person_pool = [
        SimPerson(name="Alex (Executive)",    openness=0.4, conscientiousness=0.9, extraversion=0.7,  agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe (Creative)",    openness=0.9, conscientiousness=0.2, extraversion=0.5,  agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam (Introvert)",     openness=0.5, conscientiousness=0.6, extraversion=0.1,  agreeableness=0.65, neuroticism=0.9),
        SimPerson(name="Maya (Family)",       openness=0.5, conscientiousness=0.7, extraversion=0.5,  agreeableness=0.95, neuroticism=0.3),
        SimPerson(name="Leo (Student)",       openness=0.85,conscientiousness=0.8, extraversion=0.4,  agreeableness=0.4,  neuroticism=0.55),
    ]
    person = random.choice(person_pool)

    # Generate starting conflict
    conflict = generate_conflict(difficulty)

    # Apply initial disruption to env
    obs = env.reset(conflict=conflict.primary_disruption)

    # --------------------------------------------------
    # 2. EPISODE LOOP
    # --------------------------------------------------
    total_reward = 0.0
    step_log = []
    conflicts_seen = [conflict.title]

    if verbose:
        print("\n" + "◆" * 60)
        print(f"  LIFESTACK EPISODE — {conflict.title}")
        print(f"  Person  : {person.name}")
        print(f"  Hint    : {person.get_personality_hint()}")
        print(f"  Story   : {conflict.story}")
        print("◆" * 60)
        env.render()

    while not obs["done"]:
        step = obs["step"]

        # Personality drift every 5 steps
        drift_event = person.drift(step)
        if drift_event:
            if verbose:
                print(f"\n[DRIFT] {drift_event['reason']}")
            # Apply drift event to metrics
            path = drift_event.get('metric', '')
            delta = drift_event.get('delta', 0)
            if path and '.' in path:
                dom, sub = path.split('.')
                current = getattr(getattr(env.state, dom), sub)
                setattr(getattr(env.state, dom), sub, max(0.0, min(100.0, current + delta)))

        # Randomly escalate on step 3 if difficulty < 5
        if step == 2 and conflict.difficulty < 5 and random.random() < 0.4:
            conflict = escalate_conflict(conflict)
            conflicts_seen.append(conflict.title)
            if verbose:
                print(f"\n🔥 ESCALATION! New conflict: {conflict.title}")

        # Inject few-shot context into agent memory
        few_shot = memory.build_few_shot_prompt(conflict.title, env.state.flatten())

        # Agent decision
        metrics_before = copy.deepcopy(env.state)
        budget_before = copy.deepcopy(env.budget)

        action = agent.get_action(env.state, env.budget, conflict, person)

        # Validate resource cost
        is_valid, reason = validate_action(action, env.budget)
        if not is_valid:
            if verbose:
                print(f"\n  ⚠️  Step {step+1}: Action unaffordable ({reason}). Forcing rest.")
            action.primary.metric_changes = {"mental_wellbeing.stress_level": -3.0}
            action.primary.resource_cost = {}

        # Scale metric changes by personality uptake
        current_stress = env.state.mental_wellbeing.stress_level
        uptake_score = person.respond_to_action(
            action.primary.action_type, 
            action.primary.resource_cost, 
            current_stress
        )
        scaled_changes = {}
        # Make sure that path format is 'domain.submetric'
        for path, delta in action.primary.metric_changes.items():
            if '.' not in path: # Prepend target_domain if the LLM forgot it
                path = f"{action.primary.target_domain}.{path}"
            # ensure float conversion just in case LLM put strings
            try:
                scaled_changes[path] = float(delta) * uptake_score
            except ValueError:
                pass

        # Apply action through environment
        env_action = {
            "metric_changes": scaled_changes,
            "resource_cost": action.primary.resource_cost,
            "actions_taken": 1  # any deliberate action counts
        }
        obs = env.step(env_action)
        step_reward = obs["reward"]
        total_reward += step_reward

        # Store high-quality decisions in memory
        agent.store_decision(action, step_reward)
        memory.store_decision(
            conflict.title,
            action.primary.action_type,
            action.primary.target_domain,
            step_reward,
            env.state.flatten(),
            action.reasoning
        )

        # Log the step
        step_log.append({
            "step": step + 1,
            "action": action.primary.action_type,
            "domain": action.primary.target_domain,
            "description": action.primary.description,
            "reward": round(step_reward, 3),
            "penalties": obs["breakdown"]["penalties_fired"]
        })

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  STEP {step+1} → {action.primary.action_type.upper()} on {action.primary.target_domain}")
            print(f"  \"{action.primary.description}\"")
            if action.communication:
                print(f"  💬 [{action.communication.recipient}] ({action.communication.tone}): {action.communication.content}")
            print(f"  Reward: {step_reward:.3f} | Penalties: {obs['breakdown']['penalties_fired'] or 'none'}")
            env.render()

    # --------------------------------------------------
    # 3. EPISODE SUMMARY
    # --------------------------------------------------
    final_flat = env.state.flatten()
    critical = [k for k, v in final_flat.items() if v < 20]
    improved = [k for k, v in final_flat.items() if v > 70]
    mem_stats = memory.get_stats()

    if verbose:
        print("\n" + "█" * 60)
        print("  EPISODE COMPLETE — FINAL SUMMARY")
        print("█" * 60)
        print(f"  Person         : {person.name}")
        print(f"  Conflicts Seen : {' → '.join(conflicts_seen)}")
        print(f"  Steps Taken    : {env.step_count}")
        print(f"  Total Reward   : {total_reward:.4f}")
        print(f"  Critical (<20) : {critical or 'None'}")
        print(f"  Thriving (>70) : {len(improved)} metrics")
        print(f"\n  Step-by-Step Log:")
        for s in step_log:
            flag = " ⚠️ " if s["penalties"] else "  ✅"
            print(f"  {flag} Step {s['step']}: [{s['action']}] on {s['domain']} → {s['reward']:.3f}")
        print(f"\n  Memory Bank    : {mem_stats['total_memories']} decisions stored (avg reward: {mem_stats['average_reward']})")
        print("█" * 60)

    return {
        "person": person.name,
        "total_reward": round(total_reward, 4),
        "steps": env.step_count,
        "conflicts_seen": conflicts_seen,
        "critical_metrics": critical,
        "thriving_count": len(improved),
        "step_log": step_log,
        "memory_stats": mem_stats
    }


if __name__ == "__main__":
    # Run one episode at each difficulty level for demonstration
    for d in [2, 3, 5]:
        print(f"\n{'═'*60}")
        print(f"  STARTING EPISODE AT DIFFICULTY {d}")
        print(f"{'═'*60}")
        summary = run_episode(difficulty=d, verbose=True)
        print(f"\n  → Total Reward: {summary['total_reward']}")
