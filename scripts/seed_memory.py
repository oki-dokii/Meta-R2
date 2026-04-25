"""
Synthetic Memory Seeder
-----------------------
Generates and solves N synthetic life scenarios, storing only high-reward
decisions (reward >= MIN_REWARD) into ChromaDB. Run this once to pre-populate
the memory library so the warm-start agent already acts like a "pro".

Usage:
    python scripts/seed_memory.py               # 200 scenarios, fast mode
    python scripts/seed_memory.py --n 1000      # 1000 scenarios
    python scripts/seed_memory.py --n 50 --verbose
    python scripts/seed_memory.py --stats       # just print current DB stats
"""

import sys
import os
import argparse
import random
import copy
import time

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.conflict_generator import generate_conflict, TEMPLATES
from agent.memory import LifeStackMemory
from agent.agent import LifeStackAgent
from core.lifestack_env import LifeStackEnv, LifeStackAction
from core.life_state import LifeMetrics, ResourceBudget
from intake.simperson import SimPerson
from core.metric_schema import normalize_metric_path, is_valid_metric_path

# ── Config ────────────────────────────────────────────────────────────────────
MIN_REWARD = 0.05          # Store decisions at or above this threshold (env reward range: -1.0 to 1.0)
RATE_LIMIT_SLEEP = 2.5     # Seconds between Groq API calls — 30 RPM limit = 2.0s minimum, 2.5s with buffer
MAX_RETRIES = 2            # Per scenario before skipping

# ── Diverse persona pool ──────────────────────────────────────────────────────
PERSONA_POOL = [
    SimPerson(name="Alex (Executive)",   openness=0.4,  conscientiousness=0.9, extraversion=0.7,  agreeableness=0.25, neuroticism=0.8),
    SimPerson(name="Chloe (Creative)",   openness=0.9,  conscientiousness=0.2, extraversion=0.5,  agreeableness=0.70, neuroticism=0.15),
    SimPerson(name="Sam (Introvert)",    openness=0.5,  conscientiousness=0.6, extraversion=0.1,  agreeableness=0.65, neuroticism=0.9),
    SimPerson(name="Maya (Family)",      openness=0.5,  conscientiousness=0.7, extraversion=0.5,  agreeableness=0.95, neuroticism=0.3),
    SimPerson(name="Leo (Student)",      openness=0.85, conscientiousness=0.8, extraversion=0.4,  agreeableness=0.4,  neuroticism=0.55),
    SimPerson(name="Arjun (Startup)",    openness=0.4,  conscientiousness=0.9, extraversion=0.7,  agreeableness=0.25, neuroticism=0.8),
    # Extra synthetic personas for diversity
    SimPerson(name="Dana (Retiree)",     openness=0.3,  conscientiousness=0.75, extraversion=0.35, agreeableness=0.8,  neuroticism=0.2),
    SimPerson(name="Kai (Freelancer)",   openness=0.8,  conscientiousness=0.3,  extraversion=0.6,  agreeableness=0.5,  neuroticism=0.6),
    SimPerson(name="Priya (Academic)",   openness=0.85, conscientiousness=0.85, extraversion=0.3,  agreeableness=0.6,  neuroticism=0.45),
    SimPerson(name="Marcus (Athlete)",   openness=0.45, conscientiousness=0.95, extraversion=0.65, agreeableness=0.5,  neuroticism=0.3),
]


def _normalize_metric_changes(metric_changes: dict, target_domain: str) -> dict:
    fixed = {}
    for path, delta in metric_changes.items():
        raw = str(path)
        if "." not in raw:
            raw = f"{target_domain}.{raw}"
        norm = normalize_metric_path(raw)
        if not is_valid_metric_path(norm):
            continue
        try:
            fixed[norm] = float(delta)
        except (ValueError, TypeError):
            continue
    return fixed


def run_one_scenario(agent: LifeStackAgent, memory: LifeStackMemory, conflict, person: SimPerson, verbose: bool) -> dict | None:
    """Run a single conflict+persona pair. Returns stored record or None if below threshold."""
    try:
        env = LifeStackEnv()
        env.reset(conflict=conflict.primary_disruption, budget=conflict.resource_budget)
        before_metrics = copy.deepcopy(env.state.current_metrics)
        before_budget = copy.deepcopy(env.state.budget)

        action = agent.get_action(before_metrics, before_budget, conflict, person)

        # Normalize metric changes
        action.primary.metric_changes = _normalize_metric_changes(
            action.primary.metric_changes, action.primary.target_domain
        )

        uptake = person.respond_to_action(
            action.primary.action_type,
            action.primary.resource_cost,
            before_metrics.mental_wellbeing.stress_level,
        )
        env_action = LifeStackAction.from_agent_action(action)
        env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
        obs = env.step(env_action)

        reward = obs.reward

        if reward >= MIN_REWARD:
            # Build a compact metrics diff string for the memory record
            flat_before = before_metrics.flatten()
            flat_after  = obs.metrics if isinstance(obs.metrics, dict) else {}
            changed = {
                k: round(flat_after.get(k, flat_before[k]) - flat_before[k], 1)
                for k in flat_before
                if abs(flat_after.get(k, flat_before[k]) - flat_before[k]) > 0.5
            }
            metrics_diff_str = ", ".join(f"{k}:{'+' if v > 0 else ''}{v}" for k, v in list(changed.items())[:5])

            memory.store_decision(
                conflict_title=conflict.title,
                action_type=action.primary.action_type,
                target_domain=action.primary.target_domain,
                reward=reward,
                metrics_snapshot=flat_before,
                reasoning=action.reasoning,
                route_outcome=f"{action.primary.action_type}→{action.primary.target_domain}",
            )
            # Also store as trajectory so retrieve_similar_trajectories works
            memory.store_trajectory(
                conflict_title=conflict.title,
                route_taken=f"{action.primary.action_type}→{action.primary.target_domain}",
                total_reward=reward,
                metrics_diff_str=metrics_diff_str,
                reasoning=action.reasoning,
            )

            if verbose:
                print(f"  STORED  [{action.primary.action_type:12}→{action.primary.target_domain:20}] reward={reward:.3f}  ({conflict.title} / {person.name})")
            return {"reward": reward, "stored": True}
        else:
            if verbose:
                print(f"  SKIP    [{action.primary.action_type:12}→{action.primary.target_domain:20}] reward={reward:.3f}  (below {MIN_REWARD})")
            return {"reward": reward, "stored": False}

    except Exception as e:
        if verbose:
            print(f"  ERROR   {conflict.title} / {person.name}: {e}")
        return None


def seed(n: int, verbose: bool, api_only: bool):
    print(f"\n{'='*60}")
    print(f"  LifeStack Synthetic Memory Seeder")
    print(f"  Target: {n} scenarios | Min reward: {MIN_REWARD}")
    print(f"{'='*60}\n")

    memory = LifeStackMemory(silent=not verbose)
    agent  = LifeStackAgent(api_only=api_only)

    start_count = memory.collection.count()
    print(f"ChromaDB: {start_count} existing memories\n")

    stored   = 0
    skipped  = 0
    errors   = 0
    t_start  = time.time()

    # Build a weighted scenario list: more hard conflicts (difficulty 3-5) since those
    # produce richer reasoning and more useful precedents for the RAG system.
    difficulty_weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.25, 5: 0.15}
    all_difficulties = [1, 2, 3, 4, 5]

    for i in range(n):
        # Pick difficulty by weight
        diff = random.choices(
            all_difficulties,
            weights=[difficulty_weights[d] for d in all_difficulties]
        )[0]
        conflict = generate_conflict(difficulty=diff)
        person   = random.choice(PERSONA_POOL)

        if not verbose:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta  = (n - i - 1) / rate if rate > 0 else 0
            print(
                f"\r  [{i+1:>4}/{n}] stored={stored} skipped={skipped} errors={errors}"
                f"  rate={rate:.1f}/s  ETA={eta:.0f}s   ",
                end="", flush=True
            )

        result = None
        for attempt in range(MAX_RETRIES):
            result = run_one_scenario(agent, memory, conflict, person, verbose)
            if result is not None:
                break
            time.sleep(1.5)

        if result is None:
            errors += 1
        elif result["stored"]:
            stored += 1
        else:
            skipped += 1

        time.sleep(RATE_LIMIT_SLEEP)

    elapsed = time.time() - t_start
    end_count = memory.collection.count()

    print(f"\n\n{'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Scenarios run : {n}")
    print(f"  Stored        : {stored}  (reward >= {MIN_REWARD})")
    print(f"  Skipped       : {skipped} (below threshold)")
    print(f"  Errors        : {errors}")
    print(f"  DB size       : {start_count} → {end_count} memories")
    print(f"{'='*60}\n")

    stats = memory.get_stats()
    print(f"  Avg reward in DB : {stats['average_reward']:.3f}")
    print(f"  By action type   : {stats.get('by_action_type', {})}")


def print_stats():
    memory = LifeStackMemory(silent=True)
    stats  = memory.get_stats()
    print(f"\nChromaDB Memory Stats")
    print(f"  Total memories : {stats['total_memories']}")
    print(f"  Average reward : {stats['average_reward']:.3f}")
    print(f"  By action type : {stats.get('by_action_type', {})}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ChromaDB with synthetic life scenario memories")
    parser.add_argument("--n",        type=int,  default=200,  help="Number of scenarios to run (default: 200)")
    parser.add_argument("--verbose",  action="store_true",     help="Print each decision")
    parser.add_argument("--stats",    action="store_true",     help="Just print current DB stats and exit")
    parser.add_argument("--api-only", action="store_true",     help="Force Groq API (no local model)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        seed(n=args.n, verbose=args.verbose, api_only=args.api_only)
