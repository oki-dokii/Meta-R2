"""
scripts/eval.py
---------------
Standalone evaluation runner for the LifeStack environment.

Runs N episodes with a random-action baseline (no model / GPU required) and
prints a summary table plus aggregate statistics.

Usage:
    python scripts/eval.py
    python scripts/eval.py --episodes 20
    python scripts/eval.py --episodes 20 --domain flight_crisis --verbose
"""

import argparse
import random
import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.lifestack_env import LifeStackEnv, LifeStackAction
from agent.conflict_generator import TaskGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All action_types understood by the env's tool dispatch.
_ACTION_TYPES = ["execute", "inspect", "plan", "wait", "communicate", "spend", "delegate"]

# Known route IDs across the two TaskGenerator domains — used for targeted
# "execute" actions so we occasionally hit real routes.
_KNOWN_ROUTE_IDS = [
    "rebook_premium", "wait_lounge",        # flight_crisis
    "revert_commit", "hotfix",              # code_merge_crisis
]


def _random_action(task) -> LifeStackAction:
    """Return a random LifeStackAction that exercises a variety of tool types."""
    action_type = random.choice(_ACTION_TYPES)

    # For "execute" actions, attempt to target a known route from the task.
    target = None
    if action_type == "execute":
        route_ids = [r.id for r in task.viable_routes] if task and task.viable_routes else _KNOWN_ROUTE_IDS
        target = random.choice(route_ids)
    elif action_type == "inspect":
        # Pick a random hidden-state key from the task or fall back to a default.
        if task and task.hidden_state:
            target = random.choice(list(task.hidden_state.keys()))
        else:
            target = "lounge_capacity"

    # Small, random metric nudges to keep the episode non-trivial.
    metric_changes: dict = {}
    if action_type in ("execute", "plan", "communicate"):
        domain = random.choice(
            ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"]
        )
        sub_key = random.choice(["workload", "stress_level", "liquidity", "sleep_quality", "energy", "free_hours_per_week"])
        metric_changes[f"{domain}.{sub_key}"] = random.uniform(-10.0, 10.0)

    resource_cost: dict = {}
    if action_type != "wait":
        resource_cost = {
            "time":   random.uniform(0.0, 2.0),
            "money":  random.uniform(0.0, 50.0),
            "energy": random.uniform(0.0, 10.0),
        }

    return LifeStackAction(
        action_type=action_type,
        target=target,
        metric_changes=metric_changes,
        resource_cost=resource_cost,
        actions_taken=1,
        reasoning="random baseline",
    )


def _row(ep_id: int, total_reward: float, steps: int, domain: str, success: bool) -> str:
    """Format one summary table row."""
    success_str = "✓" if success else "✗"
    return (
        f"  {ep_id:>4}  "
        f"{total_reward:>12.4f}  "
        f"{steps:>6}  "
        f"{domain:<20}  "
        f"{success_str:>7}"
    )


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_eval(n_episodes: int, domain: str | None, verbose: bool) -> None:
    generator = TaskGenerator()
    env = LifeStackEnv()

    results = []

    header = (
        f"\n  {'EP':>4}  {'TOTAL REWARD':>12}  {'STEPS':>6}  {'DOMAIN':<20}  {'SUCCESS':>7}\n"
        f"  {'─'*4}  {'─'*12}  {'─'*6}  {'─'*20}  {'─'*7}"
    )
    print(header)

    for ep in range(1, n_episodes + 1):
        # Generate task (optionally filtered by domain).
        task = generator.generate(domain=domain)

        obs = env.reset(task=task, episode_id=str(ep))

        total_reward = 0.0
        steps = 0
        success = False

        while not obs.done:
            action = _random_action(env.state.current_task)
            obs = env.step(action)
            reward = obs.reward or 0.0
            total_reward += reward
            steps += 1

            if verbose:
                print(
                    f"    step={steps:>3}  reward={reward:+.3f}  "
                    f"action={action.action_type:<12}  "
                    f"target={str(action.target):<20}  "
                    f"done={obs.done}"
                )

            if obs.metadata.get("success"):
                success = True

        task_domain = task.domain if task else "unknown"
        results.append(
            {
                "episode": ep,
                "total_reward": total_reward,
                "steps": steps,
                "domain": task_domain,
                "success": success,
            }
        )

        print(_row(ep, total_reward, steps, task_domain, success))

    # -----------------------------------------------------------------------
    # Aggregate stats
    # -----------------------------------------------------------------------
    n = len(results)
    mean_reward = sum(r["total_reward"] for r in results) / n if n else 0.0
    success_rate = sum(1 for r in results if r["success"]) / n if n else 0.0
    mean_steps = sum(r["steps"] for r in results) / n if n else 0.0

    print(
        f"\n  {'─'*60}\n"
        f"  Episodes     : {n}\n"
        f"  Mean Reward  : {mean_reward:.4f}\n"
        f"  Success Rate : {success_rate:.1%}\n"
        f"  Mean Steps   : {mean_steps:.1f}\n"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeStack environment evaluation runner (random baseline)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help=(
            "Optional domain filter passed to TaskGenerator.generate(). "
            "Supported: 'flight_crisis', 'code_merge_crisis'. "
            "Omit to cycle randomly."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-step details for every episode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        f"LifeStack Eval — episodes={args.episodes}  "
        f"domain={args.domain or 'any'}  "
        f"verbose={args.verbose}"
    )
    run_eval(n_episodes=args.episodes, domain=args.domain, verbose=args.verbose)
