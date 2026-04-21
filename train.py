"""
train.py — LifeStack Training Loop

Runs a curriculum of episodes at increasing difficulty, logs rewards,
generates a learning curve plot, and compares agent performance
before and after memory accumulation.
"""

import json
import os
import random
import shutil
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

from run_episode import run_episode
from memory import LifeStackMemory
from agent import LifeStackAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _difficulty_for_episode(episode: int) -> int:
    """Curriculum schedule: easy → medium → hard → extreme."""
    if episode <= 25:
        return random.randint(1, 2)
    elif episode <= 50:
        return random.randint(2, 3)
    elif episode <= 75:
        return random.randint(3, 4)
    else:
        return random.randint(4, 5)


def _rolling_avg(values: list, window: int = 5) -> list:
    """Compute a simple rolling average with the given window."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def _phase_avg(rewards: list, start: int, end: int) -> float:
    """Average reward for 1-indexed episodes [start, end]."""
    subset = rewards[start - 1 : end]
    return round(sum(subset) / len(subset), 3) if subset else 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training(n_episodes: int = 50, save_plot: bool = True) -> dict:
    """
    Runs the full LifeStack curriculum training loop.

    Returns:
        summary dict with per-episode logs and phase averages.
    """
    episode_log = []
    rewards = []

    print(f"\n{'═' * 50}")
    print(f"  LIFESTACK TRAINING — {n_episodes} EPISODES")
    print(f"{'═' * 50}\n")

    # Initialize shared instances once — avoids reloading model weights each episode
    print("  Initializing shared agent and memory (one-time load)...")
    shared_memory = LifeStackMemory(silent=True)  # suppress per-decision spam
    shared_agent  = LifeStackAgent()
    print("  ✅ Ready.\n")

    for ep in range(1, n_episodes + 1):
        difficulty = _difficulty_for_episode(ep)

        # Run episode with shared memory + agent
        result = run_episode(difficulty=difficulty, verbose=False,
                             memory=shared_memory, agent=shared_agent)

        total_reward = result["total_reward"]
        rewards.append(total_reward)

        record = {
            "episode": ep,
            "reward": total_reward,
            "difficulty": difficulty,
            "person": result["person"],
            "conflicts_seen": result["conflicts_seen"],
            "steps": result["steps"],
        }
        episode_log.append(record)

        # Progress: print every episode
        mem_count = result["memory_stats"]["total_memories"]
        print(
            f"  Episode {ep:>3}/{n_episodes} | "
            f"Reward: {total_reward:.3f} | "
            f"Difficulty: {difficulty} | "
            f"Memories: {mem_count}"
        )

    # ------------------------------------------------------------------
    # Phase averages
    # ------------------------------------------------------------------
    early_avg = _phase_avg(rewards, 1, 25)
    mid_avg   = _phase_avg(rewards, 26, 50)
    late_avg  = _phase_avg(rewards, 51, 75)
    final_avg = _phase_avg(rewards, 76, n_episodes)
    overall   = round(sum(rewards) / len(rewards), 3)

    print(f"\n{'═' * 42}")
    print(f"  TRAINING SUMMARY")
    print(f"{'═' * 42}")
    print(f"  {'Phase':<10} {'Episodes':<12} {'Avg Reward'}")
    print(f"  {'-'*38}")
    print(f"  {'Early':<10} {'1-25':<12} {early_avg:.3f}")
    print(f"  {'Mid':<10} {'26-50':<12} {mid_avg:.3f}")
    print(f"  {'Late':<10} {'51-75':<12} {late_avg:.3f}")
    print(f"  {'Final':<10} {'76-' + str(n_episodes):<12} {final_avg:.3f}")
    print(f"  {'Overall':<10} {'1-' + str(n_episodes):<12} {overall:.3f}")
    print(f"{'═' * 42}\n")

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    log_path = os.path.join(os.path.dirname(__file__), "training_log.json")
    with open(log_path, "w") as f:
        json.dump(episode_log, f, indent=2)
    print(f"  📄 Training log saved → {log_path}")

    # ------------------------------------------------------------------
    # Matplotlib learning curve
    # ------------------------------------------------------------------
    if save_plot:
        ep_nums = [r["episode"] for r in episode_log]
        raw     = [r["reward"]  for r in episode_log]
        rolling = _rolling_avg(raw, window=5)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ep_nums, raw,     color="steelblue", alpha=0.6, linewidth=1.2, label="Episode Reward")
        ax.plot(ep_nums, rolling, color="crimson",   linewidth=2.0, linestyle="--", label="5-Episode Rolling Avg")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

        # Phase boundary shading
        ax.axvspan(1,  25, alpha=0.04, color="green",  label="Easy (diff 1-2)")
        ax.axvspan(26, 50, alpha=0.04, color="orange", label="Mid (diff 2-3)")
        ax.axvspan(51, 75, alpha=0.04, color="red",    label="Hard (diff 3-4)")
        ax.axvspan(76, n_episodes, alpha=0.04, color="purple", label="Extreme (diff 4-5)")

        ax.set_title("LifeStack Agent Learning Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Total Reward", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(os.path.dirname(__file__), "reward_curve.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  📊 Learning curve saved → {plot_path}")

    # ------------------------------------------------------------------
    # BEHAVIORAL COMPARISON — Friday 6PM (5 runs each)
    # ------------------------------------------------------------------
    N_COMPARE = 5
    print(f"\n{'═' * 58}")
    print(f"  BEHAVIORAL COMPARISON — Friday 6PM Crisis ({N_COMPARE} runs each)")
    print(f"{'═' * 58}")

    memory_dir    = os.path.join(os.path.dirname(__file__), "lifestack_memory")
    memory_backup = memory_dir + "_backup"

    # --- WITHOUT memory: temporarily hide the ChromaDB folder ---
    had_memory = os.path.exists(memory_dir)
    if had_memory:
        shutil.move(memory_dir, memory_backup)

    no_mem_results = []
    try:
        for i in range(N_COMPARE):
            result = run_episode(difficulty=5, verbose=False)
            first_step = result["step_log"][0] if result["step_log"] else {}
            has_comm = any(
                s.get("action") == "communicate" for s in result["step_log"]
            )
            no_mem_results.append({
                "run": i + 1,
                "total_reward": result["total_reward"],
                "first_action": first_step.get("action", "unknown"),
                "first_domain": first_step.get("domain", "unknown"),
                "has_communication": has_comm,
                "steps": result["steps"],
            })
    finally:
        # Restore memory
        if had_memory and os.path.exists(memory_backup):
            if os.path.exists(memory_dir):
                shutil.rmtree(memory_dir)
            shutil.move(memory_backup, memory_dir)

    # --- WITH memory ---
    with_mem_results = []
    for i in range(N_COMPARE):
        result = run_episode(difficulty=5, verbose=False)
        first_step = result["step_log"][0] if result["step_log"] else {}
        has_comm = any(
            s.get("action") == "communicate" for s in result["step_log"]
        )
        with_mem_results.append({
            "run": i + 1,
            "total_reward": result["total_reward"],
            "first_action": first_step.get("action", "unknown"),
            "first_domain": first_step.get("domain", "unknown"),
            "has_communication": has_comm,
            "steps": result["steps"],
        })

    # --- Compute stats ---
    avg_no  = sum(r["total_reward"] for r in no_mem_results)   / N_COMPARE
    avg_yes = sum(r["total_reward"] for r in with_mem_results) / N_COMPARE
    improvement = avg_yes - avg_no
    pct = (improvement / abs(avg_no) * 100) if avg_no != 0 else 0

    # Most common first action
    from collections import Counter
    no_actions  = Counter(r["first_action"] for r in no_mem_results)
    yes_actions = Counter(r["first_action"] for r in with_mem_results)
    no_domains  = Counter(r["first_domain"] for r in no_mem_results)
    yes_domains = Counter(r["first_domain"] for r in with_mem_results)
    no_comm_pct  = sum(1 for r in no_mem_results  if r["has_communication"]) / N_COMPARE * 100
    yes_comm_pct = sum(1 for r in with_mem_results if r["has_communication"]) / N_COMPARE * 100

    # --- Print table ---
    print(f"\n  {'WITHOUT MEMORY':<28} {'WITH MEMORY':<28}")
    for i in range(N_COMPARE):
        nr = no_mem_results[i]
        wr = with_mem_results[i]
        print(f"  Run {nr['run']}: {nr['total_reward']:.3f} "
              f"({nr['first_action']:<14})"
              f"  Run {wr['run']}: {wr['total_reward']:.3f} "
              f"({wr['first_action']:<14})")
    print(f"  {'─' * 54}")
    print(f"  Avg:   {avg_no:.3f}                    Avg:   {avg_yes:.3f}")
    sign = "+" if improvement >= 0 else ""
    print(f"  Improvement: {sign}{improvement:.3f} ({sign}{pct:.1f}%)")

    print(f"\n  {'─' * 54}")
    print(f"  Most common 1st action WITHOUT memory: {no_actions.most_common(1)[0][0]}")
    print(f"  Most common 1st action WITH memory:    {yes_actions.most_common(1)[0][0]}")
    print(f"  Most common 1st domain WITHOUT memory: {no_domains.most_common(1)[0][0]}")
    print(f"  Most common 1st domain WITH memory:    {yes_domains.most_common(1)[0][0]}")
    print(f"  Communication used WITHOUT memory:     {no_comm_pct:.0f}% of runs")
    print(f"  Communication used WITH memory:        {yes_comm_pct:.0f}% of runs")

    # --- Behavioral insight ---
    if yes_actions.most_common(1)[0][0] != no_actions.most_common(1)[0][0]:
        print(f"\n  💡 Memory changed the agent's primary strategy from "
              f"'{no_actions.most_common(1)[0][0]}' to '{yes_actions.most_common(1)[0][0]}'")
    if yes_comm_pct > no_comm_pct:
        print(f"  💡 Memory taught the agent to include communication actions more often")
    print(f"{'═' * 58}\n")

    # --- Save comparison ---
    comparison = {
        "scenario": "Friday 6PM (difficulty 5)",
        "runs_per_condition": N_COMPARE,
        "without_memory": {
            "results": no_mem_results,
            "avg_reward": round(avg_no, 3),
            "most_common_first_action": no_actions.most_common(1)[0][0],
            "most_common_first_domain": no_domains.most_common(1)[0][0],
            "communication_rate": round(no_comm_pct, 1),
        },
        "with_memory": {
            "results": with_mem_results,
            "avg_reward": round(avg_yes, 3),
            "most_common_first_action": yes_actions.most_common(1)[0][0],
            "most_common_first_domain": yes_domains.most_common(1)[0][0],
            "communication_rate": round(yes_comm_pct, 1),
        },
        "improvement": {
            "absolute": round(improvement, 3),
            "percentage": round(pct, 1),
        },
    }
    comp_path = os.path.join(os.path.dirname(__file__), "before_after_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  📄 Behavioral comparison saved → {comp_path}")

    return {
        "episode_log": episode_log,
        "phase_averages": {
            "early": early_avg,
            "mid": mid_avg,
            "late": late_avg,
            "final": final_avg,
            "overall": overall,
        },
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_training(n_episodes=100)


if __name__ == "__main__":
    main()
