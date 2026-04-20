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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _difficulty_for_episode(episode: int) -> int:
    """Curriculum schedule: easy → medium → hard."""
    if episode <= 15:
        return random.randint(1, 2)
    elif episode <= 35:
        return random.randint(2, 3)
    else:
        return random.randint(3, 4)


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

    for ep in range(1, n_episodes + 1):
        difficulty = _difficulty_for_episode(ep)

        # Run episode silently
        result = run_episode(difficulty=difficulty, verbose=False)

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

        # Progress print every 5 episodes
        if ep % 5 == 0 or ep == 1:
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
    early_avg = _phase_avg(rewards, 1, 15)
    mid_avg   = _phase_avg(rewards, 16, 35)
    late_end  = min(n_episodes, 50)
    late_avg  = _phase_avg(rewards, 36, late_end)
    overall   = round(sum(rewards) / len(rewards), 3)

    print(f"\n{'═' * 42}")
    print(f"  TRAINING SUMMARY")
    print(f"{'═' * 42}")
    print(f"  {'Phase':<10} {'Episodes':<12} {'Avg Reward'}")
    print(f"  {'-'*38}")
    print(f"  {'Early':<10} {'1-15':<12} {early_avg:.3f}")
    print(f"  {'Mid':<10} {'16-35':<12} {mid_avg:.3f}")
    print(f"  {'Late':<10} {'36-' + str(late_end):<12} {late_avg:.3f}")
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
        ax.axvspan(1,  15, alpha=0.04, color="green",  label="Easy (diff 1-2)")
        ax.axvspan(16, 35, alpha=0.04, color="orange", label="Mid (diff 2-3)")
        ax.axvspan(36, n_episodes, alpha=0.04, color="red", label="Hard (diff 3-4)")

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
    # BEFORE / AFTER MEMORY COMPARISON
    # ------------------------------------------------------------------
    print(f"\n{'═' * 42}")
    print(f"  BEFORE vs AFTER MEMORY COMPARISON")
    print(f"  (Using Friday 6PM — difficulty 5)")
    print(f"{'═' * 42}")

    memory_dir = os.path.join(os.path.dirname(__file__), "lifestack_memory")
    memory_backup = memory_dir + "_backup"

    # --- Without memory: temporarily hide the ChromaDB folder ---
    had_memory = os.path.exists(memory_dir)
    if had_memory:
        shutil.move(memory_dir, memory_backup)

    try:
        result_no_mem = run_episode(difficulty=5, verbose=False)
        reward_no_mem = result_no_mem["total_reward"]
    finally:
        # Restore memory
        if had_memory and os.path.exists(memory_backup):
            # Remove the fresh blank DB created during no-memory run
            if os.path.exists(memory_dir):
                shutil.rmtree(memory_dir)
            shutil.move(memory_backup, memory_dir)

    # --- With memory ---
    result_with_mem = run_episode(difficulty=5, verbose=False)
    reward_with_mem = result_with_mem["total_reward"]

    improvement = round(reward_with_mem - reward_no_mem, 3)
    sign = "+" if improvement >= 0 else ""

    print(f"\n{'═' * 42}")
    print(f"  BEFORE vs AFTER MEMORY")
    print(f"  Without memory | Reward: {reward_no_mem:.3f}")
    print(f"  With memory    | Reward: {reward_with_mem:.3f}")
    print(f"  Improvement    : {sign}{improvement:.3f}")
    print(f"{'═' * 42}\n")

    return {
        "episode_log": episode_log,
        "phase_averages": {
            "early": early_avg,
            "mid": mid_avg,
            "late": late_avg,
            "overall": overall,
        },
        "before_memory_reward": reward_no_mem,
        "after_memory_reward": reward_with_mem,
        "improvement": improvement,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_training(n_episodes=50)


if __name__ == "__main__":
    main()
