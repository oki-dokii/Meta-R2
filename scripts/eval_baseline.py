"""
Baseline eval: Qwen2.5-1.5B-Instruct (no LoRA) on the same 50 episodes as evaluate_and_plot.

Usage (repo root or scripts/):
  python scripts/eval_baseline.py
  python scripts/eval_baseline.py --output ./baseline_results.json

Colab (one cell after deps + repo mount):
  !python scripts/eval_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from agent.conflict_generator import TaskGenerator, generate_conflict
from core.life_state import DependencyGraph, LifeMetrics, ResourceBudget
from intake.simperson import SimPerson
from scripts.train_trl import (
    ALL_DOMAINS,
    build_prompt_for_task,
    reward_task_success_fn,
)

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
N_EPISODES = 50


def _resolve_device_for_hf() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _model_device(model: Any) -> torch.device:
    d = getattr(model, "device", None)
    if d is not None:
        return d
    return next(model.parameters()).device


def load_base_model_qwen(
    model_name: str = BASE_MODEL_ID,
) -> tuple[Any, Any, str]:
    """Load base instruct model only (no PEFT), preferring Unsloth 4-bit when available."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        model.eval()
        return model, tokenizer, f"unsloth+4bit:{model_name}"
    except Exception as e:
        print(f"  Unsloth load failed ({e}), using transformers + AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _resolve_device_for_hf()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer, f"transformers:{model_name}"


def run_baseline_eval(
    model_name: str = BASE_MODEL_ID,
    n_episodes: int = N_EPISODES,
    output_path: str = "baseline_results.json",
) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print("  BASELINE EVALUATION (no LoRA)")
    print("=" * 50)

    model, tokenizer, load_tag = load_base_model_qwen(model_name)
    device = _model_device(model)
    print(f"  Loaded: {load_tag} | device={device}")

    graph = DependencyGraph()
    rewards: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    by_domain: dict[str, list[float]] = defaultdict(list)

    generator = TaskGenerator()
    for ep in range(n_episodes):
        difficulty = min(5, 1 + ep // 10)
        domain = ALL_DOMAINS[ep % len(ALL_DOMAINS)]
        ep_seed = ep * 137
        random.seed(ep_seed)
        task = generator.generate(domain=domain, difficulty=difficulty)
        random.seed()

        metrics = LifeMetrics()
        conflict = generate_conflict(difficulty)
        metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})

        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        person = SimPerson(name="Eval")

        prompt = build_prompt_for_task(task, person, metrics, budget, seed=ep_seed, step=0)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        r = float(reward_task_success_fn([completion], [prompt])[0])
        rewards.append(r)
        by_domain[domain].append(r)
        episode_rows.append(
            {
                "episode": ep,
                "domain": domain,
                "difficulty": difficulty,
                "seed": ep_seed,
                "reward": r,
            }
        )

        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep + 1}/{n_episodes} | Reward: {r:.3f} | "
                f"Running mean: {float(np.mean(rewards)):.3f}"
            )

    mean_r = float(np.mean(rewards))
    per_domain: dict[str, Any] = {}
    for d in ALL_DOMAINS:
        rs = by_domain.get(d, [])
        per_domain[d] = {
            "n": len(rs),
            "mean": float(np.mean(rs)) if rs else 0.0,
            "rewards": [float(x) for x in rs],
        }

    print("\n" + "-" * 50)
    print(f"  Mean reward (all {n_episodes} episodes): {mean_r:.4f}")
    print("  Per-domain mean (same schedule as evaluate_and_plot):")
    for d in ALL_DOMAINS:
        p = per_domain[d]
        if p["n"]:
            print(f"    {d:20s}  n={p['n']}  mean={p['mean']:.4f}")
    print("-" * 50)

    payload: dict[str, Any] = {
        "schema": "lifestack_baseline_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "load_method": load_tag,
        "n_episodes": n_episodes,
        "mean_reward": mean_r,
        "per_domain": per_domain,
        "all_domains_order": list(ALL_DOMAINS),
        "episodes": episode_rows,
    }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {output_path}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="50-episode baseline eval for Qwen2.5-1.5B-Instruct (no LoRA)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BASE_MODEL_ID,
        help="HF model id (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
        help="Number of eval episodes (default: 50, matches evaluate_and_plot)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.json",
        help="Where to write results JSON",
    )
    args = parser.parse_args()
    run_baseline_eval(
        model_name=args.model,
        n_episodes=args.episodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
