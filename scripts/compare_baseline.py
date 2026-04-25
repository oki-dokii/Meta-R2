"""
Compare base vs trained LifeStack policy on identical crisis prompts.

Usage:
  python scripts/compare_baseline.py
  python scripts/compare_baseline.py --trained-model ./lifestack_model
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from agent.conflict_generator import TaskGenerator, generate_conflict
from core.life_state import DependencyGraph, LifeMetrics, ResourceBudget
from intake.simperson import SimPerson
from scripts.train_trl import build_prompt_for_task, get_lifestack_evaluation


def _load_base_model():
    """Load base Qwen2.5-1.5B-Instruct (no training adapter)."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth/base-qwen2.5-1.5b-instruct"
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer, model_name


def _load_trained_model(model_dir: str):
    """Load trained LifeStack model from local adapter/full checkpoint directory."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
        return model, tokenizer


def _device_for(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_completion(model, tokenizer, prompt: str, temperature: float = 0.3) -> str:
    device = _device_for(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def _build_eval_cases() -> list[dict[str, Any]]:
    """Create 5 deterministic prompts spanning different crisis domains."""
    domains = [
        ("career", 3, 101),
        ("finances", 4, 202),
        ("relationships", 3, 303),
        ("transport_crisis", 4, 404),
        ("code_merge_crisis", 5, 505),
    ]

    generator = TaskGenerator()
    graph = DependencyGraph()
    person = SimPerson(name="Comparator")
    cases: list[dict[str, Any]] = []

    for domain, difficulty, seed in domains:
        random.seed(seed)
        task = generator.generate(domain=domain, difficulty=difficulty)
        conflict = generate_conflict(difficulty)
        random.seed()

        metrics = LifeMetrics()
        metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})

        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )

        prompt = build_prompt_for_task(task, person, metrics, budget, seed=seed, step=0)
        crisis_text = task.domain_metadata.get("story", task.goal)

        cases.append(
            {
                "case_id": f"{domain}_d{difficulty}",
                "domain": domain,
                "difficulty": difficulty,
                "seed": seed,
                "crisis": crisis_text,
                "prompt": prompt,
            }
        )
    return cases


def _print_case(case: dict[str, Any]) -> None:
    print("=" * 110)
    print(f"[{case['case_id']}] domain={case['domain']} difficulty={case['difficulty']}")
    print(f"crisis: {case['crisis']}")
    print(f"base_reward={case['base_reward']:.3f} | trained_reward={case['trained_reward']:.3f} | delta={case['delta']:+.3f}")
    print("- BASE RESPONSE -")
    print(case["base_response"] or "<empty>")
    print("- TRAINED RESPONSE -")
    print(case["trained_response"] or "<empty>")


def run_compare(trained_model_dir: str, output_path: str) -> dict[str, Any]:
    cases = _build_eval_cases()

    print("Loading base model...")
    base_model, base_tokenizer, base_name = _load_base_model()
    for case in cases:
        completion = _generate_completion(base_model, base_tokenizer, case["prompt"])
        eval_data = get_lifestack_evaluation(completion, case["prompt"])
        case["base_model"] = base_name
        case["base_response"] = completion
        case["base_reward"] = float(eval_data.get("reward", -0.5))
    del base_model
    torch.cuda.empty_cache()

    print("Loading trained model...")
    trained_model, trained_tokenizer = _load_trained_model(trained_model_dir)
    for case in cases:
        completion = _generate_completion(trained_model, trained_tokenizer, case["prompt"])
        eval_data = get_lifestack_evaluation(completion, case["prompt"])
        case["trained_model"] = trained_model_dir
        case["trained_response"] = completion
        case["trained_reward"] = float(eval_data.get("reward", -0.5))
        case["delta"] = round(case["trained_reward"] - case["base_reward"], 4)
        _print_case(case)
    del trained_model
    torch.cuda.empty_cache()

    avg_base = sum(c["base_reward"] for c in cases) / len(cases)
    avg_trained = sum(c["trained_reward"] for c in cases) / len(cases)
    avg_delta = avg_trained - avg_base

    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "n_cases": len(cases),
            "avg_base_reward": round(avg_base, 4),
            "avg_trained_reward": round(avg_trained, 4),
            "avg_reward_delta": round(avg_delta, 4),
            "base_model": cases[0]["base_model"] if cases else "",
            "trained_model": trained_model_dir,
        },
        "cases": cases,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 110)
    print(
        f"SUMMARY: avg_base={avg_base:.3f} | avg_trained={avg_trained:.3f} | "
        f"avg_delta={avg_delta:+.3f}"
    )
    print(f"Saved comparison JSON: {output_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Compare baseline Qwen vs trained LifeStack model.")
    parser.add_argument("--trained-model", type=str, default="./lifestack_model")
    parser.add_argument("--output", type=str, default="./data/before_after_comparison.json")
    args = parser.parse_args()

    run_compare(trained_model_dir=args.trained_model, output_path=args.output)


if __name__ == "__main__":
    main()
