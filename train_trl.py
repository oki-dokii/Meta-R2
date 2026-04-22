"""
train_trl.py — LifeStack GRPO Training via HuggingFace TRL + Unsloth

Trains a small LLM (Qwen2.5-3B or Llama-3.2-3B) to resolve life conflicts
using Group Relative Policy Optimization.

Usage (Colab):
    !pip install unsloth trl datasets transformers accelerate
    !python train_trl.py
"""

import json
import os
import copy
import random
import numpy as np

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# LifeStack imports
from life_state import LifeMetrics, DependencyGraph
from reward import compute_reward
from conflict_generator import generate_conflict, TEMPLATES
from simperson import SimPerson


# ──────────────────────────────────────────────
# 1. MODEL SETUP (Unsloth for 4-bit efficiency)
# ──────────────────────────────────────────────

def load_model():
    """Load model with Unsloth 4-bit quantization for Colab T4."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-3B-Instruct",
            max_seq_length=1024,
            dtype=None,  # auto-detect
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer
    except ImportError:
        # Fallback: standard HF loading (slower, more VRAM)
        from transformers import AutoModelForCausalLM
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        return model, tokenizer


# ──────────────────────────────────────────────
# 2. DATASET: Generate conflict prompts
# ──────────────────────────────────────────────

def build_prompt_for_conflict(conflict, person, metrics, budget):
    """Build a structured prompt from conflict state."""
    flat = metrics.flatten()
    status = "\n".join(f"  {k}: {v:.1f}" for k, v in flat.items())

    return (
        f"You are a life management AI. Resolve this crisis optimally.\n\n"
        f"CONFLICT: {conflict.title}\n"
        f"STORY: {conflict.story}\n\n"
        f"LIFE METRICS:\n{status}\n\n"
        f"RESOURCES: Time={budget.time_hours:.1f}h, "
        f"Money=${budget.money_dollars:.1f}, Energy={budget.energy_units:.1f}\n\n"
        f"Respond with ONLY valid JSON:\n"
        f'{{"action_type": "negotiate|communicate|delegate|spend|reschedule|rest|deprioritize", '
        f'"target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time", '
        f'"metric_changes": {{"domain.submetric": delta}}, '
        f'"resource_cost": {{"time": 0, "money": 0, "energy": 0}}, '
        f'"reasoning": "brief explanation"}}'
    )


def generate_dataset(n_prompts: int = 200) -> Dataset:
    """Generate n conflict prompts as a HuggingFace Dataset."""
    person_pool = [
        SimPerson(name="Alex", openness=0.4, conscientiousness=0.9,
                  extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2,
                  extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam", openness=0.5, conscientiousness=0.6,
                  extraversion=0.1, agreeableness=0.65, neuroticism=0.9),
        SimPerson(name="Maya", openness=0.5, conscientiousness=0.7,
                  extraversion=0.5, agreeableness=0.95, neuroticism=0.3),
        SimPerson(name="Leo", openness=0.85, conscientiousness=0.8,
                  extraversion=0.4, agreeableness=0.4, neuroticism=0.55),
    ]

    prompts = []
    for _ in range(n_prompts):
        conflict = random.choice(TEMPLATES)
        person = random.choice(person_pool)
        metrics = LifeMetrics()
        graph = DependencyGraph()
        metrics = graph.cascade(metrics, conflict.primary_disruption)
        
        # NOTE: Using a robust way to extract default budget to prevent NoneType errs
        budget_dict = conflict.resource_budget if hasattr(conflict, 'resource_budget') and conflict.resource_budget else {}
        budget = ResourceBudget(
            time_hours=20.0,
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=100.0,
        )
        prompt = build_prompt_for_conflict(conflict, person, metrics, budget)
        prompts.append({"prompt": prompt, "difficulty": conflict.difficulty})

    return Dataset.from_list(prompts)


# ──────────────────────────────────────────────
# 3. REWARD FUNCTION for GRPO
# ──────────────────────────────────────────────

def lifestack_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Score each LLM completion using the LifeStack reward function.
    
    GRPO calls this for each group of completions per prompt.
    Returns a list of float rewards.
    """
    rewards = []
    graph = DependencyGraph()

    for completion, prompt in zip(completions, prompts):
        try:
            # Parse the JSON action from the completion
            text = completion.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())

            # Reconstruct the conflict state from the prompt
            # (We use a default disruption for simplicity — 
            #  in production, encode the state in the prompt metadata)
            state_before = LifeMetrics()
            disruption = {"career.workload": 25.0, "finances.liquidity": -30.0}
            state_before = graph.cascade(state_before, disruption)

            # Apply the agent's proposed changes
            state_after = copy.deepcopy(state_before)
            metric_changes = data.get("metric_changes", {})
            for path, delta in metric_changes.items():
                if '.' not in path:
                    continue
                try:
                    delta = float(delta)
                except (ValueError, TypeError):
                    continue
                if abs(delta) > 5:
                    state_after = graph.cascade(state_after, {path: delta})
                else:
                    dom, sub = path.split('.', 1)
                    d = getattr(state_after, dom, None)
                    if d and hasattr(d, sub):
                        cur = getattr(d, sub)
                        setattr(d, sub, max(0.0, min(100.0, cur + delta)))

            resource_cost = data.get("resource_cost", {})
            for k in resource_cost:
                try:
                    resource_cost[k] = float(resource_cost[k])
                except (ValueError, TypeError):
                    resource_cost[k] = 0.0

            reward, _ = compute_reward(state_before, state_after, resource_cost, actions_taken=1)
            rewards.append(float(reward))

        except (json.JSONDecodeError, KeyError, Exception):
            # Invalid JSON or broken structure = penalty
            rewards.append(-0.5)

    return rewards


# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────

def train(n_prompts=200, n_epochs=3, output_dir="./lifestack_model"):
    print("=" * 50)
    print("  LIFESTACK GRPO TRAINING")
    print("=" * 50)

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate dataset
    print(f"\n[2/4] Generating {n_prompts} conflict prompts...")
    dataset = generate_dataset(n_prompts)
    print(f"  Dataset size: {len(dataset)}")

    # Configure GRPO
    print("\n[3/4] Configuring GRPO trainer...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=256,
        num_generations=4,          # 4 completions per prompt
        logging_steps=10,
        save_steps=50,
        report_to="none",          # Disable wandb for hackathon
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=lifestack_reward_fn,
    )

    # Train
    print("\n[4/4] Training...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")

    return trainer


# ──────────────────────────────────────────────
# 5. EVALUATION + REWARD CURVE
# ──────────────────────────────────────────────

def evaluate_and_plot(model_dir="./lifestack_model"):
    """Load the trained model, run 50 evaluation episodes, plot the curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 50)
    print("  EVALUATION")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    graph = DependencyGraph()
    rewards = []

    for ep in range(50):
        conflict = generate_conflict(difficulty=min(5, 1 + ep // 10))
        metrics = LifeMetrics()
        metrics = graph.cascade(metrics, conflict.primary_disruption)
        budget = ResourceBudget(time_hours=20.0, money_dollars=500.0, energy_units=100.0)
        person = SimPerson(name="Eval")

        prompt = build_prompt_for_conflict(conflict, person, metrics, budget)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.pad_token_id
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = lifestack_reward_fn([completion], [prompt])[0]
        rewards.append(r)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/50 | Reward: {r:.3f} | Avg: {np.mean(rewards):.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 51), rewards, color="steelblue", alpha=0.6, label="Episode Reward")

    # Rolling average
    window = 5
    rolling = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
    ax.plot(range(1, 51), rolling, color="crimson", linewidth=2, linestyle="--", label="5-ep Rolling Avg")

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("LifeStack GRPO — Evaluation Reward Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("grpo_reward_curve.png", dpi=150)
    plt.close(fig)
    print("📊 Saved grpo_reward_curve.png")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    trainer = train(n_prompts=200, n_epochs=3)
    evaluate_and_plot()
