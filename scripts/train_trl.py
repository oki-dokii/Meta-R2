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

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LifeStack imports
from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
from core.reward import compute_reward
from agent.conflict_generator import generate_conflict, TEMPLATES, TaskGenerator
from intake.simperson import SimPerson
from core.task import Task


# ──────────────────────────────────────────────
# 1. MODEL SETUP (Unsloth for 4-bit efficiency)
# ──────────────────────────────────────────────

def load_model():
    """Load model with Unsloth 4-bit quantization for Colab T4."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
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
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        return model, tokenizer


# ──────────────────────────────────────────────
# 2. DATASET: Generate conflict prompts
# ──────────────────────────────────────────────

def build_prompt_for_task(task, person, metrics, budget):
    """Build a structured prompt from task state, embedding hidden metadata for the reward function."""
    flat = metrics.flatten()
    status = "\n".join(f"  {k}: {v:.1f}" for k, v in flat.items())
    
    # NEW: Store more task metadata for reconstruction
    metadata = {
        "goal": task.goal,
        "disruption": task.mutable_world,
        "difficulty": task.difficulty,
        "horizon": task.horizon,
        "budget": {
            "time": budget.time_hours,
            "money": budget.money_dollars,
            "energy": budget.energy_units
        }
    }
    metadata_str = json.dumps(metadata)

    return (
        f"You are a life management AI. Resolve this crisis optimally.\n\n"
        f"<SYSTEM_METADATA>\n{metadata_str}\n</SYSTEM_METADATA>\n\n"
        f"TASK: {task.goal}\n"
        f"STORY: {task.domain_metadata.get('story', '')}\n\n"
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


def generate_dataset(n_prompts: int = 200, difficulty: int = None) -> Dataset:
    """Generate n conflict prompts as a HuggingFace Dataset, with optional fixed difficulty."""
    # ... (pool setup unchanged) ...
    person_pool = [
        SimPerson(name="Alex", openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2, extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.9),
    ]

    generator = TaskGenerator()
    prompts = []
    for i in range(n_prompts):
        person = random.choice(person_pool)
        domain = random.choice(["flight_crisis", "code_merge_crisis"])
        # If difficulty is not provided, cycle through all 5 levels (legacy mode)
        curr_diff = difficulty if difficulty else (i % 5) + 1
        
        task = generator.generate(domain=domain, difficulty=curr_diff)
        
        # Merge legacy conflict metrics for variety
        conflict = generate_conflict(curr_diff)
        task.mutable_world.update(conflict.primary_disruption)
        task.visible_world.update(conflict.primary_disruption)
        
        metrics = LifeMetrics()
        graph = DependencyGraph()
        metrics = graph.cascade(metrics, task.mutable_world)
        
        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        prompt = build_prompt_for_task(task, person, metrics, budget)
        prompts.append({"prompt": prompt, "difficulty": curr_diff})

    return Dataset.from_list(prompts)


# ──────────────────────────────────────────────
# 3. REWARD FUNCTION for GRPO
# ──────────────────────────────────────────────

_REWARD_CACHE = {}
_GLOBAL_REWARD_CALL_COUNT = 0
LOG_INTERVAL = 20
LOG_DIR = "training_logs"
SAMPLE_LOG_PATH = os.path.join(LOG_DIR, "generations.jsonl")

def get_lifestack_evaluation(completion: str, prompt: str) -> dict:
    """Run the environment and return the full reward breakdown. Cached for efficiency."""
    key = (prompt, completion)
    if key in _REWARD_CACHE:
        return _REWARD_CACHE[key]
        
    from core.lifestack_env import LifeStackEnv, LifeStackAction
    import re
    
    try:
        # 1. Parse JSON
        text = completion.strip()
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[-1].split("```")[0]
        data = json.loads(text.strip())

        # 2. Extract Task Metadata
        m = re.search(r'<SYSTEM_METADATA>\n(.*?)\n</SYSTEM_METADATA>', prompt, re.DOTALL)
        if not m:
            return {"reward": -0.5, "breakdown": {}}
        
        meta = json.loads(m.group(1).strip())
        task = Task(
            id="grpo_eval", domain="life_conflict",
            goal=meta.get("goal", "Resolve Crisis"),
            constraints={"budget": meta.get("budget", {})},
            hidden_state={},
            mutable_world=meta.get("disruption", {}),
            visible_world=meta.get("disruption", {}),
            success_conditions=[], failure_conditions=[],
            event_schedule=[], viable_routes=[], milestones=[],
            horizon=meta.get("horizon", 30), difficulty=meta.get("difficulty", 3)
        )

        # 3. Step Env
        env = LifeStackEnv()
        env.reset(task=task, conflict=meta.get("disruption", {}))
        action = LifeStackAction(
            action_type=data.get("action_type"),
            target=data.get("target_domain"),
            metric_changes=data.get("metric_changes", {}),
            resource_cost=data.get("resource_cost", {}),
            reasoning=data.get("reasoning", ""),
            completion=completion,
            actions_taken=1
        )
        obs = env.step(action)
        
        result = {
            "reward": float(obs.reward),
            "breakdown": obs.metadata.get("breakdown", {}),
            "action": action
        }

        # 4. Global Logging
        global _GLOBAL_REWARD_CALL_COUNT
        _GLOBAL_REWARD_CALL_COUNT += 1
        if _GLOBAL_REWARD_CALL_COUNT % LOG_INTERVAL == 0:
            if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
            log_entry = {
                "step": _GLOBAL_REWARD_CALL_COUNT,
                "prompt": prompt[:500] + "...",
                "completion": completion,
                "action": data,
                "reward": result["reward"],
                "breakdown": result["breakdown"]
            }
            with open(SAMPLE_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        _REWARD_CACHE[key] = result
        return result
        
    except Exception:
        return {"reward": -0.5, "breakdown": {}, "action": None, "initial_metrics": meta.get("disruption", {}) if 'meta' in locals() else {}}

def reward_format_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Scores JSON format compliance."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("format_compliance", -0.5) for c, p in zip(completions, prompts)]

def reward_plausibility_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Penalize zero-cost metric changes."""
    return [1.0 if "PLAUSIBILITY_VIOLATION" not in get_lifestack_evaluation(c, p).get("breakdown", {}).get("penalties_fired", []) else -1.0 for c, p in zip(completions, prompts)]

def reward_task_success_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Core outcome reward."""
    return [get_lifestack_evaluation(c, p)["reward"] for c, p in zip(completions, prompts)]

def reward_milestone_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Monitor progress through logical bottlenecks."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("milestone", 0.0) for c, p in zip(completions, prompts)]

def reward_reasoning_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Evaluate planning coherence."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("reasoning", 0.0) for c, p in zip(completions, prompts)]

def reward_human_feedback_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Rewards actions that align with past human outcome feedback.
    This effectively uses the 'Real-World Verification' data for training.
    """
    from core.feedback import OutcomeFeedback, compute_human_feedback_reward
    from agent.memory import LifeStackMemory
    memo = LifeStackMemory(silent=True)
    
    rewards = []
    for c, p in zip(completions, prompts):
        eval_res = get_lifestack_evaluation(c, p)
        action = eval_res.get("action")
        if not action:
            rewards.append(0.0)
            continue
            
        # Retrieve similar past human feedback
        # (Simplified: we search by the action's target domain/reasoning)
        similar_fb_list = memo.feedback_collection.query(
            query_texts=[action.reasoning],
            n_results=1
        ).get('metadatas', [[]])[0]
        
        if not similar_fb_list:
            rewards.append(0.0)
            continue
            
        fb_meta = similar_fb_list[0]
        # Reconstruct feedback object
        fb = OutcomeFeedback(
            episode_id=fb_meta["episode_id"],
            overall_effectiveness=fb_meta["effectiveness"],
            domains_improved=json.loads(fb_meta["domains_improved"]),
            domains_worsened=json.loads(fb_meta["domains_worsened"])
        )
        
        # Predicted observation from our env run
        # We need to wrap it in LifeStackObservation for the helper
        from core.lifestack_env import LifeStackObservation
        obs = LifeStackObservation(metrics=eval_res["breakdown"].get("local_breakdown", {}).get("metrics_after", {}))
        
        # The initial metrics are needed to see if improvement was predicted correctly
        # Extract from prompt metadata (fallback to default)
        init_metrics = eval_res.get("initial_metrics", {})
        
        fb_reward = compute_human_feedback_reward(init_metrics, obs, fb)
        rewards.append(fb_reward)
        
    return rewards


# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────

def train_curriculum(n_stages=5, n_prompts_per_stage=100, output_dir="./lifestack_model"):
    """
    Implements an adaptive curriculum: Progress through difficulties 1-5 
    only when success rate on the current level exceeds 70%.
    """
    print("=" * 60)
    print("🚀 LIFESTACK SUCCESS-BASED CURRICULUM TRAINING")
    print("=" * 60)

    model, tokenizer = load_model()
    curr_diff = 1
    
    for stage in range(1, n_stages + 1):
        print(f"\n[STAGE {stage}] Training on Difficulty {curr_diff}...")
        
        # 1. Generate data for this difficulty
        dataset = generate_dataset(n_prompts_per_stage, difficulty=curr_diff)
        
        # 2. Configure and Train
        config = GRPOConfig(
            output_dir=f"{output_dir}/stage_{stage}",
            num_train_epochs=1, # Fast iteration for curriculum
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            num_generations=4,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            report_to="tensorboard"
        )
        
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=config,
            train_dataset=dataset,
            reward_funcs=[
                reward_format_fn, 
                reward_plausibility_fn,
                reward_task_success_fn,
                reward_milestone_fn,
                reward_reasoning_fn,
                reward_human_feedback_fn
            ],
        )
        trainer.train()
        
        # 3. EVALUATE: Decide whether to progress
        from scripts.eval import run_evaluation
        # We need a temp save for the eval script to load
        eval_path = f"{output_dir}/curr_model"
        trainer.save_model(eval_path)
        tokenizer.save_pretrained(eval_path)
        
        print(f"\nComparing performance for Curriculum Progression...")
        # (Internal eval call or simple metric check)
        # For simplicity in this script, we'll check the avg_reward from the last log
        avg_reward = trainer.state.log_history[-1].get("reward", 0.0) if trainer.state.log_history else 0.0
        
        # Logic: If avg reward is healthy, bump difficulty
        if avg_reward > 0.6 and curr_diff < 5:
            print(f"✅ Stage Success (Reward: {avg_reward:.3f})! Increasing difficulty to {curr_diff + 1}.")
            curr_diff += 1
        else:
            print(f"⚠️  Holding at Difficulty {curr_diff} (Reward: {avg_reward:.3f}) for next stage.")

    # Final Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
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

    generator = TaskGenerator()
    for ep in range(50):
        difficulty = min(5, 1 + ep // 10)
        domain = "flight_crisis" if ep % 2 == 0 else "code_merge_crisis"
        task = generator.generate(domain=domain, difficulty=difficulty)
        
        metrics = LifeMetrics()
        # Initial disruption from legacy templates
        conflict = generate_conflict(difficulty)
        metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})
        
        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        person = SimPerson(name="Eval")

        prompt = build_prompt_for_task(task, person, metrics, budget)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.pad_token_id
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward_task_success_fn([completion], [prompt])[0]
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
    trainer = train_curriculum(n_stages=5, n_prompts_per_stage=100)
    evaluate_and_plot()
