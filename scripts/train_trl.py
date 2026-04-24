"""
train_trl.py — LifeStack GRPO Training via HuggingFace TRL + Unsloth

Trains a small LLM (Qwen2.5-1.5B-Instruct) to resolve daily-life conflicts
across 8 domains using Group Relative Policy Optimization (GRPO).

Supported domains:
    career, finances, relationships, physical_health,
    mental_wellbeing, time, flight_crisis, code_merge_crisis

Usage (Colab / GPU):
    !pip install unsloth trl datasets transformers accelerate
    !python train_trl.py                       # full curriculum (5 stages)
    !python train_trl.py --dry-run             # 1-step smoke test (CPU OK)
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
from core.task import Task, FlightCrisisTask


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

def build_prompt_for_task(task, person, metrics, budget, seed=42):
    """Build a structured prompt from task state, embedding hidden metadata for the reward function."""
    flat = metrics.flatten()
    status = "\n".join(f"  {k}: {v:.1f}" for k, v in flat.items())
    
    # NEW: Store more task metadata for reconstruction
    metadata = {
        "goal": task.goal,
        "domain": task.domain,
        "disruption": task.mutable_world,
        "difficulty": task.difficulty,
        "horizon": task.horizon,
        "seed": seed,
        "budget": {
            "time": budget.time_hours,
            "money": budget.money_dollars,
            "energy": budget.energy_units
        }
    }
    metadata_str = json.dumps(metadata)

    # Render viable routes
    routes_str = "\n".join(
        f"  - {r.id}: {r.name} ({r.description}). Requires: {', '.join(r.required_action_types)}"
        for r in task.viable_routes
    )

    return (
        f"You are a life management AI. Resolve this crisis optimally.\n\n"
        f"<SYSTEM_METADATA>\n{metadata_str}\n</SYSTEM_METADATA>\n\n"
        f"TASK: {task.goal}\n"
        f"STORY: {task.domain_metadata.get('story', '')}\n\n"
        f"LIFE METRICS:\n{status}\n\n"
        f"RESOURCES: Time={budget.time_hours:.1f}h, "
        f"Money=${budget.money_dollars:.1f}, Energy={budget.energy_units:.1f}\n\n"
        f"AVAILABLE ROUTES (To execute a route, prerequisites must be met):\n{routes_str}\n\n"
        f"Respond with ONLY valid JSON:\n"
        f'{{"action_type": "negotiate|communicate|delegate|spend|reschedule|rest|deprioritize|execute", '
        f'"target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time OR <route_id>", '
        f'"metric_changes": {{"domain.submetric": delta}}, '
        f'"resource_cost": {{"time": 0, "money": 0, "energy": 0}}, '
        f'"reasoning": "brief explanation"}}'
    )


# All 8 TaskGenerator domains — covers the full daily-life action space.
# transport_crisis randomly dispatches to: flight, train, car, rideshare, transit-strike
ALL_DOMAINS = [
    "career",
    "finances",
    "relationships",
    "physical_health",
    "mental_wellbeing",
    "time",
    "transport_crisis",   # ← was flight_crisis; now covers all 5 transport modes
    "code_merge_crisis",
]

def generate_dataset(n_prompts: int = 200, difficulty: int = None) -> Dataset:
    """
    Generate n conflict prompts as a HuggingFace Dataset.

    Samples evenly across ALL 8 daily-life domains (career, finances,
    relationships, physical_health, mental_wellbeing, time,
    transport_crisis [flight/train/car/rideshare/transit-strike], code_merge_crisis)
    so GRPO learns a general life-management policy.

    Args:
        n_prompts: Total number of prompts to generate.
        difficulty: If given, fix all prompts to this difficulty (1-5).
                    If None, cycles evenly through levels 1-5.
    """
    person_pool = [
        SimPerson(name="Alex",  openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2, extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam",   openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.90),
        SimPerson(name="Jordan",openness=0.7, conscientiousness=0.5, extraversion=0.6, agreeableness=0.50, neuroticism=0.40),
        SimPerson(name="Maya",  openness=0.8, conscientiousness=0.7, extraversion=0.3, agreeableness=0.80, neuroticism=0.60),
    ]

    generator = TaskGenerator()
    prompts = []
    for i in range(n_prompts):
        person = random.choice(person_pool)
        # Round-robin across all 8 domains — guarantees balanced coverage
        domain = ALL_DOMAINS[i % len(ALL_DOMAINS)]
        # Cycle difficulty 1-5 unless fixed
        curr_diff = difficulty if difficulty else (i % 5) + 1

        task_seed = random.randint(0, 999999)
        random.seed(task_seed)
        task = generator.generate(domain=domain, difficulty=curr_diff)

        # Overlay a matching legacy conflict disruption for richer metric seeding
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
        prompt = build_prompt_for_task(task, person, metrics, budget, seed=task_seed)
        prompts.append({"prompt": prompt, "difficulty": curr_diff, "domain": domain})

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
    global _REWARD_CACHE
    if len(_REWARD_CACHE) > 1000:
        _REWARD_CACHE.clear()

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
        try:
            # Use TaskGenerator so routes/milestones/success_conditions are populated.
            from agent.conflict_generator import TaskGenerator
            gen = TaskGenerator()
            domain = meta.get("domain", "flight_crisis")
            import random
            random.seed(meta.get("seed", 42))
            task = gen.generate(domain=domain, difficulty=meta.get("difficulty", 3))
            random.seed() # reset
            # Overlay the actual disruption that was presented in the prompt
            task.mutable_world.update(meta.get("disruption", {}))
            task.visible_world.update(meta.get("disruption", {}))
        except Exception as e:
            print(f"[reward] Task construction failed: {e}")
            return {"reward": -0.5, "breakdown": {"error": str(e)}}

        # Validate required fields are present and non-None.
        _required = ("id", "goal", "constraints", "mutable_world", "visible_world")
        if any(getattr(task, f, None) is None for f in _required):
            print("[reward] Task missing required fields after construction.")
            return {"reward": -0.5, "breakdown": {"error": "missing_fields"}}

        # 3. Step Env
        env = LifeStackEnv()
        env.reset(task=task, conflict=meta.get("disruption", {}))
        initial_metrics = dict(env.state.current_metrics.flatten())
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
            "action": action,
            "obs_metrics": dict(obs.metrics),
            "initial_metrics": initial_metrics
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
                "breakdown": result["breakdown"],
                "components": result["breakdown"].get("components", {})
            }
            with open(SAMPLE_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            components = result["breakdown"].get("components", {})
            if components:
                comp_str = " | ".join(f"{k}={v:.3f}" for k, v in components.items())
                print(f"[step {_GLOBAL_REWARD_CALL_COUNT}] reward={result['reward']:.3f} | {comp_str}")

        _REWARD_CACHE[key] = result
        return result
        
    except Exception:
        return {"reward": -0.5, "breakdown": {}, "action": None, "initial_metrics": meta.get("disruption", {}) if 'meta' in locals() else {}}

def reward_format_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Scores JSON format compliance independently."""
    from core.reward import reward_format_compliance
    return [reward_format_compliance(c) for c in completions]

def reward_plausibility_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Penalize zero-cost metric changes using the severity-aware score from the env."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("plausibility", 0.0) for c, p in zip(completions, prompts)]

def reward_task_success_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Core outcome reward isolated to completion (avoiding double-dip). Returns penalty if evaluation failed."""
    results = []
    for c, p in zip(completions, prompts):
        eval_res = get_lifestack_evaluation(c, p)
        # If breakdown is empty, it means the evaluation failed (likely JSON error)
        # We return the top-level reward which contains the -0.5 failure penalty.
        if not eval_res.get("breakdown"):
            results.append(eval_res.get("reward", -0.5))
        else:
            results.append(eval_res.get("breakdown", {}).get("components", {}).get("completion", 0.0))
    return results

def reward_milestone_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Monitor progress through logical bottlenecks."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("milestone", 0.0) for c, p in zip(completions, prompts)]

def reward_reasoning_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Evaluate planning coherence. Scaled 10x to match task_success variance."""
    return [get_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("reasoning", 0.0) * 10.0 for c, p in zip(completions, prompts)]

def reward_human_feedback_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Rewards actions that align with past human outcome feedback (ChromaDB memory).

    Requires chromadb + a pre-populated LifeStackMemory database.
    Falls back silently to neutral 0.0 when:
      - chromadb is not installed (e.g. fresh Kaggle / Colab session)
      - the memory DB is empty or unreachable
    Returns 0.0 (abstain) rather than penalising the model.
    """
    # ── Guard: skip gracefully if chromadb / memory unavailable ──────────
    try:
        from core.feedback import OutcomeFeedback, compute_human_feedback_reward
        from agent.memory import LifeStackMemory
        memo = LifeStackMemory(silent=True)
    except (ImportError, Exception) as e:
        print(f"[warning] reward_human_feedback_fn unavailable ({e}), applying small penalty.")
        # chromadb not installed or DB init failed — apply small penalty
        return [-0.01] * len(completions)

    rewards = []
    for c, p in zip(completions, prompts):
        try:
            eval_res = get_lifestack_evaluation(c, p)
            action = eval_res.get("action")
            if not action:
                rewards.append(0.0)
                continue

            # Use task prompt to query feedback instead of model-generated reasoning
            # to avoid reward-hacking ChromaDB
            similar_fb_list = memo.feedback_collection.query(
                query_texts=[p],
                n_results=1
            ).get('metadatas', [[]])[0]

            if not similar_fb_list:
                rewards.append(0.0)
                continue

            fb_meta = similar_fb_list[0]
            fb = OutcomeFeedback(
                episode_id=fb_meta["episode_id"],
                overall_effectiveness=fb_meta["effectiveness"],
                domains_improved=json.loads(fb_meta["domains_improved"]),
                domains_worsened=json.loads(fb_meta["domains_worsened"])
            )

            from core.lifestack_env import LifeStackObservation
            obs = LifeStackObservation(metrics=eval_res.get("obs_metrics", {}))
            init_metrics = eval_res.get("initial_metrics", {})
            fb_reward = compute_human_feedback_reward(init_metrics, obs, fb)
            rewards.append(fb_reward)

        except Exception:
            rewards.append(0.0)

    return rewards


# ──────────────────────────────────────────────
# 4. CHECKPOINT HELPERS
# ──────────────────────────────────────────────

def find_latest_checkpoint(stage_dir: str) -> str | None:
    """
    Scan a stage output directory for the most recent Trainer checkpoint.
    Returns the checkpoint path, or None if none exist.
    """
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(stage_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1])
    )
    return checkpoints[-1] if checkpoints else None


_CURRICULUM_STATE_FILE = "curriculum_state.json"

def save_stage_state(output_dir: str, stage: int, curr_diff: int):
    """Persist curriculum progress so we can resume after a session cut."""
    path = os.path.join(output_dir, _CURRICULUM_STATE_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed_stage": stage, "next_difficulty": curr_diff}, f)
    print(f"  [ckpt] Curriculum state saved → stage={stage}, next_diff={curr_diff}")


def load_stage_state(output_dir: str) -> tuple[int, int]:
    """
    Returns (start_stage, curr_diff) from a previous run.
    Falls back to (1, 1) if no state file exists.
    """
    path = os.path.join(output_dir, _CURRICULUM_STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        start_stage = state["completed_stage"] + 1
        curr_diff   = state["next_difficulty"]
        print(f"  [ckpt] Resuming from stage {start_stage}, difficulty {curr_diff}")
        return start_stage, curr_diff
    return 1, 1


# ──────────────────────────────────────────────
# 5. TRAINING LOOP  (checkpoint-aware)
# ──────────────────────────────────────────────

def train_curriculum(
    n_stages=5,
    n_prompts_per_stage=100,
    output_dir="./lifestack_model",
    resume=False,
    start_stage=None,
):
    """
    Curriculum training with automatic checkpoint saving and resume.

    Each stage saves a checkpoint every 25 steps and persists curriculum
    state to curriculum_state.json.  If the session is killed mid-stage,
    re-run with --resume and the trainer will pick up from the last
    saved checkpoint automatically.

    Args:
        resume:      If True, read curriculum_state.json to find the last
                     completed stage and continue from there.
        start_stage: Override the starting stage (1-indexed). Useful for
                     manual restart (e.g. --start-stage 3).
    """
    print("=" * 60)
    print("🚀 LIFESTACK SUCCESS-BASED CURRICULUM TRAINING")
    print("=" * 60)

    model, tokenizer = load_model()

    # ── Determine where to start ────────────────────────────────────────
    if resume:
        first_stage, curr_diff = load_stage_state(output_dir)
    elif start_stage:
        first_stage = start_stage
        curr_diff   = 1          # difficulty resets; user can edit state file for fine control
    else:
        first_stage, curr_diff = 1, 1

    for stage in range(first_stage, n_stages + 1):
        print(f"\n[STAGE {stage}/{n_stages}] Difficulty={curr_diff}")

        stage_dir = f"{output_dir}/stage_{stage}"

        # ── Check for a mid-stage checkpoint from a previous session ─────
        resume_ckpt = find_latest_checkpoint(stage_dir) if resume else None
        if resume_ckpt:
            print(f"  [ckpt] Resuming mid-stage from: {resume_ckpt}")
        else:
            # Generate fresh data only for a clean start of the stage
            dataset = generate_dataset(n_prompts_per_stage, difficulty=curr_diff)

        # ── GRPOConfig with checkpoint cadence ───────────────────────────
        config = GRPOConfig(
            output_dir=stage_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            # TRL 1.x rule: num_generations must divide per_device_train_batch_size
            # batch=2, num_generations=2 → 2 % 2 = 0 ✓
            num_generations=2,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            # ── Checkpoint settings ──────────────────────────────────────
            save_strategy="steps",
            save_steps=25,
            save_total_limit=3,
            # ── Logging ─────────────────────────────────────────────────
            logging_steps=5,
            report_to="tensorboard",
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,  # TRL 1.x: renamed from tokenizer=
            args=config,
            train_dataset=dataset if not resume_ckpt else generate_dataset(n_prompts_per_stage, difficulty=curr_diff),
            reward_funcs=[
                # reward_format_fn MUST be first: it's the only signal that varies
                # between completions in early training (partial JSON vs garbage).
                # Without it, all completions fail JSON parse with the same default
                # score → reward_std=0 → zero GRPO gradient.
                reward_format_fn,
                reward_plausibility_fn,
                reward_task_success_fn,
                reward_milestone_fn,
                reward_reasoning_fn,
                reward_human_feedback_fn,
            ],
        )

        # Pass the checkpoint path — Trainer will reload weights + optimizer state
        trainer.train(resume_from_checkpoint=resume_ckpt)

        # ── Save completed stage model ───────────────────────────────────
        trainer.save_model(stage_dir)
        tokenizer.save_pretrained(stage_dir)
        print(f"  ✅ Stage {stage} model saved → {stage_dir}")

        # ── Curriculum progression logic ─────────────────────────────────
        avg_reward = (
            trainer.state.log_history[-1].get("reward", 0.0)
            if trainer.state.log_history else 0.0
        )
        if avg_reward > 0.6 and curr_diff < 5:
            print(f"  ✅ Reward {avg_reward:.3f} > 0.6 — advancing to difficulty {curr_diff + 1}")
            curr_diff += 1
        else:
            print(f"  ⚠️  Reward {avg_reward:.3f} — holding at difficulty {curr_diff}")

        # ── Persist curriculum state AFTER each stage ────────────────────
        # This is what lets us resume correctly on next session
        save_stage_state(output_dir, stage, curr_diff)

    # ── Final model save ─────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n🏁 Training complete. Final model → {output_dir}")
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

    # Use Unsloth's loader to avoid peft version conflicts on Kaggle/Colab
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("  Loaded via Unsloth FastLanguageModel")
    except Exception as unsloth_err:
        print(f"  Unsloth load failed ({unsloth_err}), falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float16, device_map="auto"
        )
    model.eval()

    graph = DependencyGraph()
    rewards = []

    generator = TaskGenerator()
    for ep in range(50):
        difficulty = min(5, 1 + ep // 10)
        # Cycle through all 8 domains during evaluation
        domain = ALL_DOMAINS[ep % len(ALL_DOMAINS)]
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

# ──────────────────────────────────────────────
# 6. POST-TRAINING VALIDATION
# ──────────────────────────────────────────────

MIN_MODEL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB — a placeholder will always be < this

def validate_saved_model(output_dir: str = "./lifestack_model"):
    """
    Validates that a real model was saved (not a placeholder).
    Raises RuntimeError if pytorch_model.bin or model.safetensors is missing / too small.
    """
    import glob
    weight_files = (
        glob.glob(os.path.join(output_dir, "*.bin")) +
        glob.glob(os.path.join(output_dir, "*.safetensors")) +
        glob.glob(os.path.join(output_dir, "**", "*.safetensors"), recursive=True) +
        glob.glob(os.path.join(output_dir, "**", "*.bin"), recursive=True)
    )
    # Deduplicate
    weight_files = list(set(weight_files))

    if not weight_files:
        raise RuntimeError(
            f"[VALIDATION FAIL] No weight files found in {output_dir}.\n"
            "Real training never completed — run train_trl.py on a GPU instance."
        )

    total_bytes = sum(os.path.getsize(f) for f in weight_files)
    if total_bytes < MIN_MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"[VALIDATION FAIL] Total weight size = {total_bytes} bytes ({total_bytes/1e6:.2f} MB).\n"
            f"Expected > {MIN_MODEL_SIZE_BYTES/1e6:.0f} MB for a real model.\n"
            f"Found files: {weight_files}\n"
            "This looks like a placeholder. Run full training on a GPU."
        )

    print(f"[VALIDATION PASS] Model saved correctly.")
    print(f"  Weight files : {len(weight_files)}")
    print(f"  Total size   : {total_bytes / 1e6:.1f} MB")
    return total_bytes


# ──────────────────────────────────────────────
# 7. DRY-RUN MODE (validates pipeline without GPU)
# ──────────────────────────────────────────────

def dry_run(output_dir: str = "./lifestack_model_dryrun"):
    """
    Runs a single GRPO training step on a minimal dataset (4 prompts).
    Verifies the entire pipeline: dataset → prompt → reward → trainer.train() → save.
    Does NOT require a GPU.  Saved weights will be small (< 50 MB) — that is expected.

    Use this to confirm:
    - All imports resolve
    - Reward functions are callable
    - Trainer.train() completes without error
    - model.save_pretrained() writes real weight files
    """
    print("=" * 60)
    print("🧪 LIFESTACK DRY-RUN (1 step, CPU, tiny dataset)")
    print("=" * 60)

    model, tokenizer = load_model()

    dataset = generate_dataset(n_prompts=4, difficulty=1)
    print(f"  Dataset size : {len(dataset)} prompts")

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        # TRL 1.x has two hard rules:
        #   1. num_generations >= 2  (needs ≥2 samples to compute advantages)
        #   2. per_device_train_batch_size % num_generations == 0
        # Minimum valid config: batch=2, num_generations=2
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_generations=2,
        max_steps=1,          # ONE step — just proves the pipeline works
        bf16=False,
        fp16=False,
        report_to="none",     # No tensorboard for dry-run
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 1.x: renamed from tokenizer=
        args=config,
        train_dataset=dataset,
        reward_funcs=[
            reward_plausibility_fn,
            reward_task_success_fn,
        ],
    )

    print("  Running 1 training step...")
    trainer.train()
    print("  ✅ trainer.train() completed.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✅ model.save_pretrained() → {output_dir}")

    # Check something real was saved
    import glob
    weight_files = (
        glob.glob(os.path.join(output_dir, "*.bin")) +
        glob.glob(os.path.join(output_dir, "*.safetensors")) +
        glob.glob(os.path.join(output_dir, "**", "*.safetensors"), recursive=True)
    )
    weight_files = list(set(weight_files))
    total_bytes = sum(os.path.getsize(f) for f in weight_files)

    print(f"\n  Weight files saved : {len(weight_files)}")
    for f in weight_files:
        print(f"    {f}  ({os.path.getsize(f)/1e6:.2f} MB)")
    print(f"  Total weight size  : {total_bytes/1e6:.2f} MB")

    if total_bytes == 0:
        raise RuntimeError("[DRY-RUN FAIL] No bytes written. save_pretrained() did not produce weights.")
    if total_bytes <= 100:  # 17 bytes = placeholder
        raise RuntimeError(
            f"[DRY-RUN FAIL] Only {total_bytes} bytes written — this is a placeholder, not real weights."
        )

    print("\n  ✅ DRY-RUN PASSED — full training pipeline is wired correctly.")
    print("  → Run train_curriculum() on a GPU for a production model (> 50 MB).")
    return trainer


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LifeStack GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (CPU, no GPU needed)
  python train_trl.py --dry-run

  # Fresh full run
  python train_trl.py --stages 5 --prompts-per-stage 200

  # Resume after Colab / Kaggle session cut
  python train_trl.py --resume

  # Manually restart from stage 3
  python train_trl.py --start-stage 3
        """
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 training step on 4 prompts to validate the full pipeline (no GPU required)."
    )
    parser.add_argument(
        "--stages", type=int, default=5,
        help="Number of curriculum stages (default: 5)."
    )
    parser.add_argument(
        "--prompts-per-stage", type=int, default=100,
        help="Prompts per curriculum stage (default: 100)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./lifestack_model",
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last saved checkpoint + curriculum_state.json."
    )
    parser.add_argument(
        "--start-stage", type=int, default=None,
        help="Force-start from a specific stage number (1-indexed). Ignores curriculum_state.json."
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(output_dir="./lifestack_model_dryrun")
    else:
        trainer = train_curriculum(
            n_stages=args.stages,
            n_prompts_per_stage=args.prompts_per_stage,
            output_dir=args.output_dir,
            resume=args.resume,
            start_stage=args.start_stage,
        )
        validate_saved_model(args.output_dir)
        evaluate_and_plot(args.output_dir)
