"""
train_trl.py — HuggingFace TRL PPO Training Script for LifeStack.
Uses Qwen/Qwen2-0.5B to train a local RL agent on the LifeStack environment.
"""

import torch
import json
import re
import copy
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from lifestack_env import LifeStackEnv
from simperson import SimPerson
from conflict_generator import generate_conflict
from action_space import AgentAction, PrimaryAction, CommunicationAction

def parse_action(text: str) -> AgentAction:
    try:
        # Extract JSON from model output
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        
        metric_changes = {}
        for k, v in data.get("metric_changes", {}).items():
            metric_changes[k] = float(v)
            
        resource_cost = {}
        for k, v in data.get("resource_cost", {}).items():
            resource_cost[k] = float(v)
            
        primary = PrimaryAction(
            action_type=data.get("action_type", "rest"),
            target_domain=data.get("target_domain", "mental_wellbeing"),
            metric_changes=metric_changes,
            resource_cost=resource_cost,
            description=data.get("description", "")
        )
        comm = None
        if data.get("recipient") and data.get("recipient") != "none":
            comm = CommunicationAction(
                recipient=data["recipient"],
                message_type=data.get("message_type", "inform"),
                tone=data.get("tone", "calm"),
                content=data.get("message_content", "")
            )
        return AgentAction(primary=primary, communication=comm, reasoning=data.get("reasoning", ""))
    except Exception:
        return None

def reward_func(response: str, env: LifeStackEnv, person: SimPerson) -> float:
    action = parse_action(response)
    if not action:
        return -0.5 # Parsing failed
        
    current_stress = env.state.mental_wellbeing.stress_level
    uptake_score = person.respond_to_action(
        action.primary.action_type, 
        action.primary.resource_cost, 
        current_stress
    )
    scaled_changes = {}
    for path, delta in action.primary.metric_changes.items():
        if '.' not in path:
            path = f"{action.primary.target_domain}.{path}"
        scaled_changes[path] = delta * uptake_score
        
    env_action = {
        "metric_changes": scaled_changes,
        "resource_cost": action.primary.resource_cost,
        "actions_taken": 1
    }
    obs, step_reward, terminated, truncated, env_info = env.step(env_action)
    return step_reward

def main():
    set_seed(42)
    model_name = "Qwen/Qwen2-0.5B"
    
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        ppo_epochs=1,
        optimize_device_cache=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading model. This may take a moment...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    device = ppo_trainer.accelerator.device
    print(f"Using device: {device}")
    
    sys_prompt = "You are a life coach agent. Given a life crisis, suggest one action in JSON format."
    
    # Test items for evaluation
    test_env = LifeStackEnv()
    test_person = SimPerson()
    test_conflict = generate_conflict(difficulty=3)
    test_env.reset(conflict=test_conflict.primary_disruption)
    
    test_user_prompt = f"Graph: {test_env.state.flatten()}\nBudget: {test_env.budget}\nConflict: {test_conflict.story}\nFormat: JSON {{ 'action_type': '...', 'description': '...', 'metric_changes': {{...}} }}"
    test_query = f"System: {sys_prompt}\nUser: {test_user_prompt}\nAssistant:"
    
    test_q_tensor = tokenizer.encode(test_query, return_tensors="pt").to(device)[0]
    
    print("Capturing untrained response...")
    with torch.no_grad():
        untrained_response_tensor = ppo_trainer.generate([test_q_tensor], return_prompt=False, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]
        untrained_text = tokenizer.decode(untrained_response_tensor, skip_special_tokens=True)
        untrained_reward = reward_func(untrained_text, copy.deepcopy(test_env), test_person)

    epoch_rewards = []
    
    print("Starting PPO training loop...")
    for step in range(50):
        query_tensors = []
        envs = []
        persons = []
        
        for _ in range(config.batch_size):
            e = LifeStackEnv()
            c = generate_conflict(difficulty=3)
            p = SimPerson()
            e.reset(conflict=c.primary_disruption)
            q = f"System: {sys_prompt}\nUser: Graph: {e.state.flatten()}\nConflict: {c.story}\nAssistant:"
            q_tensor = tokenizer.encode(q, return_tensors="pt").to(device)[0]
            query_tensors.append(q_tensor)
            envs.append(e)
            persons.append(p)
            
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        
        rewards = []
        for i in range(config.batch_size):
            resp_str = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
            r = reward_func(resp_str, envs[i], persons[i])
            rewards.append(torch.tensor(r, dtype=torch.float32).to(device))
            
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        avg_reward = sum([r.item() for r in rewards]) / len(rewards)
        epoch_rewards.append(avg_reward)
        kl = stats.get('objective/kl', 0) if isinstance(stats, dict) else 0.0
        print(f"Step {step+1}/50 | Avg Reward: {avg_reward:.3f} | KL: {kl:.3f}")
        
    print("Saving model to ./lifestack_model...")
    ppo_trainer.save_pretrained("./lifestack_model")
    tokenizer.save_pretrained("./lifestack_model")
    
    print("Capturing trained response...")
    with torch.no_grad():
        trained_response_tensor = ppo_trainer.generate([test_q_tensor], return_prompt=False, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]
        trained_text = tokenizer.decode(trained_response_tensor, skip_special_tokens=True)
        trained_reward = reward_func(trained_text, copy.deepcopy(test_env), test_person)
        
    print("\n" + "="*60)
    print("  BEFORE / AFTER PPO TRAINING COMPARISON")
    print("="*60)
    print(f"Untrained Reward : {untrained_reward:.3f}")
    print(f"Untrained Output : {untrained_text}\n")
    print("-" * 60)
    print(f"Trained Reward   : {trained_reward:.3f}")
    print(f"Trained Output   : {trained_text}")
    print("="*60)

    # Save plot for Colab notebook
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 51), epoch_rewards, marker='o', linestyle='-', color='b')
    plt.title("TRL PPO Training: Reward over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Average Batch Reward")
    plt.grid(True)
    plt.savefig("trl_reward_curve.png", dpi=150)
    print("Training curve saved to trl_reward_curve.png")

if __name__ == "__main__":
    main()
