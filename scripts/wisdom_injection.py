
import os
import sys
import time
import random
from tqdm import tqdm

# Add the project root to sys.path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.life_state import LifeMetrics, ResourceBudget
from core.lifestack_env import LifeStackEnv, LifeStackAction
from agent.agent import LifeStackAgent
from agent.memory import LifeStackMemory
from agent.conflict_generator import generate_conflict, TEMPLATES
from intake.simperson import PERSONS

def inject_wisdom(count=200):
    print(f"🚀 Starting Wisdom Injection: Generating {count} expert precedents...")
    
    # Initialize components
    agent = LifeStackAgent(api_only=True) # Use Groq for speed and variety
    memory = LifeStackMemory(silent=True)
    
    # Track stats
    stored_count = 0
    start_time = time.time()
    
    # We'll vary the "Person" to get different reasoning styles
    person_list = list(PERSONS.values())
    
    for i in tqdm(range(count)):
        try:
            # 1. Setup a fresh random environment
            env = LifeStackEnv()
            
            # Randomize difficulty and persona
            difficulty = random.randint(2, 5)
            person = random.choice(person_list)
            conflict = generate_conflict(difficulty=difficulty)
            
            # Reset env with these parameters
            env.reset(conflict=conflict.primary_disruption, budget=conflict.resource_budget)
            
            before_metrics = env.state.current_metrics
            before_budget = env.state.budget
            
            # 2. Let the Agent solve it
            # We don't use few-shot here because we want "raw" expert reasoning to seed the memory
            action = agent.get_action(before_metrics, before_budget, conflict, person)
            
            # 3. Simulate performance
            # Simple uptake logic based on persona
            uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost, 
                                              before_metrics.mental_wellbeing.stress_level)
            
            env_action = LifeStackAction.from_agent_action(action)
            # Scale metric changes by persona uptake
            env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
            
            obs = env.step(env_action)
            
            # 4. Store ONLY if the reward is decent (Wisdom should be smart)
            if obs.reward > 0.4:
                memory.store_decision(
                    conflict_title=conflict.title,
                    action_type=action.primary.action_type,
                    target_domain=action.primary.target_domain,
                    reward=obs.reward,
                    metrics_snapshot=before_metrics.flatten(),
                    reasoning=action.reasoning
                )
                stored_count += 1
            
            # 5. Throttle to stay within Groq rate limits (approx 3 calls per minute for free tier)
            # Adjust sleep based on actual throughput
            time.sleep(1.5) 
            
        except Exception as e:
            if "429" in str(e):
                print(f"\n⚠️ Rate limit hit at step {i}. Waiting 30s...")
                time.sleep(30)
            else:
                print(f"\n❌ Error at step {i}: {e}")
                continue

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ Wisdom Injection Complete!")
    print(f"   - Total Attempted: {count}")
    print(f"   - Expert Precedents Stored: {stored_count}")
    print(f"   - Time taken: {duration:.1f}s")
    print(f"Memory now contains {memory.collection.count()} high-quality traces.")

if __name__ == "__main__":
    # Start with 50 first to ensure stability, then we can run more if time permits
    # 200 might take 20+ minutes due to rate limits
    inject_wisdom(count=200)
