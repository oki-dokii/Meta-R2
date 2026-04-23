"""
smoke_test.py — Remote Environment Verification
Checks if the simulation engine is correctly installed and functional.
"""

import os
import sys

# Ensure parent directory is in path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def smoke_test():
    print("🔍 Starting LifeStack Remote Smoke Test...")
    
    try:
        from core.lifestack_env import LifeStackEnv, LifeStackAction
        from agent.conflict_generator import TaskGenerator
        print("✅ Core modules imported successfully.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

    try:
        env = LifeStackEnv()
        generator = TaskGenerator()
        task = generator.generate(domain="flight_crisis", difficulty=1)
        obs = env.reset(task=task)
        print(f"✅ Environment reset successful (Task: {task.goal})")
        
        # Test a simple action
        action = LifeStackAction(
            action_type="rest",
            target="mental_wellbeing",
            metric_changes={"mental_wellbeing.stress_level": -5.0},
            resource_cost={}
        )
        obs = env.step(action)
        print(f"✅ Environment step successful (Step: {obs.step}, Reward: {obs.reward:.4f})")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        sys.exit(1)

    print("\n🚀 SMOKE TEST PASSED: LifeStack is ready for deployment.")

if __name__ == "__main__":
    smoke_test()
