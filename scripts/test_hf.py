
import os
import sys
from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.agent import LifeStackAgent
from core.life_state import LifeMetrics, ResourceBudget
from intake.simperson import SimPerson
from agent.conflict_generator import ConflictEvent, TEMPLATES

def test_hf_connection():
    load_dotenv()
    print("📢 Testing Hugging Face 'Golden Model' Connectivity...")
    
    agent = LifeStackAgent()
    metrics = LifeMetrics()
    budget = ResourceBudget()
    conflict = TEMPLATES[0]
    person = SimPerson(name="Test User", openness=0.5, conscientiousness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
    
    try:
        # Force API only to skip local loading
        print("🚀 Sending request to Hugging Face...")
        action = agent.get_action(metrics, budget, conflict, person, api_only=True)
        
        print(f"📡 Model Used: {action.model_used}")
        
        if action.model_used.startswith("hf:"):
            print("✅ SUCCESS: Inference through Hugging Face Golden Model confirmed.")
            print(f"Agent Reasoning: {action.reasoning[:100]}...")
        elif "FALLBACK" in action.reasoning:
            print("❌ FAILED: System returned a hard fallback action.")
            print(f"Error Log: {action.reasoning}")
        else:
            print("⚠️ SEMI-SUCCESS: System fell back to Groq, but reasoning is intact.")
            print(f"Agent Reasoning: {action.reasoning[:100]}...")
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_hf_connection()
