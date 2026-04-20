import random
import json
import math
from dataclasses import dataclass, field, asdict

@dataclass
class SimPerson:
    openness: float = field(default_factory=lambda: random.uniform(0, 1))
    conscientiousness: float = field(default_factory=lambda: random.uniform(0, 1))
    extraversion: float = field(default_factory=lambda: random.uniform(0, 1))
    agreeableness: float = field(default_factory=lambda: random.uniform(0, 1))
    neuroticism: float = field(default_factory=lambda: random.uniform(0, 1))
    name: str = "Anonymous"

    def respond_to_action(self, action_type: str, resource_cost: dict, current_stress: float) -> float:
        """
        Determines how likely the person is to successfully 'uptake' an action.
        Uptake determines effectiveness of metric changes.
        """
        uptake = 0.70
        
        # Stress interaction
        stress_penalty = 0.0
        if current_stress > 70:
            stress_penalty = 0.20
            # Personality amplification of stress
            if self.neuroticism > 0.7:
                stress_penalty *= 1.3
        
        uptake -= stress_penalty
        
        # Personality-Action alignment
        if action_type == 'communicate' and self.agreeableness > 0.6:
            uptake += 0.15
        
        if action_type == 'structured_plan' and self.conscientiousness > 0.7:
            uptake += 0.10
            
        if action_type == 'delegate' and self.neuroticism > 0.7:
            uptake -= 0.10
            
        if action_type == 'rest' and self.extraversion < 0.4:
            uptake += 0.10
            
        return max(0.1, min(1.0, uptake))

    def drift(self, timestep: int) -> dict:
        """Personality shifts slightly over time, occasionally triggering events."""
        if timestep % 5 != 0 or timestep == 0:
            return {}
            
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        trait = random.choice(traits)
        change = random.choice([0.05, -0.05])
        
        current = getattr(self, trait)
        setattr(self, trait, max(0.0, min(1.0, current + change)))
        
        return {
            'metric': 'career.satisfaction', 
            'delta': -5, 
            'reason': f'Minor internal shift in {trait} causing professional friction.'
        }

    def get_personality_hint(self) -> str:
        """Returns a human-readable summary of the person's personality and tendencies."""
        traits = []
        if self.openness > 0.7: traits.append("intellectually curious")
        elif self.openness < 0.3: traits.append("grounded in tradition")
        
        if self.conscientiousness > 0.7: traits.append("highly organized")
        elif self.conscientiousness < 0.3: traits.append("spontaneous/relaxed")
        
        if self.extraversion > 0.7: traits.append("energetic/social")
        elif self.extraversion < 0.3: traits.append("reserved/introspective")
        
        if self.agreeableness > 0.7: traits.append("deeply cooperative")
        elif self.agreeableness < 0.3: traits.append("skeptical/competitive")
        
        if self.neuroticism > 0.7: traits.append("anxious/sensitive")
        elif self.neuroticism < 0.3: traits.append("emotionally resilient")
        
        trait_str = ", ".join(traits) if traits else "balanced"
        
        # Strategy hints
        strategies = []
        if self.conscientiousness > 0.7: strategies.append("structured plans")
        if self.agreeableness > 0.6: strategies.append("open communication")
        if self.extraversion < 0.4: strategies.append("quiet rest")
        
        hint = f"{self.name} is {trait_str}."
        if strategies:
            hint += f" Responds best to {', '.join(strategies)}."
        if self.neuroticism > 0.7:
            hint += " Caution: Heavily impacted by high stress."
            
        return hint

def generate_and_save_profiles():
    """Generates 5 diverse profiles as requested."""
    profiles_data = [
        {
            "name": "Alex (High-Stress Executive)",
            "openness": 0.4, "conscientiousness": 0.9, "extraversion": 0.7, 
            "agreeableness": 0.25, "neuroticism": 0.8
        },
        {
            "name": "Chloe (Laid-Back Creative)",
            "openness": 0.9, "conscientiousness": 0.2, "extraversion": 0.5, 
            "agreeableness": 0.7, "neuroticism": 0.15
        },
        {
            "name": "Sam (Anxious Introvert)",
            "openness": 0.5, "conscientiousness": 0.6, "extraversion": 0.1, 
            "agreeableness": 0.65, "neuroticism": 0.9
        },
        {
            "name": "Maya (Balanced Family Person)",
            "openness": 0.5, "conscientiousness": 0.7, "extraversion": 0.5, 
            "agreeableness": 0.95, "neuroticism": 0.3
        },
        {
            "name": "Leo (Ambitious Student)",
            "openness": 0.85, "conscientiousness": 0.8, "extraversion": 0.4, 
            "agreeableness": 0.4, "neuroticism": 0.55
        }
    ]
    
    with open('simperson_profiles.json', 'w') as f:
        json.dump(profiles_data, f, indent=4)
    print("Saved 5 diverse profiles to simperson_profiles.json")

def main():
    # 1. Setup
    generate_and_save_profiles()
    
    # 2. Create 3 test instances
    test_people = [
        SimPerson(name="Alex (Executive)", openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe (Creative)", openness=0.9, conscientiousness=0.2, extraversion=0.5, agreeableness=0.7, neuroticism=0.15),
        SimPerson(name="Sam (Introvert)", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.9)
    ]
    
    actions = ['communicate', 'structured_plan', 'delegate', 'rest']
    stress_levels = [30.0, 80.0]
    
    print("\n--- PERSONALITY ANALYSIS ---")
    for person in test_people:
        print(f"\n[{person.name}]")
        print(f"Hint: {person.get_personality_hint()}")
        
        print(f"{'Action':20} | {'Low Stress (30)':15} | {'High Stress (80)':15}")
        print("-" * 55)
        for action in actions:
            uptake_low = person.respond_to_action(action, {}, 30.0)
            uptake_high = person.respond_to_action(action, {}, 80.0)
            print(f"{action:20} | {uptake_low:15.2f} | {uptake_high:15.2f}")

if __name__ == "__main__":
    main()
