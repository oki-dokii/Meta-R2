import os
import json
import copy
from openai import OpenAI
from core.life_state import LifeMetrics, ResourceBudget
from core.metric_schema import format_valid_metrics, normalize_metric_path, is_valid_metric_path
from agent.conflict_generator import ConflictEvent, generate_conflict
from core.action_space import AgentAction, PrimaryAction, CommunicationAction, apply_action
from intake.simperson import SimPerson

class LifeStackAgent:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        
        # Fallback to .env file if env var is missing
        if not self.api_key and os.path.exists('.env'):
            try:
                with open('.env') as f:
                    for line in f:
                        if line.startswith('GROQ_API_KEY='):
                            self.api_key = line.split('=', 1)[1].strip()
                            break
            except Exception:
                pass

        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url='https://api.groq.com/openai/v1',
                api_key=self.api_key
            )
        self.model = 'llama-3.1-8b-instant'  # 500k TPD on free tier (vs 100k for 70b)
        self.memory = [] # Will store last 10 decisions

    def build_prompt(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, few_shot_context: str = "") -> str:
        # 1. Build Status Board
        flat = metrics.flatten()
        status_board = ""
        domains = ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"]
        
        for dom in domains:
            status_board += f"\n{dom.upper()}:\n"
            submetrics = {k: v for k, v in flat.items() if k.startswith(dom + ".")}
            for k, v in submetrics.items():
                name = k.split('.')[1]
                icon = "🟢" if v > 70 else ("🟡" if v >= 40 else "🔴")
                status_board += f"  {icon} {name:20}: {v:.1f}\n"

        # 2. Build Memory Section
        memory_str = ""
        if self.memory:
            recent = self.memory[-2:]
            memory_str = "\n--- RECENT HISTORY ---\n"
            for mem in recent:
                memory_str += f"Past decision that worked: [{mem['action']}] → reward [{mem['reward']}]\n"

        prompt = f"""
ROLE: You are the LifeStack AI Agent. Your goal is to help the user navigate a life crisis.
PERSONALITY CONTEXT: Observe how the person responds to your actions over time. You do not have direct access to their personality traits.

CURRENT CONFLICT:
Title: {conflict.title}
Story: {conflict.story}

--- LIFE STATUS BOARD ---
{status_board}

--- RESOURCES REMAINING ---
Time: {budget.time_hours:.1f} hours
Money: ${budget.money_dollars:.1f}
Energy: {budget.energy_units:.1f} units
{memory_str}
{few_shot_context}

TASK:
Choose the best action to address the conflict. Consider the person's personality and resource constraints.
Respond ONLY with valid JSON following the schema below. No markdown fences, no extra text.

VALID METRICS (use ONLY these exact keys in metric_changes):
{format_valid_metrics()}

SCHEMA:
{{
  "action_type": "communicate|rest|delegate|negotiate|spend|reschedule|deprioritize",
  "target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time",
  "metric_changes": {{"domain.submetric": "delta_value (use ONLY valid metrics above)"}},
  "resource_cost": {{"time": 0.0, "money": 0.0, "energy": 0.0}},
  "description": "one sentence what you are doing",
  "recipient": "boss|partner|family|friend|colleague|none",
  "message_type": "apologize|negotiate|inform|request|reassure|none",
  "tone": "formal|warm|urgent|calm|assertive|none",
  "message_content": "actual message text or empty string",
  "reasoning": "why this action helps most given the personality and resources"
}}
"""
        return prompt

    def get_action_for_type(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, forced_type: str) -> "AgentAction":
        """Generate an action specifically for a given action_type."""
        base_prompt = self.build_prompt(metrics, budget, conflict, person)
        forced_prompt = base_prompt + f"\n\nCRITICAL REQUIREMENT: You MUST set 'action_type' to exactly '{forced_type}'."
        return self._get_action_from_prompt(forced_prompt, fallback_type=forced_type)

    def get_action(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, few_shot_context: str = "") -> "AgentAction":
        if not self.api_key:
            return self._fallback_action("Error: GROQ_API_KEY not set.")

        prompt = self.build_prompt(metrics, budget, conflict, person, few_shot_context)
        return self._get_action_from_prompt(prompt)

    def _get_action_from_prompt(self, prompt: str, fallback_type: str = "rest") -> "AgentAction":
        import time as _t
        try:
            for attempt in range(3):  # Up to 3 retries on rate limit
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=350
                    )
                    break  # Success
                except Exception as e:
                    err = str(e)
                    if "429" in err and attempt < 2:
                        import re
                        # Handle formats: "2m10.5s", "10.5s", "240ms"
                        wait_secs = 30  # default
                        m = re.search(r'try again in (\d+)m([\d.]+)s', err)
                        if m:
                            wait_secs = int(m.group(1)) * 60 + float(m.group(2))
                        else:
                            m = re.search(r'try again in ([\d.]+)s', err)
                            if m:
                                wait_secs = float(m.group(1))
                            else:
                                m = re.search(r'try again in ([\d.]+)ms', err)
                                if m:
                                    wait_secs = float(m.group(1)) / 1000.0
                        wait_secs = max(0.5, min(wait_secs + 0.5, 120))  # add 0.5s buffer, cap at 2min
                        if wait_secs > 5.0:
                            print(f"  ⏳ Rate limit wait too long ({wait_secs:.1f}s). Executing fallback...")
                            return self._fallback_action(f"Rate limited (wait: {wait_secs:.1f}s)")
                        print(f"  ⏳ Rate limit hit. Waiting {wait_secs:.1f}s before retry {attempt+2}/3...")
                        _t.sleep(wait_secs)
                    else:
                        raise  # Re-raise if not 429 or out of retries
            _t.sleep(1)  # Light throttle — retry logic handles actual TPM limits
            content = response.choices[0].message.content.strip()
            
            # Clean possible markdown fences
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            
            # Cast values to float in case the LLM returned strings
            metric_changes = data.get("metric_changes", {})
            normalized_metric_changes = {}
            for k, v in list(metric_changes.items()):
                norm_key = normalize_metric_path(k)
                if not is_valid_metric_path(norm_key):
                    continue
                try:
                    normalized_metric_changes[norm_key] = float(v)
                except ValueError:
                    continue
            metric_changes = normalized_metric_changes
                    
            resource_cost = data.get("resource_cost", {})
            for k, v in list(resource_cost.items()):
                try:
                    resource_cost[k] = float(v)
                except ValueError:
                    resource_cost[k] = 0.0
            
            # Build structures
            primary = PrimaryAction(
                action_type=data.get("action_type", "rest"),
                target_domain=data.get("target_domain", "mental_wellbeing"),
                metric_changes=metric_changes,
                resource_cost=resource_cost,
                description=data.get("description", "Taking a moment to breathe.")
            )
            
            comm = None
            if data.get("recipient") and data.get("recipient") != "none":
                comm = CommunicationAction(
                    recipient=data["recipient"],
                    message_type=data.get("message_type", "inform"),
                    tone=data.get("tone", "calm"),
                    content=data.get("message_content", "")
                )
                
            return AgentAction(
                primary=primary,
                communication=comm,
                reasoning=data.get("reasoning", "Decided to rest due to high pressure."),
                raw_completion=content
            )
            
        except Exception as e:
            print(f"Error calling LLM or parsing response: {e}")
            return self._fallback_action(f"Exception: {str(e)}", fallback_type=fallback_type)

    def _fallback_action(self, error_msg: str, fallback_type: str = "rest") -> "AgentAction":
        return AgentAction(
            primary=PrimaryAction(
                action_type=fallback_type, target_domain="mental_wellbeing",
                metric_changes={"mental_wellbeing.stress_level": -5.0},
                resource_cost={},
                description="Short breather to regain composure."
            ),
            reasoning=f"FALLBACK: {error_msg}"
        )

    def store_decision(self, action: AgentAction, reward: float):
        self.memory.append({
            'action': action.primary.description,
            'reward': round(reward, 3)
        })
        if len(self.memory) > 10:
            self.memory.pop(0)

def main():
    if not os.getenv('GROQ_API_KEY'):
        print("CRITICAL ERROR: GROQ_API_KEY environment variable is not set.")
        return

    # Initialize components
    agent = LifeStackAgent()
    person = SimPerson(name="Sam (Introvert)", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.9)
    conflict = generate_conflict(difficulty=3)
    metrics = LifeMetrics()
    budget = ResourceBudget()
    
    print(f"--- GENERATING ACTION FOR: {conflict.title} ---")
    print(f"Context: {conflict.story}")
    print(f"Human: {person.get_personality_hint()}")
    
    # Get Decision
    action = agent.get_action(metrics, budget, conflict, person)
    
    print("\n--- AGENT DECISION ---")
    print(f"Type:      {action.primary.action_type}")
    print(f"Domain:    {action.primary.target_domain}")
    print(f"Description: {action.primary.description}")
    print(f"Cost:      {action.primary.resource_cost}")
    print(f"Changes:   {action.primary.metric_changes}")
    
    if action.communication:
        print(f"Message:   [{action.communication.recipient}] ({action.communication.tone}) {action.communication.content}")
        
    print(f"\nReasoning: {action.reasoning}")

if __name__ == "__main__":
    main()
