import os
import json
import copy
from openai import OpenAI
from core.life_state import LifeMetrics, ResourceBudget
from core.metric_schema import format_valid_metrics, normalize_metric_path, is_valid_metric_path
from agent.conflict_generator import ConflictEvent, generate_conflict
from core.action_space import AgentAction, PrimaryAction, CommunicationAction, apply_action
from intake.simperson import SimPerson

DEFAULT_HF_MODEL_REPO = "jdsb06/lifestack-grpo"

class LifeStackAgent:
    def __init__(self, local_model_path: str = None, api_only: bool = False):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        self.api_only = api_only  # if True, always use Groq, never load local model
        self.local_model_path = local_model_path or os.getenv('LIFESTACK_MODEL_PATH')

        # 1. Check for local folder (Kaggle / local dev)
        if not self.api_only and not self.local_model_path and os.path.exists("./lifestack_model"):
            self.local_model_path = "./lifestack_model"

        # 2. Fall back to HuggingFace Hub
        if not self.api_only and not self.local_model_path:
            self.local_model_path = DEFAULT_HF_MODEL_REPO

        # Wire up HF Inference API (Premium Priority - Direct Protocol)
        from huggingface_hub import InferenceClient
        self.hf_client = None
        if self.hf_token:
            print("🚀 HF_TOKEN found. Prioritizing Direct Hugging Face Inference.")
            self.hf_client = InferenceClient(token=self.hf_token)
        self.hf_model = os.getenv("LIFESTACK_HF_MODEL", DEFAULT_HF_MODEL_REPO)

        # Wire up Groq as a fallback
        if self.api_key:
            self.client = OpenAI(
                base_url='https://api.groq.com/openai/v1',
                api_key=self.api_key
            )
        self.model = 'llama-3.3-70b-versatile'
        self.tokenizer = None
        self.local_model = None
        self._model_load_attempted = False
        self.memory = []  # Will store last 10 decisions

    def _try_load_model(self):
        """Attempt to load the local/HF model lazily on first inference call."""
        self._model_load_attempted = True
        if not self.local_model_path:
            return
        try:
            print(f"📦 Loading GRPO model from {self.local_model_path}...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftConfig, PeftModel

            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            peft_config = PeftConfig.from_pretrained(self.local_model_path)
            base_model_name = peft_config.base_model_name_or_path

            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )
            self.local_model = PeftModel.from_pretrained(base_model, self.local_model_path)
            self.local_model.eval()
            device_label = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"✅ GRPO adapter loaded on {device_label}.")
        except Exception as e:
            print(f"⚠️ Failed to load local model: {e}. Falling back to APIs.")
            self.local_model_path = None

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
Choose the best action to address the conflict. Respond ONLY with valid JSON following the schema below.

SCHEMA:
{{
  "action_type": "communicate|rest|delegate|negotiate|spend|reschedule|deprioritize",
  "target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time",
  "metric_changes": {{"domain.submetric": "delta_value"}},
  "resource_cost": {{"time": 0.0, "money": 0.0, "energy": 0.0}},
  "description": "one sentence action",
  "recipient": "none|boss|partner|family",
  "message_content": "text",
  "reasoning": "strategy explanation"
}}
"""
        return prompt

    def get_action_for_type(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, forced_type: str, api_only: bool = False) -> "AgentAction":
        """Generate an action specifically for a given action_type."""
        force_api = self.api_only or api_only
        if not force_api and not self._model_load_attempted:
            self._try_load_model()
        base_prompt = self.build_prompt(metrics, budget, conflict, person)
        forced_prompt = base_prompt + f"\n\nCRITICAL REQUIREMENT: You MUST set 'action_type' to exactly '{forced_type}'."
        return self._get_action_from_prompt(forced_prompt, fallback_type=forced_type, force_api=force_api)

    def get_action(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, few_shot_context: str = "", api_only: bool = False) -> "AgentAction":
        # Lazy-load the trained model on first real inference, unless caller forces api_only.
        force_api = self.api_only or api_only
        if not force_api and not self._model_load_attempted:
            self._try_load_model()

        if not self.local_model and not self.api_key and not self.hf_token:
            return self._fallback_action("Error: No model configured (set GROQ_API_KEY, HF_TOKEN, or LIFESTACK_MODEL_PATH).")

        prompt = self.build_prompt(metrics, budget, conflict, person, few_shot_context)
        return self._get_action_from_prompt(prompt, force_api=force_api)

    def _get_action_from_prompt(self, prompt: str, fallback_type: str = "rest", force_api: bool = False) -> "AgentAction":
        """Run LLM inference inside a daemon thread with a hard 25-second timeout."""
        import threading
        import time as _t
        import re

        result_box = [None]  # thread writes its result here

        def _call():
            try:
                import torch
                content = None
                
                used_model_name = "unknown"
                if self.local_model and not force_api:
                    # ── Local / HF Transformers model ─────────────────────
                    used_model_name = self.local_model_path
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
                    with torch.no_grad():
                        outputs = self.local_model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.3,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    content = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                
                elif self.hf_client:
                    # ── Hugging Face Inference API (Golden Pool) ──────────
                    used_model_name = f"hf:{self.hf_model}"
                    try:
                        content = self.hf_client.text_generation(
                            prompt,
                            model=self.hf_model,
                            max_new_tokens=350,
                            temperature=0.3
                        )
                        if prompt in content:
                            content = content.replace(prompt, "").strip()
                    except Exception as hf_err:
                        print(f"⚠️ HF Inference Error: {hf_err}. Falling back to Groq.")
                
                if content is None:
                    # ── Groq API Fallback (Llama-3.3-70B) ──────────────────
                    used_model_name = f"groq:{self.model}"
                    response = None
                    for attempt in range(2):
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=350,
                                timeout=20,
                            )
                            break
                        except Exception as e:
                            err = str(e)
                            if "429" in err and attempt == 0:
                                wait_secs = 6.0
                                m = re.search(r'try again in (\d+)m([\d.]+)s', err)
                                if m: wait_secs = int(m.group(1)) * 60 + float(m.group(2))
                                elif re.search(r'try again in ([\d.]+)s', err): 
                                    wait_secs = float(re.search(r'try again in ([\d.]+)s', err).group(1))
                                if wait_secs > 3.0:
                                    result_box[0] = self._fallback_action(f"Rate limited ({wait_secs:.0f}s).", fallback_type)
                                    return
                                _t.sleep(wait_secs)
                            else: raise
                    
                    if response:
                        content = response.choices[0].message.content.strip()

                if content:
                    # Parse JSON
                    if "```json" in content: content = content.split("```json")[-1].split("```")[0].strip()
                    elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(content)
                    metric_changes = {}
                    for k, v in data.get("metric_changes", {}).items():
                        norm_key = normalize_metric_path(k)
                        if is_valid_metric_path(norm_key):
                            try: metric_changes[norm_key] = float(v)
                            except (ValueError, TypeError): pass
                    
                    result_box[0] = AgentAction(
                        primary=PrimaryAction(
                            action_type=data.get("action_type", "rest"),
                            target_domain=data.get("target_domain", "mental_wellbeing"),
                            metric_changes=metric_changes,
                            resource_cost=data.get("resource_cost", {}),
                            description=data.get("description", "Taking a moment.")
                        ),
                        communication=CommunicationAction(
                            recipient=data.get("recipient"),
                            message_type=data.get("message_type") or "none",
                            tone=data.get("tone") or "none",
                            content=data.get("message_content") or ""
                        ) if data.get("recipient") and data.get("recipient") != "none" else None,
                        reasoning=data.get("reasoning", "Strategic choice."),
                        model_used=used_model_name,
                        raw_completion=content
                    )
            except Exception as e:
                print(f"LLM call error: {e}")
                result_box[0] = self._fallback_action(f"Exception: {e}", fallback_type)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=25)

        if result_box[0] is None:
            return self._fallback_action("LLM timed out.", fallback_type)
        return result_box[0]

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
        self.memory.append({'action': action.primary.description, 'reward': round(reward, 3)})
        if len(self.memory) > 10: self.memory.pop(0)

def main():
    if not os.getenv('GROQ_API_KEY'):
        print("CRITICAL ERROR: GROQ_API_KEY environment variable is not set.")
        return
    agent = LifeStackAgent()
    person = SimPerson(name="Sam (Introvert)", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.9)
    conflict = generate_conflict(difficulty=3)
    metrics = LifeMetrics()
    budget = ResourceBudget()
    print(f"--- GENERATING ACTION FOR: {conflict.title} ---")
    action = agent.get_action(metrics, budget, conflict, person)
    print(f"\nType: {action.primary.action_type} | Reasoning: {action.reasoning}")

if __name__ == "__main__":
    main()
