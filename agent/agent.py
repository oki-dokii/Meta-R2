import os
import re
import json
import copy
import threading
import time as _t
from openai import OpenAI
from core.life_state import LifeMetrics, ResourceBudget
from core.metric_schema import format_valid_metrics, normalize_metric_path, is_valid_metric_path
from agent.conflict_generator import ConflictEvent, generate_conflict
from core.action_space import AgentAction, PrimaryAction, CommunicationAction, apply_action
from intake.simperson import SimPerson

DEFAULT_HF_MODEL_REPO = os.getenv("LIFESTACK_HF_MODEL", "jdsb06/lifestack-grpo-v4")   # latest trained GRPO adapter

class LifeStackAgent:
    def __init__(self, local_model_path: str = None, api_only: bool = False):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')

        # v4 base is Qwen2.5-1.5B (~3GB VRAM) — fits on T4 alongside Flask+ChromaDB.
        # api_only is now only forced when explicitly requested, not by on_hf_spaces.
        self.api_only = api_only
        self.local_model_path = local_model_path or os.getenv('LIFESTACK_MODEL_PATH')

        if not self.api_only and not self.local_model_path and os.path.exists("./lifestack_model"):
            self.local_model_path = "./lifestack_model"

        if not self.api_only and not self.local_model_path:
            self.local_model_path = DEFAULT_HF_MODEL_REPO

        if self.api_key:
            self.client = OpenAI(
                base_url='https://api.groq.com/openai/v1',
                api_key=self.api_key
            )
        self.model = 'llama-3.3-70b-versatile'
        self.tokenizer = None
        self.local_model = None
        self._model_load_attempted = False
        self._last_model_name = "unknown"
        self.memory = []

        # Eager background model load — starts loading weights immediately at
        # init so the first real request doesn't pay the cold-start penalty.
        if not self.api_only and self.local_model_path:
            import threading
            threading.Thread(target=self._try_load_model, daemon=True).start()

    def _try_load_model(self):
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
                dtype=dtype,          # was torch_dtype — fixed deprecation
                device_map=device_map,
            )
            self.local_model = PeftModel.from_pretrained(base_model, self.local_model_path)
            self.local_model.eval()
            # Clear max_length from the model's default generation config so that
            # our explicit max_new_tokens= doesn't trigger a "both are set" warning.
            if hasattr(self.local_model, 'generation_config'):
                self.local_model.generation_config.max_length = None
            device_label = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"✅ GRPO adapter loaded on {device_label}.")
        except Exception as e:
            print(f"⚠️ Failed to load local model: {e}. Falling back to APIs.")
            self.local_model_path = None

    # ── Prompt builder ────────────────────────────────────────────────────────

    def build_prompt(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, few_shot_context: str = "") -> str:
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

        memory_str = ""
        if self.memory:
            recent = self.memory[-2:]
            memory_str = "\n--- RECENT HISTORY ---\n"
            for mem in recent:
                memory_str += f"Past decision that worked: [{mem['action']}] → reward [{mem['reward']}]\n"

        persona_hint = person.get_personality_hint() if person else ""

        prompt = f"""ROLE: You are the LifeStack AI Agent. Your goal is to help the user navigate a life crisis.

CURRENT CONFLICT:
Title: {conflict.title}
Story: {conflict.story}

SUBJECT PERSONA:
{persona_hint}
Choose an action that fits this person's personality and coping style.

--- LIFE STATUS BOARD ---
{status_board}
--- RESOURCES REMAINING ---
Time: {budget.time_hours:.1f} hours
Money: ${budget.money_dollars:.1f}
Energy: {budget.energy_units:.1f} units
{memory_str}
{few_shot_context}
TASK:
Respond with a SINGLE LINE of minified JSON only. No newlines inside the JSON. No pretty-printing. No prose. No markdown. No trailing commas.

OUTPUT FORMAT (one line, no whitespace between fields):
{{"action_type":"<type>","target_domain":"<domain>","metric_changes":{{"domain.submetric":<number>}},"resource_cost":{{"time":<n>,"money":<n>,"energy":<n>}},"description":"<one sentence>","recipient":"none","message_content":"","reasoning":"<one sentence>"}}

VALID action_type values: communicate, rest, delegate, negotiate, spend, reschedule, deprioritize, prepare, self_care
VALID target_domain values: career, finances, relationships, physical_health, mental_wellbeing, time
STRATEGY: Prioritize high-agency actions (delegate/negotiate/prepare). Use 'prepare' for exams/deadlines. Use 'self_care' for emotional stability. Use 'rest' ONLY if energy < 30. Avoid generic advice.
"""
        return prompt

    # ── JSON extraction with multi-stage repair ───────────────────────────────

    def _extract_json(self, raw: str) -> dict:
        """Strip formatting, extract the first complete JSON object, repair common LLM quirks."""
        text = raw.strip()

        # Stage 1: strip markdown code fences
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()

        # Stage 2: extract the first balanced { ... } block
        start = text.find('{')
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break

        # Stage 3: fix common LLM formatting violations
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r',\s*([}\]])', r'\1', text)          # trailing commas
        text = re.sub(r"(?<![\"\\])'([^']*?)'", r'"\1"', text)  # single → double quotes

        return json.loads(text)

    _VALID_ACTION_TYPES = {
        "negotiate", "communicate", "delegate", "spend",
        "reschedule", "rest", "deprioritize", "execute",
        "prepare", "self_care"
    }
    # Map out-of-vocab types the model sometimes generates to the nearest valid one.
    _ACTION_TYPE_MAP = {
        "plan":        "prepare",
        "work":        "execute",
        "study":       "execute",
        "exercise":    "rest",
        "workout":     "rest",
        "sleep":       "rest",
        "relax":       "rest",
        "save":        "deprioritize",
        "cut":         "deprioritize",
        "talk":        "communicate",
        "call":        "communicate",
        "meet":        "communicate",
        "buy":         "spend",
        "hire":        "delegate",
        "assign":      "delegate",
        "postpone":    "reschedule",
        "delay":       "reschedule",
        "bargain":     "negotiate",
        "compromise":  "negotiate",
    }

    def _normalize_action_type(self, raw_type: str) -> str:
        t = (raw_type or "execute").lower().strip()
        if t in self._VALID_ACTION_TYPES:
            return t
        if t in self._ACTION_TYPE_MAP:
            mapped = self._ACTION_TYPE_MAP[t]
            print(f"[action_type] '{raw_type}' → '{mapped}' (normalised)")
            return mapped
        # Prefix match as last resort
        for valid in self._VALID_ACTION_TYPES:
            if t.startswith(valid) or valid.startswith(t):
                print(f"[action_type] '{raw_type}' → '{valid}' (prefix match)")
                return valid
        print(f"[action_type] '{raw_type}' unknown → 'execute'")
        return "execute"

    # ── Backend inference dispatcher ──────────────────────────────────────────

    def _run_inference(self, prompt: str, temperature: float = 0.3, force_api: bool = False,
                       trained_model_only: bool = False, max_new_tokens: int = 192) -> str | None:
        """
        Call the best available backend and return raw text.

        Priority order:
          1. Local GRPO adapter weights (loaded at startup in background thread)
          2. Groq 70B (general-purpose fallback — skipped when trained_model_only=True)

        trained_model_only=True is used for counterfactual generation so that
        the What-If Lab always reflects the trained model's policy, never Groq.
        """
        import torch
        content = None

        # 1. Local GRPO adapter weights (dev machine / local GPU)
        if self.local_model and not force_api:
            self._last_model_name = self.local_model_path
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            content = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

        # 2. Groq 70B — general fallback only (never used for counterfactuals)
        # Note: HF Serverless removed — it doesn't support LoRA adapters;
        # local model weights (priority 1) are the only trained-model path.
        if content is None and not trained_model_only and hasattr(self, 'client'):
            self._last_model_name = f"groq:{self.model}"
            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=512,
                        timeout=20,
                    )
                    content = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err and attempt == 0:
                        wait_secs = 6.0
                        m = re.search(r'try again in (\d+)m([\d.]+)s', err)
                        if m:
                            wait_secs = int(m.group(1)) * 60 + float(m.group(2))
                        else:
                            m2 = re.search(r'try again in ([\d.]+)s', err)
                            if m2:
                                wait_secs = float(m2.group(1))
                        if wait_secs > 3.0:
                            return None
                        _t.sleep(wait_secs)
                    else:
                        raise

        return content

    # ── Public API ────────────────────────────────────────────────────────────

    def get_action_for_type(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, forced_type: str, api_only: bool = False) -> "AgentAction":
        """
        Generate an action of a specific type for What-If Lab counterfactuals.

        v3 was trained on episode format {"actions":[...]} not single-action format,
        so we ask it in episode format and extract the first action.  If the model
        output cannot be parsed as an episode (i.e. local/HF model gives generic text),
        the Groq fallback produces a properly-formatted single action.
        """
        force_api = self.api_only or api_only
        if not force_api and not self._model_load_attempted:
            self._try_load_model()

        base_prompt = self.build_prompt(metrics, budget, conflict, person)

        # Ask v3 in its native episode format: one action, forced type
        episode_prompt = (
            base_prompt
            + f'\n\nEpisode objective: return a compact JSON object with an actions list.\n'
            + f'Plan exactly 1 action. The action_type MUST be "{forced_type}".\n'
            + f'Schema: {{"actions":[{{"action_type":"{forced_type}","target_domain":"<domain>",'
            + '"metric_changes":{"domain.submetric":0},"resource_cost":{"time":0,"money":0,"energy":0},'
            + '"reasoning":"brief","description":"one sentence"}}]}}'
        )

        # Try trained model first (HF v3 via InferenceClient or local weights)
        raw = self._run_inference(episode_prompt, temperature=0.4, force_api=force_api,
                                  trained_model_only=False, max_new_tokens=192)

        # Try to extract first action from episode {"actions":[...]} format
        if raw:
            try:
                import re as _re
                m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if m:
                    outer = json.loads(m.group())
                    actions = outer.get("actions") if isinstance(outer.get("actions"), list) else None
                    if not actions:
                        # Maybe it returned a single action directly
                        actions = [outer] if outer.get("action_type") else None
                    if actions:
                        a = actions[0]
                        # Validate and normalise metric_changes
                        metric_changes = {}
                        for k, v in a.get("metric_changes", {}).items():
                            norm_key = normalize_metric_path(k)
                            if is_valid_metric_path(norm_key):
                                try:
                                    metric_changes[norm_key] = float(v)
                                except (ValueError, TypeError):
                                    pass
                        return AgentAction(
                            primary=PrimaryAction(
                                action_type=self._normalize_action_type(a.get("action_type", forced_type)),
                                target_domain=a.get("target_domain", "mental_wellbeing"),
                                metric_changes=metric_changes,
                                resource_cost=a.get("resource_cost", {}),
                                description=a.get("description", a.get("reasoning", ""))
                            ),
                            communication=None,
                            reasoning=a.get("reasoning", ""),
                            model_used=self._last_model_name,
                            raw_completion=raw
                        )
            except Exception:
                pass  # fall through to Groq

        # Groq fallback: single-action format, which Groq handles reliably
        single_prompt = (
            base_prompt
            + f"\n\nCRITICAL REQUIREMENT: You MUST set 'action_type' to exactly '{forced_type}'."
        )
        return self._get_action_from_prompt(single_prompt, fallback_type=forced_type,
                                            force_api=force_api, trained_model_only=False)

    def get_action(self, metrics: LifeMetrics, budget: ResourceBudget, conflict: ConflictEvent, person: SimPerson, few_shot_context: str = "", api_only: bool = False, timeout: int = 55) -> "AgentAction":
        force_api = self.api_only or api_only
        if not force_api and not self._model_load_attempted:
            self._try_load_model()

        if not self.local_model and not self.api_key and not self.hf_token:
            return self._fallback_action("Error: No model configured (set GROQ_API_KEY, HF_TOKEN, or LIFESTACK_MODEL_PATH).")

        prompt = self.build_prompt(metrics, budget, conflict, person, few_shot_context)
        effective_timeout = 25 if force_api else timeout
        # Agent's Choice must always come from the trained model — never Groq.
        return self._get_action_from_prompt(prompt, force_api=force_api, timeout=effective_timeout,
                                            trained_model_only=not force_api)

    # ── Core inference + parse loop ───────────────────────────────────────────

    def _get_action_from_prompt(self, prompt: str, fallback_type: str = "rest", force_api: bool = False,
                                timeout: int = 55, trained_model_only: bool = False) -> "AgentAction":
        result_box = [None]

        def _call():
            try:
                content = self._run_inference(prompt, temperature=0.3, force_api=force_api,
                                              trained_model_only=trained_model_only, max_new_tokens=192)
                if not content:
                    result_box[0] = self._fallback_action("No content returned.", fallback_type)
                    return

                # Parse with one automatic retry on failure
                data = None
                for parse_attempt in range(2):
                    try:
                        data = self._extract_json(content)
                        break
                    except (json.JSONDecodeError, ValueError) as parse_err:
                        if parse_attempt == 0:
                            print(f"JSON parse attempt 1 failed ({parse_err}). Retrying with strict prompt...")
                            retry_prompt = prompt + "\n\nRETURN ONLY VALID COMPACT JSON. NO PROSE. NO MARKDOWN. NO TRAILING COMMAS."
                            retry_content = self._run_inference(retry_prompt, temperature=0.1, force_api=force_api,
                                                                trained_model_only=trained_model_only, max_new_tokens=192)
                            if retry_content:
                                content = retry_content
                        else:
                            print(f"LLM call error (both attempts failed): {parse_err}")
                            result_box[0] = self._fallback_action(f"JSON parse failed: {parse_err}", fallback_type)
                            return

                if data is None:
                    result_box[0] = self._fallback_action("JSON extraction returned None.", fallback_type)
                    return

                metric_changes = {}
                for k, v in data.get("metric_changes", {}).items():
                    norm_key = normalize_metric_path(k)
                    if is_valid_metric_path(norm_key):
                        try:
                            metric_changes[norm_key] = float(v)
                        except (ValueError, TypeError):
                            pass

                result_box[0] = AgentAction(
                    primary=PrimaryAction(
                        action_type=self._normalize_action_type(data.get("action_type", "execute")),
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
                    model_used=self._last_model_name,
                    raw_completion=content
                )

            except Exception as e:
                print(f"LLM call error: {e}")
                result_box[0] = self._fallback_action(f"Exception: {e}", fallback_type)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if result_box[0] is None:
            return self._fallback_action("LLM timed out.", fallback_type)
        return result_box[0]

    def _fallback_action(self, error_msg: str, fallback_type: str = "rest") -> "AgentAction":
        target = "mental_wellbeing"
        change = {"mental_wellbeing.stress_level": -5.0}
        
        # More intelligent fallback: if workload is the issue, try to deprioritize
        if "workload" in error_msg.lower() or "career" in error_msg.lower():
            fallback_type = "deprioritize"
            target = "career"
            change = {"career.workload": -10.0}

        return AgentAction(
            primary=PrimaryAction(
                action_type=fallback_type, target_domain=target,
                metric_changes=change,
                resource_cost={"energy": 5.0},
                description="Adjusting course to manage immediate pressure."
            ),
            reasoning=f"ADAPTIVE FALLBACK: {error_msg}"
        )

    def store_decision(self, action: AgentAction, reward: float):
        self.memory.append({'action': action.primary.description, 'reward': round(reward, 3)})
        if len(self.memory) > 10:
            self.memory.pop(0)


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
