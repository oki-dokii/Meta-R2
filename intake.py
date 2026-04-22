"""
intake.py — LifeStack Conversational Onboarding
Extracts a structured life state, conflict, and personality profile
from a user's natural language description + slider inputs.
"""

import os
import json
from openai import OpenAI
from life_state import LifeMetrics, ResourceBudget
from conflict_generator import ConflictEvent


class LifeIntake:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        # Fallback to .env file
        if not self.api_key and os.path.exists(".env"):
            try:
                with open(".env") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass

        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key,
            )
        self.model = "llama-3.1-8b-instant"
        self.conversation_history = []

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Internal LLM call with basic retry on rate limit."""
        import time as _t
        import re

        if not self.client:
            return ""

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content.strip()
                # Strip markdown fences if present
                if content.startswith("```json"):
                    content = content[7:].rsplit("```", 1)[0].strip()
                elif content.startswith("```"):
                    content = content[3:].rsplit("```", 1)[0].strip()
                return content
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < 2:
                    wait_secs = 5.0
                    m = re.search(r"try again in (\d+)m([\d.]+)s", err)
                    if m:
                        wait_secs = int(m.group(1)) * 60 + float(m.group(2))
                    else:
                        m = re.search(r"try again in ([\d.]+)s", err)
                        if m:
                            wait_secs = float(m.group(1))
                    if wait_secs > 5.0:
                        print(f"  ⚠️  Rate limit — skipping LLM call ({wait_secs:.0f}s wait)")
                        return ""
                    _t.sleep(wait_secs)
                else:
                    print(f"  ⚠️  LLM call failed: {e}")
                    return ""
        return ""

    # ─── 1. Slider → LifeMetrics ──────────────────────────────────────────────
    def extract_life_state(
        self,
        user_description: str,
        work_stress: int,
        money_stress: int,
        relationship_quality: int,
        energy_level: int,
        time_pressure: int,
    ) -> LifeMetrics:
        """
        Maps slider values (0-10) directly to life metrics and returns
        a fully populated LifeMetrics object.
        """
        def clamp(v: float) -> float:
            return max(0.0, min(100.0, v))

        metrics = LifeMetrics()

        # Career
        metrics.career.workload               = clamp(50 + work_stress * 5)
        # (other career fields stay at 70)

        # Mental wellbeing
        metrics.mental_wellbeing.stress_level = clamp(40 + work_stress * 6)

        # Finances
        metrics.finances.liquidity            = clamp(100 - money_stress * 7)
        metrics.finances.debt_pressure        = clamp(40 + money_stress * 5)

        # Relationships
        metrics.relationships.romantic        = clamp(relationship_quality * 10)
        metrics.relationships.social          = clamp(40 + relationship_quality * 4)

        # Physical health
        metrics.physical_health.energy        = clamp(energy_level * 10)
        metrics.physical_health.sleep_quality = clamp(30 + energy_level * 7)

        # Time
        metrics.time.free_hours_per_week      = clamp(100 - time_pressure * 8)

        return metrics

    # ─── 2. NL description → ConflictEvent ───────────────────────────────────
    def extract_conflict(self, user_description: str, metrics: LifeMetrics) -> ConflictEvent:
        """
        Sends the user description + key metric snapshot to the LLM
        and parses the response into a structured ConflictEvent.
        """
        flat = metrics.flatten()
        stress     = flat.get("mental_wellbeing.stress_level", 70)
        liquidity  = flat.get("finances.liquidity", 70)
        energy     = flat.get("physical_health.energy", 70)
        free_hours = flat.get("time.free_hours_per_week", 70)

        valid_paths = (
            "career.workload, career.satisfaction, career.stability, career.growth_trajectory, "
            "finances.liquidity, finances.debt_pressure, finances.monthly_runway, finances.long_term_health, "
            "relationships.romantic, relationships.family, relationships.social, relationships.professional_network, "
            "physical_health.energy, physical_health.fitness, physical_health.sleep_quality, physical_health.nutrition, "
            "mental_wellbeing.stress_level, mental_wellbeing.clarity, mental_wellbeing.motivation, mental_wellbeing.emotional_stability, "
            "time.free_hours_per_week, time.commute_burden, time.admin_overhead"
        )
        prompt = (
            f"The user described their situation as: {user_description}\n"
            f"Their life metrics show: stress={stress:.1f}, liquidity={liquidity:.1f}, "
            f"energy={energy:.1f}, free_hours={free_hours:.1f}.\n"
            "Extract a structured conflict. Respond ONLY with valid JSON (no markdown fences).\n"
            f"Use ONLY these exact metric path keys for primary_disruption: {valid_paths}\n"
            '{"title": "2-4 word title", "story": "one sentence description of the crisis", '
            '"primary_disruption": {"exact.metric_path": delta_as_float}, '
            '"decisions_required": ["option1", "option2", "option3"], '
            '"difficulty": integer_from_1_to_5}'
        )

        raw = self._call_llm(prompt, max_tokens=400)

        try:
            data = json.loads(raw)
            disruption = {}
            for k, v in data.get("primary_disruption", {}).items():
                try:
                    disruption[k] = float(v)
                except (ValueError, TypeError):
                    pass

            return ConflictEvent(
                id="custom_intake",
                title=str(data.get("title", "Your Situation")),
                story=str(data.get("story", user_description)),
                primary_disruption=disruption or {"mental_wellbeing.stress_level": 20.0},
                decisions_required=list(data.get("decisions_required", ["Take action", "Seek help", "Rest"])),
                resource_budget={"time": 10.0, "money": 200.0, "energy": 50.0},
                difficulty=int(data.get("difficulty", 3)),
            )
        except Exception as e:
            print(f"  ⚠️  Conflict parsing failed ({e}). Using fallback.")
            return ConflictEvent(
                id="custom_intake",
                title="Your Situation",
                story=user_description or "Feeling overwhelmed and unsure what to do.",
                primary_disruption={"mental_wellbeing.stress_level": 20.0},
                decisions_required=["Take action", "Seek help", "Rest"],
                resource_budget={"time": 10.0, "money": 200.0, "energy": 50.0},
                difficulty=3,
            )

    # ─── 3. NL description → OCEAN personality dict ───────────────────────────
    def get_personality_from_description(self, user_description: str) -> dict:
        """
        Infers OCEAN personality trait scores from the user's natural
        language description. Returns a dict or balanced defaults on failure.
        """
        prompt = (
            f"Based on this description of someone's situation:\n{user_description}\n\n"
            "Infer their likely OCEAN personality traits as float values between 0.0 and 1.0. "
            "Also infer a likely first name that fits the personality. "
            "Respond ONLY with valid JSON, no extra text:\n"
            '{"openness": 0.65, "conscientiousness": 0.75, '
            '"extraversion": 0.30, "agreeableness": 0.55, '
            '"neuroticism": 0.80, "name": "Sam"}'
        )

        raw = self._call_llm(prompt, max_tokens=200)

        defaults = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            "name": "You",
        }

        try:
            data = json.loads(raw)
            result = {}
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                try:
                    result[trait] = float(data[trait])
                except (KeyError, ValueError, TypeError):
                    result[trait] = defaults[trait]
            result["name"] = str(data.get("name", "You"))
            return result
        except Exception as e:
            print(f"  ⚠️  Personality parsing failed ({e}). Using balanced defaults.")
            return defaults

    # ─── 4. Full intake — single entry point for app.py Tab 2 ─────────────────
    def full_intake(
        self,
        user_description: str,
        work_stress: int,
        money_stress: int,
        relationship_quality: int,
        energy_level: int,
        time_pressure: int,
        calendar_signals: dict = None,
        gmail_signals: dict = None,
    ) -> tuple:
        """
        Runs all three extraction steps and returns:
            (LifeMetrics, ResourceBudget, ConflictEvent, personality_dict)
        """
        metrics = self.extract_life_state(
            user_description, work_stress, money_stress,
            relationship_quality, energy_level, time_pressure
        )

        # Apply Gmail/Calendar signal adjustments if provided
        signals = {}
        if calendar_signals: signals.update(calendar_signals)
        if gmail_signals: signals.update(gmail_signals)

        for path, val in signals.items():
            if '.' not in path: continue
            domain_name, sub_name = path.split('.')
            domain = getattr(metrics, domain_name, None)
            if domain and hasattr(domain, sub_name):
                # Signals like social/romantic/network from Gmail are treated as base values (overrides)
                # while others like stress/free_time are cumulative deltas.
                if any(x in sub_name for x in ["social", "romantic", "network", "professional"]):
                    setattr(domain, sub_name, max(0.0, min(100.0, val)))
                else:
                    current = getattr(domain, sub_name)
                    setattr(domain, sub_name, max(0.0, min(100.0, current + val)))

        conflict    = self.extract_conflict(user_description, metrics)
        personality = self.get_personality_from_description(user_description)
        budget      = ResourceBudget()

        return metrics, budget, conflict, personality


# ─── Main test ────────────────────────────────────────────────────────────────
def main():
    description = (
        "My boss keeps piling on work and I haven't slept properly in weeks. "
        "My partner says I am distant and I don't have the energy to fix it."
    )
    work_stress         = 8
    money_stress        = 4
    relationship_quality = 5
    energy_level        = 3
    time_pressure       = 7

    print("🚀 Running LifeIntake...\n")
    intake = LifeIntake()
    metrics, budget, conflict, personality = intake.full_intake(
        description, work_stress, money_stress,
        relationship_quality, energy_level, time_pressure
    )

    print("-" * 50)
    print("📊 EXTRACTED LIFE METRICS")
    print("-" * 50)
    flat = metrics.flatten()
    for key, val in flat.items():
        icon = "🟢" if val > 70 else ("🟡" if val >= 40 else "🔴")
        print(f"  {icon} {key:40}: {val:.1f}")

    print("\n─" * 50)
    print("⚡ EXTRACTED CONFLICT")
    print("-" * 50)
    print(f"  Title      : {conflict.title}")
    print(f"  Difficulty : {conflict.difficulty}/5")
    print(f"  Story      : {conflict.story}")
    print(f"  Disruption : {conflict.primary_disruption}")
    print(f"  Options    : {conflict.decisions_required}")

    print("\n─" * 50)
    print("🧠 INFERRED PERSONALITY")
    print("-" * 50)
    for trait, val in personality.items():
        if trait != "name":
            print(f"  {trait:20}: {val:.2f}")
    print(f"  {'name':20}: {personality['name']}")

    print(f"\n✅ Budget — Time: {budget.time_hours}h | Money: ${budget.money_dollars} | Energy: {budget.energy_units}")


if __name__ == "__main__":
    main()
