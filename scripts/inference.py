"""
LifeStack Model — Inference Script
===================================
Usage:
    python scripts/inference.py --model ./lifestack_model
    python scripts/inference.py --model ./lifestack_model --scenario "My car broke down and I have a meeting in 2 hours"
    python scripts/inference.py --model ./lifestack_model --interactive
"""

import argparse
import json
import re
import torch
from transformers import AutoTokenizer

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_dir: str):
    """Load the LoRA adapter on top of base Qwen2.5-1.5B.
    Tries Unsloth first (2x faster), falls back to standard PEFT.
    """
    try:
        from unsloth import FastLanguageModel
        print("Loading with Unsloth (fast)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("✅ Loaded via Unsloth")
        return model, tokenizer

    except ImportError:
        print("Unsloth not installed — using standard PEFT (slower)...")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
        print("✅ Loaded via PEFT")
        return model, tokenizer


# ── Build prompt ───────────────────────────────────────────────────────────────

SYSTEM = (
    "You are LifeStack, an AI life-management agent. "
    "Given a real-life crisis, respond with a single optimal action as valid JSON.\n\n"
    "Required JSON format:\n"
    '{"action_type": "negotiate|communicate|delegate|spend|reschedule|rest|deprioritize|execute", '
    '"target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time|transport_crisis|flight_crisis|code_merge_crisis", '
    '"metric_changes": {"domain.submetric": delta_value}, '
    '"resource_cost": {"time": hours, "money": dollars, "energy": units}, '
    '"reasoning": "brief explanation of why this is the best action"}'
)

def build_prompt(scenario: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"CRISIS: {scenario}\n\n"
        f"Respond with ONLY valid JSON — no markdown, no explanation outside the JSON.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class JsonCompleteStopping(torch.nn.Module):
    """Stop once the first complete JSON object closes."""

    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        depth = 0
        entered = False
        for ch in text:
            if ch == "{":
                depth += 1
                entered = True
            elif ch == "}":
                depth -= 1
            if entered and depth == 0:
                return True
        return False


def extract_json_payload(completion: str) -> dict:
    """Parse the first complete JSON object, ignoring trailing text."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", completion.strip()):
        try:
            obj, _ = decoder.raw_decode(completion[match.start():].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {"raw_output": completion, "parse_error": "no valid JSON object found"}


# ── Inference ──────────────────────────────────────────────────────────────────

def resolve(model, tokenizer, scenario: str, temperature: float = 0.3) -> dict:
    prompt = build_prompt(scenario)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[JsonCompleteStopping(tokenizer, prompt_len)],
        )

    # Decode only the new tokens (not the prompt)
    completion = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True
    ).strip()

    return extract_json_payload(completion)


# ── Built-in scenarios for quick demo ─────────────────────────────────────────

DEMO_SCENARIOS = [
    "My flight got cancelled and my card got declined at the rebooking desk. I have a client presentation tomorrow morning.",
    "My car broke down on the highway. The repair will take 3 days and I have no other transport.",
    "I haven't slept properly in 2 weeks. My productivity is shot and my partner says I'm distant.",
    "A surprise tax audit letter arrived. I owe $4,000 I don't have liquid.",
    "My boss just dropped a 12-hour task on me at 5PM Friday and said it's due Monday morning.",
    "The morning train is delayed 90 minutes and I have a 9AM client meeting I can't miss.",
    "I've been double-booked every weekend for a month and I can't say no to anyone.",
    "A critical git merge broke the staging environment 2 hours before a demo.",
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LifeStack Inference")
    parser.add_argument("--model",       type=str, default="./lifestack_model",
                        help="Path to the downloaded/unzipped model directory")
    parser.add_argument("--scenario",    type=str, default=None,
                        help="Describe your crisis (quoted string)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive REPL — type your own crises")
    parser.add_argument("--demo",        action="store_true",
                        help="Run all 8 built-in demo scenarios")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature (default 0.3 = focused)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  LifeStack — AI Life Management Agent")
    print(f"{'='*60}\n")

    model, tokenizer = load_model(args.model)
    print()

    if args.demo:
        for i, scenario in enumerate(DEMO_SCENARIOS, 1):
            print(f"[{i}/8] {scenario[:80]}...")
            action = resolve(model, tokenizer, scenario, args.temperature)
            print(json.dumps(action, indent=2))
            print()

    elif args.interactive:
        print("Interactive mode — type your crisis and press Enter.")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                scenario = input("Crisis > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if scenario.lower() in ("quit", "exit", "q"):
                break
            if not scenario:
                continue
            action = resolve(model, tokenizer, scenario, args.temperature)
            print("\nLifeStack Action:")
            print(json.dumps(action, indent=2))
            print()

    elif args.scenario:
        action = resolve(model, tokenizer, args.scenario, args.temperature)
        print("LifeStack Action:")
        print(json.dumps(action, indent=2))

    else:
        # Default: run 1 demo scenario
        scenario = DEMO_SCENARIOS[0]
        print(f"Demo scenario: {scenario}\n")
        action = resolve(model, tokenizer, scenario, args.temperature)
        print("LifeStack Action:")
        print(json.dumps(action, indent=2))


if __name__ == "__main__":
    main()
