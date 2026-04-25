"""
Interactive Gradio demo for LifeStack trained model.

Usage:
  python scripts/gradio_demo.py --model-dir ./lifestack_model
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Any

import gradio as gr
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from agent.conflict_generator import TaskGenerator, generate_conflict
from core.life_state import (
    CASCADE_DAMPENING_DEFAULT,
    DependencyGraph,
    LifeMetrics,
    ResourceBudget,
)
from intake.simperson import SimPerson
from scripts.inference import load_model
from scripts.train_trl import ALL_DOMAINS, build_prompt_for_task, generate_dataset, get_lifestack_evaluation

MODEL = None
TOKENIZER = None
MODEL_DIR = "./lifestack_model"


def _device_for(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model_loaded():
    global MODEL, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = load_model(MODEL_DIR)


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[-1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
        return {"json_value": data}
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception as err:
                return {"raw_output": text, "parse_error": str(err)}
        return {"raw_output": text, "parse_error": "no valid JSON object found"}


class _JsonCompleteStopping(torch.nn.Module):
    """Stop generation as soon as the first complete JSON object is closed.

    Tracks open/close brace balance across already-decoded tokens.
    When balance drops to 0 after having been > 0, the JSON is complete
    and we signal the trainer to stop.
    """

    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # Only look at tokens generated after the prompt
        gen_ids = input_ids[0][self.prompt_len:]
        if len(gen_ids) == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        depth = 0
        entered = False
        for ch in text:
            if ch == "{":
                depth += 1
                entered = True
            elif ch == "}":
                depth -= 1
            if entered and depth == 0:
                return True  # first complete JSON object closed — stop
        return False


def _generate_completion(prompt: str, temperature: float = 0.3) -> str:
    _ensure_model_loaded()
    device = _device_for(MODEL)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(device)
    pad_token_id = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else TOKENIZER.eos_token_id
    prompt_len = inputs["input_ids"].shape[1]

    stopper = _JsonCompleteStopping(TOKENIZER, prompt_len)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=160,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            stopping_criteria=[stopper],
        )

    raw = TOKENIZER.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    # Extract only the first complete JSON object — discard any trailing text
    import re as _re
    import json as _json
    decoder = _json.JSONDecoder()
    for _m in _re.finditer(r"\{", raw):
        try:
            obj, end = decoder.raw_decode(raw[_m.start():].strip())
            if isinstance(obj, dict):
                return _json.dumps(obj)
        except _json.JSONDecodeError:
            continue
    return raw  # fallback: return as-is if no JSON found


def _build_crisis_prompt(crisis_text: str, domain: str, difficulty: int) -> tuple[str, dict[str, float]]:
    generator = TaskGenerator()
    graph = DependencyGraph()
    person = SimPerson(name="DemoUser")

    eval_seed = random.randint(1, 999999)
    random.seed(eval_seed)
    task = generator.generate(domain=domain, difficulty=int(difficulty))
    conflict = generate_conflict(int(difficulty))
    random.seed()

    if crisis_text.strip():
        task.goal = crisis_text.strip()
        task.domain_metadata["story"] = crisis_text.strip()

    metrics = LifeMetrics()
    metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})
    budget_dict = task.constraints.get("budget", {})
    budget = ResourceBudget(
        time_hours=budget_dict.get("time", 20.0),
        money_dollars=budget_dict.get("money", 500.0),
        energy_units=budget_dict.get("energy", 100.0),
    )
    prompt = build_prompt_for_task(task, person, metrics, budget, seed=eval_seed, step=0)
    return prompt, dict(task.mutable_world)


def _select_metric_keys(before: dict[str, float], after: dict[str, float]) -> list[str]:
    priority = [
        "career.workload",
        "finances.liquidity",
        "relationships.romantic",
        "physical_health.energy",
        "mental_wellbeing.stress_level",
        "time.free_hours_per_week",
    ]
    keys = [k for k in priority if k in before or k in after]
    if len(keys) < 6:
        pool = sorted(set(before.keys()) | set(after.keys()))
        for k in pool:
            if k not in keys:
                keys.append(k)
            if len(keys) == 6:
                break
    return keys


def _plot_before_after(before: dict[str, float], after: dict[str, float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    if not before and not after:
        ax.text(0.5, 0.5, "No metric data available", ha="center", va="center")
        ax.axis("off")
        return fig

    keys = _select_metric_keys(before, after)
    x = range(len(keys))
    before_vals = [before.get(k, 0.0) for k in keys]
    after_vals = [after.get(k, 0.0) for k in keys]

    ax.bar([i - 0.2 for i in x], before_vals, width=0.4, label="Before", color="#9ca3af")
    ax.bar([i + 0.2 for i in x], after_vals, width=0.4, label="After", color="#16a34a")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(x))
    ax.set_xticklabels([k.split(".")[-1] for k in keys], rotation=20, ha="right")
    ax.set_title("Life Metrics Before vs After")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_trajectory(trajectory: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(8, 4))
    if not trajectory:
        ax.text(0.5, 0.5, "No trajectory data available", ha="center", va="center")
        ax.axis("off")
        return fig

    days = [point.get("step", idx + 1) for idx, point in enumerate(trajectory)]
    rewards = [point.get("reward", 0.0) for point in trajectory]
    stress = [point.get("metrics", {}).get("mental_wellbeing.stress_level", 0.0) for point in trajectory]

    ax.plot(days, rewards, marker="o", linewidth=2, color="#1d4ed8", label="Daily Reward")
    ax.set_xlabel("Day")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(days, stress, marker="s", linestyle="--", color="#dc2626", label="Stress Level")
    ax2.set_ylabel("Stress")

    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")
    ax.set_title("7-Day Trajectory")
    fig.tight_layout()
    return fig


def visualize_cascade(disruption_dict: dict[str, float]) -> str:
    """Render a lightweight ASCII cascade tree for a disruption dict."""
    graph = DependencyGraph()
    if not disruption_dict:
        return "No disruption provided."

    lines: list[str] = []
    for source_key, source_delta in disruption_dict.items():
        lines.append(f"{source_key} ({source_delta:+.1f})")
        level_1 = graph.edges.get(source_key, [])[:3]
        if not level_1:
            lines.append("  └─ (no downstream edges)")
            continue

        for i, (target_key, weight) in enumerate(level_1):
            branch = "└─" if i == len(level_1) - 1 else "├─"
            level_1_delta = source_delta * weight * CASCADE_DAMPENING_DEFAULT
            lines.append(f"  {branch} {target_key} (w={weight:+.2f}, est={level_1_delta:+.1f})")

            level_2 = graph.edges.get(target_key, [])[:2]
            for j, (target_2, weight_2) in enumerate(level_2):
                branch_2 = "└─" if j == len(level_2) - 1 else "├─"
                indent = "     " if i == len(level_1) - 1 else "  │  "
                level_2_delta = level_1_delta * weight_2 * CASCADE_DAMPENING_DEFAULT
                lines.append(f"{indent}{branch_2} {target_2} (w={weight_2:+.2f}, est={level_2_delta:+.1f})")
    return "\n".join(lines)


def _render_advice(action_json: dict[str, Any], reward: float, domain: str, difficulty: int) -> str:
    action_type = action_json.get("action_type", "unknown")
    target_domain = action_json.get("target_domain", "unknown")
    reasoning = action_json.get("reasoning", "")
    metric_changes = action_json.get("metric_changes", {})
    resource_cost = action_json.get("resource_cost", {})

    lines = [
        "### LifeStack Recommendation",
        f"- Domain: `{domain}` | Difficulty: `{difficulty}`",
        f"- Reward Score: `{reward:.3f}`",
        f"- Action: `{action_type}`",
        f"- Target: `{target_domain}`",
    ]
    if reasoning:
        lines.append(f"- Why: {reasoning}")
    if metric_changes:
        top_changes = list(metric_changes.items())[:5]
        lines.append("- Expected metric impact: " + ", ".join(f"`{k}` {v:+.1f}" for k, v in top_changes))
    if resource_cost:
        lines.append(
            "- Resource cost: "
            f"time={resource_cost.get('time', 0)}, "
            f"money={resource_cost.get('money', 0)}, "
            f"energy={resource_cost.get('energy', 0)}"
        )
    return "\n".join(lines)


def sample_random_crisis():
    ds = generate_dataset(n_prompts=1)
    row = ds[0]
    prompt = row["prompt"]
    domain = row.get("domain", "career")
    difficulty = int(row.get("difficulty", 3))

    m = re.search(r"(?:Task|TASK):\s*(.+)", prompt)
    crisis_text = m.group(1).strip() if m else "My life is spiraling in multiple domains. What should I do first?"
    return crisis_text, domain, difficulty


def run_live_demo(crisis_text: str, domain: str, difficulty: int):
    if not crisis_text or not crisis_text.strip():
        crisis_text = "I am facing a multi-domain crisis and need a single best next action."

    prompt, disruption = _build_crisis_prompt(crisis_text, domain, int(difficulty))
    completion = _generate_completion(prompt, temperature=0.4)
    action_json = _extract_json_payload(completion)
    eval_data = get_lifestack_evaluation(completion, prompt)

    reward = float(eval_data.get("reward", -0.5))
    before = eval_data.get("initial_metrics", {})
    after = eval_data.get("obs_metrics", {})
    trajectory = eval_data.get("trajectory", [])

    advice_md = _render_advice(action_json, reward, domain, int(difficulty))
    before_after_fig = _plot_before_after(before, after)
    trajectory_fig = _plot_trajectory(trajectory)
    cascade_tree = "```text\n" + visualize_cascade(disruption) + "\n```"

    return advice_md, action_json, before_after_fig, trajectory_fig, cascade_tree


def build_app():
    with gr.Blocks(title="LifeStack GRPO Demo") as demo:
        gr.Markdown("# LifeStack GRPO Demo")
        gr.Markdown("Resolve a crisis and inspect action quality, life metric impact, trajectory, and cascade effects.")

        with gr.Row():
            crisis_input = gr.Textbox(
                label="Describe your life crisis",
                lines=4,
                placeholder="My flight got cancelled, my card was declined, and I have a client meeting tomorrow.",
            )
        with gr.Row():
            domain_input = gr.Dropdown(choices=ALL_DOMAINS, value="career", label="Domain")
            difficulty_input = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Difficulty")

        with gr.Row():
            run_btn = gr.Button("Resolve Crisis", variant="primary")
            random_btn = gr.Button("Try Random Crisis")

        advice_out = gr.Markdown()
        action_json_out = gr.JSON(label="Model JSON Decision")
        with gr.Row():
            before_after_out = gr.Plot(label="Before/After Metrics")
            trajectory_out = gr.Plot(label="7-Day Trajectory")
        cascade_out = gr.Markdown()

        run_btn.click(
            fn=run_live_demo,
            inputs=[crisis_input, domain_input, difficulty_input],
            outputs=[advice_out, action_json_out, before_after_out, trajectory_out, cascade_out],
        )
        random_btn.click(
            fn=sample_random_crisis,
            inputs=[],
            outputs=[crisis_input, domain_input, difficulty_input],
        )

    return demo


def main():
    global MODEL_DIR

    parser = argparse.ArgumentParser(description="LifeStack Gradio demo.")
    parser.add_argument("--model-dir", type=str, default="./lifestack_model")
    parser.add_argument("--share", action="store_true", default=True, help="Launch with public share URL.")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio share URL.")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    demo = build_app()
    demo.launch(share=(args.share and not args.no_share), server_port=args.server_port)


if __name__ == "__main__":
    main()
