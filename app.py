"""
app.py — LifeStack Gradio Demo App
Hackathon presentation interface for the LifeStack simulation engine.
"""

import os
import json
import copy
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── LifeStack modules ────────────────────────────────────────────────────────
from life_state import LifeMetrics, ResourceBudget
from lifestack_env import LifeStackEnv
from agent import LifeStackAgent
from simperson import SimPerson
from conflict_generator import ConflictEvent, generate_conflict, TEMPLATES
from action_space import apply_action, validate_action
from memory import LifeStackMemory
from reward import compute_reward
from intake import LifeIntake
from conflict_predictor import ConflictPredictor

# ─── Pre-load at startup ──────────────────────────────────────────────────────
print("🚀 LifeStack booting…")

AGENT  = LifeStackAgent()
MEMORY = LifeStackMemory(silent=True)
INTAKE = LifeIntake()

# Friday 6PM is always the default demo conflict
DEMO_CONFLICT = next(t for t in TEMPLATES if t.id == "d5_friday")

PERSONS = {
    "Alex (Executive) — driven, high-stress":
        SimPerson(openness=0.4, conscientiousness=0.9, extraversion=0.7,  agreeableness=0.25, neuroticism=0.8,  name="Alex (Executive)"),
    "Chloe (Creative) — spontaneous, resilient":
        SimPerson(openness=0.9, conscientiousness=0.2, extraversion=0.5,  agreeableness=0.70, neuroticism=0.15, name="Chloe (Creative)"),
    "Sam (Introvert) — anxious, thoughtful":
        SimPerson(openness=0.5, conscientiousness=0.6, extraversion=0.1,  agreeableness=0.65, neuroticism=0.9,  name="Sam (Introvert)"),
    "Maya (Family) — empathetic, nurturing":
        SimPerson(openness=0.5, conscientiousness=0.7, extraversion=0.5,  agreeableness=0.95, neuroticism=0.3,  name="Maya (Family)"),
    "Leo (Student) — curious, organised":
        SimPerson(openness=0.85, conscientiousness=0.8, extraversion=0.4, agreeableness=0.4,  neuroticism=0.55, name="Leo (Student)"),
}

CONFLICT_CHOICES      = {f"[Diff {t.difficulty}] {t.title}": t for t in TEMPLATES}
PERSON_CHOICES        = list(PERSONS.keys())
CONFLICT_CHOICES_LIST = list(CONFLICT_CHOICES.keys())
DEFAULT_CONFLICT      = next(k for k in CONFLICT_CHOICES_LIST if "Friday 6PM" in k)

DEMO_PREDICTOR = ConflictPredictor()

print("✅ LifeStack ready.")

# ─── Helpers ──────────────────────────────────────────────────────────────────
DOMAIN_EMOJI = {
    "career": "💼", "finances": "💰", "relationships": "❤️",
    "physical_health": "💪", "mental_wellbeing": "🧠", "time": "📅",
}

# Metrics where HIGH = BAD (inverted color logic)
INVERTED_METRICS = {"stress_level", "debt_pressure", "workload", "commute_burden", "admin_overhead"}

def _metric_color(key: str, val: float) -> str:
    """Return CSS color: inverted for 'bad-when-high' metrics."""
    sub = key.split(".")[-1]
    if sub in INVERTED_METRICS:
        return "#f87171" if val > 70 else ("#facc15" if val >= 40 else "#4ade80")
    return "#4ade80" if val > 70 else ("#facc15" if val >= 40 else "#f87171")

def metrics_html(flat: dict, title: str = "", before: dict = None) -> str:
    """Render metrics as coloured progress bars.
    If `before` is supplied, metrics that changed >1 pt show ↑/↓ + delta.
    """
    domains = ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"]
    rows = []
    if title:
        rows.append(f"<h3 style='margin:0 0 8px;font-size:14px;color:#aaa'>{title}</h3>")
    for dom in domains:
        emoji = DOMAIN_EMOJI[dom]
        rows.append(f"<div style='margin:6px 0 2px;font-size:12px;font-weight:700;color:#ccc'>{emoji} {dom.upper()}</div>")
        sub = {k: v for k, v in flat.items() if k.startswith(dom + ".")}
        for key, val in sub.items():
            name  = key.split(".")[1].replace("_", " ")
            color = _metric_color(key, val)
            pct   = min(val, 100)

            delta_str = ""
            if before is not None and key in before:
                delta = val - before[key]
                if abs(delta) > 1.0:
                    arrow = "↑" if delta > 0 else "↓"
                    dc    = "#4ade80" if delta > 0 else "#f87171"
                    delta_str = (
                        f"<span style='font-size:10px;color:{dc};margin-left:4px;font-weight:700'>"
                        f"{arrow} ({delta:+.1f})</span>"
                    )

            rows.append(
                f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
                f"  <span style='width:140px;font-size:11px;color:#bbb'>{name}</span>"
                f"  <div style='flex:1;background:#333;border-radius:4px;height:10px'>"
                f"    <div style='width:{pct}%;background:{color};border-radius:4px;height:10px'></div>"
                f"  </div>"
                f"  <span style='width:38px;font-size:11px;color:#ccc;text-align:right'>{val:.1f}</span>"
                f"  {delta_str}"
                f"</div>"
            )
    return "<div style='font-family:monospace;padding:8px'>" + "\n".join(rows) + "</div>"


def _init_env(conflict: ConflictEvent) -> LifeStackEnv:
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget=conflict.resource_budget)
    return env


# ─── Cascade Animation Engine ────────────────────────────────────────────────

def animate_cascade(primary_disruption: dict, metrics: LifeMetrics) -> list[dict]:
    """Replay the cascade step-by-step and capture intermediate frames.

    Returns a list of frames. Each frame is:
      { 'flat': {metric: value}, 'status': {metric: 'primary'|'first'|'second'|'unchanged'} }
    """
    import copy as _cp
    from life_state import DependencyGraph, CASCADE_DAMPENING_DEFAULT

    graph = DependencyGraph()
    dampening = CASCADE_DAMPENING_DEFAULT
    frames = []

    # Frame 0 — initial stable state
    base = _cp.deepcopy(metrics)
    base_flat = base.flatten()
    frames.append({
        'flat': dict(base_flat),
        'status': {k: 'unchanged' for k in base_flat},
    })

    # Frame 1 — primary disruption only (no cascade)
    f1 = _cp.deepcopy(metrics)
    primary_keys = set()
    for path, amount in primary_disruption.items():
        if '.' not in path:
            continue
        primary_keys.add(path)
        dom_name, sub_name = path.split('.', 1)
        dom = getattr(f1, dom_name, None)
        if dom and hasattr(dom, sub_name):
            cur = getattr(dom, sub_name)
            setattr(dom, sub_name, max(0.0, min(100.0, cur + amount)))
    f1_flat = f1.flatten()
    f1_status = {}
    for k in f1_flat:
        f1_status[k] = 'primary' if k in primary_keys else 'unchanged'
    frames.append({'flat': dict(f1_flat), 'status': f1_status})

    # Frame 2 — first-order cascade effects
    f2 = _cp.deepcopy(f1)
    first_order_keys = set()
    queue_next = []
    for path, amount in primary_disruption.items():
        if '.' not in path:
            continue
        if path in graph.edges:
            for target, weight in graph.edges[path]:
                impact = amount * weight * dampening
                if abs(impact) >= 0.05:
                    first_order_keys.add(target)
                    dom_name, sub_name = target.split('.', 1)
                    dom = getattr(f2, dom_name, None)
                    if dom and hasattr(dom, sub_name):
                        cur = getattr(dom, sub_name)
                        setattr(dom, sub_name, max(0.0, min(100.0, cur + impact)))
                    queue_next.append((target, impact))
    f2_flat = f2.flatten()
    f2_status = {}
    for k in f2_flat:
        if k in primary_keys:
            f2_status[k] = 'primary'
        elif k in first_order_keys:
            f2_status[k] = 'first'
        else:
            f2_status[k] = 'unchanged'
    frames.append({'flat': dict(f2_flat), 'status': f2_status})

    # Frame 3 — second-order cascade effects
    f3 = copy.deepcopy(f2)
    second_order_keys = set()
    for src_path, src_mag in queue_next:
        if src_path in graph.edges:
            for target, weight in graph.edges[src_path]:
                impact = src_mag * weight * dampening
                if abs(impact) >= 0.05:
                    second_order_keys.add(target)
                    dom_name, sub_name = target.split('.', 1)
                    dom = getattr(f3, dom_name, None)
                    if dom and hasattr(dom, sub_name):
                        cur = getattr(dom, sub_name)
                        setattr(dom, sub_name, max(0.0, min(100.0, cur + impact)))
    f3_flat = f3.flatten()
    f3_status = {}
    for k in f3_flat:
        if k in primary_keys:
            f3_status[k] = 'primary'
        elif k in first_order_keys:
            f3_status[k] = 'first'
        elif k in second_order_keys:
            f3_status[k] = 'second'
        else:
            f3_status[k] = 'unchanged'
    frames.append({'flat': dict(f3_flat), 'status': f3_status})

    return frames


# Cascade-aware CSS colours
CASCADE_COLORS = {
    'primary':   '#ef4444',   # 🔴 red
    'first':     '#f97316',   # 🟠 orange
    'second':    '#eab308',   # 🟡 yellow
    'improved':  '#22c55e',   # 🟢 green
    'unchanged': '#6b7280',   # ⚪ grey
}

CASCADE_EMOJI = {
    'primary':   '🔴', 'first': '🟠', 'second': '🟡',
    'improved':  '🟢', 'unchanged': '⚪',
}


def cascade_metrics_html(flat: dict, status: dict, title: str = "",
                         before: dict = None) -> str:
    """Render metrics with cascade propagation colours."""
    domains = ["career", "finances", "relationships",
               "physical_health", "mental_wellbeing", "time"]
    rows = []
    if title:
        rows.append(f"<h3 style='margin:0 0 8px;font-size:14px;color:#aaa'>{title}</h3>")
    for dom in domains:
        emoji = DOMAIN_EMOJI[dom]
        rows.append(f"<div style='margin:6px 0 2px;font-size:12px;"
                     f"font-weight:700;color:#ccc'>{emoji} {dom.upper()}</div>")
        sub = {k: v for k, v in flat.items() if k.startswith(dom + ".")}
        for key, val in sub.items():
            name = key.split(".")[1].replace("_", " ")
            st   = status.get(key, 'unchanged')

            # If we have a 'before' snapshot and val improved, override status
            if before and key in before and st == 'unchanged':
                if val - before[key] > 1.0:
                    st = 'improved'

            color = CASCADE_COLORS[st]
            tag   = CASCADE_EMOJI[st]
            pct   = min(val, 100)

            delta_str = ""
            if before is not None and key in before:
                delta = val - before[key]
                if abs(delta) > 1.0:
                    arrow = "↑" if delta > 0 else "↓"
                    dc = "#22c55e" if delta > 0 else "#ef4444"
                    delta_str = (
                        f"<span style='font-size:10px;color:{dc};"
                        f"margin-left:4px;font-weight:700'>"
                        f"{arrow} ({delta:+.1f})</span>"
                    )

            rows.append(
                f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
                f"  <span style='font-size:10px'>{tag}</span>"
                f"  <span style='width:130px;font-size:11px;color:#bbb'>{name}</span>"
                f"  <div style='flex:1;background:#333;border-radius:4px;height:10px'>"
                f"    <div style='width:{pct}%;background:{color};border-radius:4px;"
                f"height:10px;transition:width 0.4s ease'></div>"
                f"  </div>"
                f"  <span style='width:38px;font-size:11px;color:#ccc;"
                f"text-align:right'>{val:.1f}</span>"
                f"  {delta_str}"
                f"</div>"
            )
    return "<div style='font-family:monospace;padding:8px'>" + "\n".join(rows) + "</div>"


NARRATIVE = [
    "Your life graph — stable state",
    "💥 Crisis hits: {title}",
    "🌊 Stress cascades to sleep and free time…",
    "⚡ Relationships and motivation begin degrading…",
    "🤖 Agent intervenes: {action_desc}",
]


# ─── Tab 1 — Live Demo (animated) ────────────────────────────────────────────
def run_demo(person_label: str, conflict_label: str):
    """Generator that yields (before_html, after_html, decision_html) at each animation frame."""
    import time as _t

    conflict = CONFLICT_CHOICES[conflict_label]
    person   = PERSONS[person_label]

    # Build cascade frames from a clean LifeMetrics
    base_metrics = LifeMetrics()
    frames = animate_cascade(conflict.primary_disruption, base_metrics)

    # Build predictor HTML
    summary = DEMO_PREDICTOR.get_prediction_summary()
    rscore = DEMO_PREDICTOR.get_risk_score()
    rcolor = "#4ade80" if rscore < 0.3 else ("#facc15" if rscore <= 0.6 else "#f87171")
    pct = min(100, int(rscore * 100))
    pred_html = f"""
    <div style='background:#1e1e2f;border:1px solid #333;border-left:4px solid {rcolor};border-radius:6px;padding:12px;margin-bottom:16px;font-family:sans-serif'>
        <div style='font-size:14px;font-weight:700;color:#ccc;margin-bottom:8px'>⚠️ TRAJECTORY ANALYSIS — Next 7 Days</div>
        <div style='margin-bottom:10px;font-size:13px;color:#ddd'>{summary}</div>
        <div style='display:flex;align-items:center;gap:10px'>
            <span style='font-size:12px;color:#aaa'>Risk Score:</span>
            <div style='flex:1;background:#333;border-radius:4px;height:12px'>
                <div style='width:{pct}%;background:{rcolor};border-radius:4px;height:12px'></div>
            </div>
            <span style='font-size:12px;color:{rcolor};font-weight:700'>{rscore:.2f}</span>
        </div>
    </div>
    """

    # ── Frame 0 — stable state ────────────────────────────────────────────
    f0 = frames[0]
    narr = f"<div style='padding:8px;color:#9ca3af;font-style:italic'>{NARRATIVE[0]}</div>"
    yield (
        pred_html,
        cascade_metrics_html(f0['flat'], f0['status'], "BEFORE"),
        narr,
        "",
    )
    _t.sleep(0.5)

    # ── Frame 1 — primary hit ─────────────────────────────────────────────
    f1 = frames[1]
    narr = (f"<div style='padding:8px;color:#ef4444;font-weight:700'>"
            f"{NARRATIVE[1].format(title=conflict.title)}</div>")
    yield (
        pred_html,
        cascade_metrics_html(f1['flat'], f1['status'], "DISRUPTION", before=f0['flat']),
        narr,
        "",
    )
    _t.sleep(0.5)

    # ── Frame 2 — first-order cascade ─────────────────────────────────────
    f2 = frames[2]
    narr = (f"<div style='padding:8px;color:#f97316;font-weight:700'>"
            f"{NARRATIVE[2]}</div>")
    yield (
        pred_html,
        cascade_metrics_html(f2['flat'], f2['status'], "CASCADE — 1st ORDER", before=f0['flat']),
        narr,
        "",
    )
    _t.sleep(0.5)

    # ── Frame 3 — second-order cascade ────────────────────────────────────
    f3 = frames[3]
    narr = (f"<div style='padding:8px;color:#eab308;font-weight:700'>"
            f"{NARRATIVE[3]}</div>")
    yield (
        pred_html,
        cascade_metrics_html(f3['flat'], f3['status'], "CASCADE — 2nd ORDER", before=f0['flat']),
        narr,
        "",
    )
    _t.sleep(0.5)

    # ── Frame 4 — agent intervention (final) ──────────────────────────────
    env            = _init_env(conflict)
    before_metrics = copy.deepcopy(env.state)
    before_budget  = copy.deepcopy(env.budget)

    action = AGENT.get_action(before_metrics, before_budget, conflict, person)

    # Normalise metric keys
    fixed_changes = {}
    for path, delta in action.primary.metric_changes.items():
        if "." not in str(path):
            path = f"{action.primary.target_domain}.{path}"
        try:
            fixed_changes[path] = float(delta)
        except (ValueError, TypeError):
            pass
    action.primary.metric_changes = fixed_changes

    is_valid, _ = validate_action(action, before_budget)
    if not is_valid:
        action.primary.metric_changes = {"mental_wellbeing.stress_level": -5.0}
        action.primary.resource_cost  = {}

    current_stress = before_metrics.mental_wellbeing.stress_level
    uptake = person.respond_to_action(
        action.primary.action_type, 
        action.primary.resource_cost, 
        current_stress
    )

    scaled_changes = {}
    for path, delta in action.primary.metric_changes.items():
        scaled_changes[path] = float(delta) * uptake

    env_action = {
        "metric_changes": scaled_changes,
        "resource_cost": action.primary.resource_cost,
        "actions_taken": 1
    }

    obs, reward, terminated, truncated, env_info = env.step(env_action)
    updated_metrics = env.state

    after_flat = updated_metrics.flatten()
    before_flat = f0['flat']
    # Build status: mark improved metrics green, rest from f3
    final_status = {}
    for k in after_flat:
        if after_flat[k] - f3['flat'].get(k, after_flat[k]) > 1.0:
            final_status[k] = 'improved'
        else:
            final_status[k] = f3['status'].get(k, 'unchanged')

    after_html = cascade_metrics_html(after_flat, final_status, "AFTER AGENT ACTION",
                                       before=before_flat)

    comm_block = ""
    if action.communication:
        comm_block = (
            f"<div style='margin-top:8px;padding:8px;background:#1e3a5f;"
            f"border-radius:6px;font-size:12px'>"
            f"💬 <b>Message to {action.communication.recipient}</b> "
            f"({action.communication.tone}): "
            f"<em>{action.communication.content}</em></div>"
        )

    cost = action.primary.resource_cost
    cost_str     = (f"⏱ {cost.get('time',0):.1f}h · "
                    f"💵 ${cost.get('money',0):.0f} · "
                    f"⚡ {cost.get('energy',0):.0f}")
    reward_color = "#4ade80" if reward > 0.4 else ("#facc15" if reward > 0 else "#f87171")

    narr = (f"<div style='padding:8px;color:#22c55e;font-weight:700'>"
            f"{NARRATIVE[4].format(action_desc=action.primary.description)}</div>")

    legend = (
        "<div style='margin-top:6px;padding:6px;font-size:11px;color:#aaa;"
        "border-top:1px solid #333;display:flex;gap:12px;flex-wrap:wrap'>"
        "🔴 Primary hit · 🟠 1st-order cascade · 🟡 2nd-order cascade · "
        "🟢 Agent improved · ⚪ Unchanged</div>"
    )

    decision_html = f"""
<div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;padding:16px;font-family:sans-serif'>
  <div style='font-size:18px;font-weight:700;margin-bottom:6px'>
    {action.primary.action_type.upper()} → {action.primary.target_domain}
  </div>
  <div style='color:#ccc;margin-bottom:8px'>{action.primary.description}</div>
  {comm_block}
  <div style='margin-top:10px;font-size:12px;color:#aaa;border-top:1px solid #333;padding-top:8px'>
    <b>Reasoning:</b> {action.reasoning}
  </div>
  <div style='margin-top:8px;display:flex;gap:16px;font-size:13px'>
    <span>{cost_str}</span>
    <span>🎯 Personality uptake: {uptake:.0%}</span>
    <span style='color:{reward_color};font-weight:700'>★ Reward: {reward:.3f}</span>
  </div>
  {legend}
</div>"""

    DEMO_PREDICTOR.add_snapshot(updated_metrics)
    summary = DEMO_PREDICTOR.get_prediction_summary()
    rscore = DEMO_PREDICTOR.get_risk_score()
    rcolor = "#4ade80" if rscore < 0.3 else ("#facc15" if rscore <= 0.6 else "#f87171")
    pct = min(100, int(rscore * 100))
    after_pred_html = f"""
    <div style='background:#1e1e2f;border:1px solid #333;border-left:4px solid {rcolor};border-radius:6px;padding:12px;margin-bottom:16px;font-family:sans-serif'>
        <div style='font-size:14px;font-weight:700;color:#ccc;margin-bottom:8px'>⚠️ TRAJECTORY ANALYSIS — Next 7 Days</div>
        <div style='margin-bottom:10px;font-size:13px;color:#ddd'>{summary}</div>
        <div style='display:flex;align-items:center;gap:10px'>
            <span style='font-size:12px;color:#aaa'>Risk Score:</span>
            <div style='flex:1;background:#333;border-radius:4px;height:12px'>
                <div style='width:{pct}%;background:{rcolor};border-radius:4px;height:12px'></div>
            </div>
            <span style='font-size:12px;color:{rcolor};font-weight:700'>{rscore:.2f}</span>
        </div>
    </div>
    """

    yield (after_pred_html, after_html, narr, decision_html)


# ─── Tab 2 — Try Your Situation (intake-powered) ─────────────────────────────
def run_custom(situation: str, work_stress: int, money_stress: int,
               relationship_q: int, energy: int, time_pressure: int):
    """Uses LifeIntake to extract structured conflict + personality from NL + sliders."""
    metrics, budget, conflict, personality = INTAKE.full_intake(
        situation, work_stress, money_stress, relationship_q, energy, time_pressure
    )

    person = SimPerson(
        name=personality.get("name", "You"),
        openness=personality.get("openness", 0.5),
        conscientiousness=personality.get("conscientiousness", 0.5),
        extraversion=personality.get("extraversion", 0.5),
        agreeableness=personality.get("agreeableness", 0.5),
        neuroticism=personality.get("neuroticism", 0.5),
    )

    life_html = (
        "<div style='font-family:sans-serif;font-size:13px;color:#a78bfa;"
        "padding:8px 8px 4px;font-style:italic'>"
        "Based on what you described, here is how your life looks right now:"
        "</div>"
        + metrics_html(metrics.flatten(), "YOUR LIFE RIGHT NOW")
    )

    action = AGENT.get_action(metrics, budget, conflict, person)

    fixed = {}
    for path, delta in action.primary.metric_changes.items():
        if "." not in str(path):
            path = f"{action.primary.target_domain}.{path}"
        try:
            fixed[path] = float(delta)
        except (ValueError, TypeError):
            pass
    action.primary.metric_changes = fixed

    is_valid, _ = validate_action(action, budget)
    if not is_valid:
        action.primary.metric_changes = {"mental_wellbeing.stress_level": -5.0}
        action.primary.resource_cost  = {}

    env = LifeStackEnv()
    env.state = metrics
    env.budget = budget
    
    current_stress = metrics.mental_wellbeing.stress_level
    uptake = person.respond_to_action(
        action.primary.action_type, 
        action.primary.resource_cost, 
        current_stress
    )

    scaled_changes = {}
    for path, delta in action.primary.metric_changes.items():
        scaled_changes[path] = float(delta) * uptake

    env_action = {
        "metric_changes": scaled_changes,
        "resource_cost": action.primary.resource_cost,
        "actions_taken": 1
    }

    obs, reward, terminated, truncated, env_info = env.step(env_action)
    updated_metrics = env.state

    after_html = metrics_html(updated_metrics.flatten(), "AFTER ACTION", before=metrics.flatten())
    reward_color = "#4ade80" if reward > 0.4 else ("#facc15" if reward > 0 else "#f87171")

    trait_bar = lambda v: "█" * int(v * 10) + "░" * (10 - int(v * 10))
    personality_html = f"""
<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:8px;padding:12px;
            margin-bottom:12px;font-family:monospace;font-size:11px;color:#ccc'>
  <div style='font-size:13px;font-weight:700;color:#a78bfa;margin-bottom:8px'>🧠 Inferred Personality: {person.name}</div>
  <div>Openness&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {trait_bar(personality.get('openness',0.5))} {personality.get('openness',0.5):.2f}</div>
  <div>Conscientiousness {trait_bar(personality.get('conscientiousness',0.5))} {personality.get('conscientiousness',0.5):.2f}</div>
  <div>Extraversion&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {trait_bar(personality.get('extraversion',0.5))} {personality.get('extraversion',0.5):.2f}</div>
  <div>Agreeableness&nbsp;&nbsp;&nbsp;&nbsp; {trait_bar(personality.get('agreeableness',0.5))} {personality.get('agreeableness',0.5):.2f}</div>
  <div>Neuroticism&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {trait_bar(personality.get('neuroticism',0.5))} {personality.get('neuroticism',0.5):.2f}</div>
</div>"""

    steps = [f"<b>Step 1:</b> {action.primary.description}"]
    if action.communication:
        steps.append(
            f"<b>Message to {action.communication.recipient}</b> "
            f"({action.communication.tone}): <em>{action.communication.content}</em>"
        )
    cost     = action.primary.resource_cost
    cost_str = f"⏱ {cost.get('time', 0):.1f}h · 💵 ${cost.get('money', 0):.0f} · ⚡ {cost.get('energy', 0):.0f}"

    plan_html = f"""
{personality_html}
<div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;padding:16px;font-family:sans-serif;color:#eee'>
  <div style='font-size:13px;font-weight:700;color:#60a5fa;margin-bottom:4px'>
    📋 {conflict.title} (Difficulty {conflict.difficulty}/5)
  </div>
  <div style='font-size:12px;color:#aaa;margin-bottom:10px'>{conflict.story}</div>
  <div style='font-size:16px;font-weight:700;margin-bottom:10px'>🎯 Resolution Plan for {person.name}</div>
  <div style='margin-bottom:8px'>{"<br>".join(steps)}</div>
  <div style='margin:10px 0;padding:8px;background:#0d1b2a;border-radius:6px;font-size:12px;color:#aaa'>
    <b>Why:</b> {action.reasoning}
  </div>
  <div style='display:flex;gap:20px;font-size:13px;border-top:1px solid #333;padding-top:8px'>
    <span>{cost_str}</span>
    <span>🎯 Personality fit: {uptake:.0%}</span>
    <span style='color:{reward_color};font-weight:700'>★ Confidence: {min(reward/0.6, 1.0):.0%}</span>
  </div>
</div>"""

    return life_html, after_html, plan_html


# ─── Tab 3 — Training Results ─────────────────────────────────────────────────
def load_training_tab():
    html_parts = []

    try:
        stats = MEMORY.get_stats()
        html_parts.append(f"""
<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px'>
  <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:140px;text-align:center'>
    <div style='font-size:28px;font-weight:700;color:#4ade80'>{stats['total_memories']}</div>
    <div style='color:#aaa;font-size:12px'>Decisions Stored</div>
  </div>
  <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:140px;text-align:center'>
    <div style='font-size:28px;font-weight:700;color:#60a5fa'>{stats['average_reward']:.3f}</div>
    <div style='color:#aaa;font-size:12px'>Avg Memory Reward</div>
  </div>
  <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:200px'>
    <div style='font-size:12px;color:#aaa;margin-bottom:6px'>By Action Type</div>
    {''.join(f"<div style='font-size:12px'><b>{k}</b>: {v}</div>" for k,v in stats['by_action_type'].items())}
  </div>
</div>""")
    except Exception as e:
        html_parts.append(f"<p style='color:#f87171'>Memory error: {e}</p>")

    log_path = os.path.join(os.path.dirname(__file__), "training_log.json")
    if os.path.exists(log_path):
        try:
            data    = json.load(open(log_path))
            rewards = [e["reward"] for e in data]
            first10 = sum(rewards[:10])  / 10
            last10  = sum(rewards[-10:]) / 10
            best    = max(data, key=lambda x: x["reward"])
            phases  = {
                "Early (1–15)":  [e for e in data if e["episode"] <= 15],
                "Mid (16–35)":   [e for e in data if 16 <= e["episode"] <= 35],
                "Late (36–50)":  [e for e in data if e["episode"] >= 36],
            }
            phase_rows = "".join(
                f"<tr><td style='padding:4px 10px'>{name}</td><td style='padding:4px 10px;text-align:center'>{len(eps)}</td>"
                f"<td style='padding:4px 10px;text-align:center;color:#4ade80'>{sum(e['reward'] for e in eps)/len(eps):.3f}</td></tr>"
                for name, eps in phases.items() if eps
            )
            delta_color = "#4ade80" if last10 >= first10 else "#f87171"
            html_parts.append(f"""
<div style='margin-bottom:16px'>
  <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px'>
    <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:140px;text-align:center'>
      <div style='font-size:28px;font-weight:700;color:#a78bfa'>{len(data)}</div>
      <div style='color:#aaa;font-size:12px'>Total Episodes</div>
    </div>
    <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:140px;text-align:center'>
      <div style='font-size:28px;font-weight:700;color:#4ade80'>{sum(rewards)/len(rewards):.3f}</div>
      <div style='color:#aaa;font-size:12px'>Overall Avg Reward</div>
    </div>
    <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:140px;text-align:center'>
      <div style='font-size:28px;font-weight:700;color:#fbbf24'>{best["reward"]:.3f}</div>
      <div style='color:#aaa;font-size:12px'>Best Episode (#{best["episode"]})</div>
    </div>
    <div style='background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:12px;min-width:160px;text-align:center'>
      <div style='font-size:22px;font-weight:700;color:{delta_color}'>
        {"+" if last10>=first10 else ""}{(last10-first10):.3f}
      </div>
      <div style='color:#aaa;font-size:12px'>Ep 1–10 → 41–50 Δ</div>
    </div>
  </div>
  <table style='border-collapse:collapse;width:100%;max-width:400px;font-size:13px;color:#eee'>
    <tr style='color:#aaa;border-bottom:1px solid #333'>
      <th style='padding:4px 10px;text-align:left'>Phase</th>
      <th style='padding:4px 10px'>Episodes</th>
      <th style='padding:4px 10px'>Avg Reward</th>
    </tr>
    {phase_rows}
  </table>
</div>""")
        except Exception as e:
            html_parts.append(f"<p style='color:#f87171'>Log parse error: {e}</p>")
    else:
        html_parts.append("<p style='color:#aaa'>training_log.json not found — run train.py first.</p>")

    return "<div style='font-family:sans-serif;color:#eee'>" + "\n".join(html_parts) + "</div>"


# ─── Gradio App ───────────────────────────────────────────────────────────────
with gr.Blocks(
    title="LifeStack — AI Life Coach",
) as app:

    gr.HTML("""
    <div style='text-align:center;padding:24px 0 8px;font-family:sans-serif'>
      <div style='font-size:36px;font-weight:900;letter-spacing:-1px;
                  background:linear-gradient(90deg,#a78bfa,#60a5fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        LifeStack
      </div>
      <div style='color:#888;font-size:14px;margin-top:4px'>
        AI that handles life's worst Fridays
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Live Demo ─────────────────────────────────────────────────
        with gr.Tab("🎯 Live Demo"):
            gr.HTML(f"""
            <div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;padding:16px;
                        margin-bottom:16px;font-family:sans-serif'>
              <div style='font-size:16px;font-weight:700;color:#a78bfa;margin-bottom:6px'>
                🚨 Friday 6PM
              </div>
              <div style='color:#ddd;font-size:14px'>{DEMO_CONFLICT.story}</div>
              <div style='margin-top:8px;font-size:12px;color:#888'>
                Difficulty: ⭐⭐⭐⭐⭐ &nbsp;|&nbsp;
                Domains hit: Career, Finances, Mental Health, Time
              </div>
            </div>
            """)

            prediction_ui = gr.HTML()
            
            with gr.Row():
                conflict_dd = gr.Dropdown(
                    choices=CONFLICT_CHOICES_LIST,
                    value=DEFAULT_CONFLICT,
                    label="📋 Conflict Scenario",
                )
                person_dd = gr.Dropdown(
                    choices=PERSON_CHOICES,
                    value=PERSON_CHOICES[0],
                    label="👤 Choose Your Person",
                )

            run_btn = gr.Button("▶  Run Agent", variant="primary", size="lg")

            cascade_narrative = gr.HTML(label="Cascade Narrative")

            with gr.Row():
                before_out = gr.HTML(label="Life State")
                after_out  = gr.HTML(label="Agent Decision")

            run_btn.click(
                fn=run_demo,
                inputs=[person_dd, conflict_dd],
                outputs=[prediction_ui, before_out, cascade_narrative, after_out],
            )

        # ── Tab 2: Try Your Situation ────────────────────────────────────────
        with gr.Tab("💭 Try Your Situation"):
            gr.Markdown(
                "Describe your situation in plain English. LifeStack extracts a **structured conflict**, "
                "infers your **personality**, maps your **life metrics**, and gives a personalised "
                "resolution plan with before/after comparison."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    situation_input = gr.Textbox(
                        label="What's stressing you out right now?",
                        placeholder="e.g. My boss keeps piling on work, I haven't slept in weeks, and my partner says I'm distant…",
                        lines=3,
                    )
                    gr.Markdown("**Rate your current state (0 = none / low · 10 = extreme / high):**")
                    work_sl   = gr.Slider(0, 10, value=7, step=1, label="💼 Work Stress")
                    money_sl  = gr.Slider(0, 10, value=5, step=1, label="💰 Money Stress")
                    rel_sl    = gr.Slider(0, 10, value=6, step=1, label="❤️ Relationship Quality")
                    energy_sl = gr.Slider(0, 10, value=4, step=1, label="⚡ Energy Level")
                    time_sl   = gr.Slider(0, 10, value=7, step=1, label="📅 Time Pressure")
                    submit_btn = gr.Button("✨ Analyse & Get My Plan", variant="primary", size="lg")

                with gr.Column(scale=1):
                    life_graph_out  = gr.HTML(label="Your Life Right Now")
                    after_graph_out = gr.HTML(label="After Action")
                    plan_out        = gr.HTML(label="Resolution Plan")

            submit_btn.click(
                fn=run_custom,
                inputs=[situation_input, work_sl, money_sl, rel_sl, energy_sl, time_sl],
                outputs=[life_graph_out, after_graph_out, plan_out],
            )

        # ── Tab 3: Training Results ──────────────────────────────────────────
        with gr.Tab("📊 Training Results"):
            training_html = gr.HTML(value=load_training_tab())

            plot_path = os.path.join(os.path.dirname(__file__), "reward_curve.png")
            if os.path.exists(plot_path):
                gr.Image(value=plot_path, label="Learning Curve — 100 Episode Training Run")

    gr.HTML("""
    <div style='text-align:center;padding:16px;color:#444;font-size:11px;border-top:1px solid #222;margin-top:16px'>
      LifeStack · Built for hackathon demo · Powered by Groq + ChromaDB + Sentence Transformers
    </div>
    """)


if __name__ == "__main__":
    app.launch(
        share=False,
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
        css="""
        body { background:#0d0d1a; }
        .gradio-container { max-width: 1100px; margin: auto; }
        h1 { text-align:center; }
        .tab-nav button { font-size:14px; font-weight:600; }
        """
    )
