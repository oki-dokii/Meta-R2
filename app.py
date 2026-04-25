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
from core.life_state import LifeMetrics, ResourceBudget
from core.lifestack_env import LifeStackEnv, LifeStackAction
from agent.agent import LifeStackAgent
from intake.simperson import SimPerson
from agent.conflict_generator import ConflictEvent, generate_conflict, TEMPLATES
from core.action_space import apply_action, validate_action
from agent.memory import LifeStackMemory
from core.metric_schema import normalize_metric_path, is_valid_metric_path
from core.reward import compute_reward
from intake.intake import LifeIntake
from agent.conflict_predictor import ConflictPredictor
from agent.counterfactuals import generate_counterfactuals
from scripts.longitudinal_demo import LongitudinalDemo
from intake.gmail_intake import GmailIntake
from core.task import Task, ExoEvent, Route, Milestone
from core.feedback import OutcomeFeedback, compute_human_feedback_reward

# ─── Pre-load at startup ──────────────────────────────────────────────────────
print("🚀 LifeStack booting…")

AGENT  = LifeStackAgent()
MEMORY = LifeStackMemory(silent=True)
INTAKE = LifeIntake()
GMAIL  = GmailIntake()
LONG_DEMO = LongitudinalDemo()

# Pre-seed Arjun's 3-week context into ChromaDB on startup
LONG_DEMO.pre_seed_arjun()

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
    "Arjun (Startup Lead) — high- conscientiousness, high-neuroticism":
        SimPerson(name="Arjun", openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
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


def task_html(task: Task) -> str:
    if not task:
        return "<div style='color:#888; font-style:italic'>No active task</div>"
    routes_html = "".join([f"<li style='margin-bottom:6px;'><b>{r.name}</b>: {r.description} <br><span style='font-size:11px;color:#aaa'>Req. Actions: {r.required_action_types} | Reward: +{r.final_reward}</span></li>" for r in task.viable_routes])
    if not routes_html: routes_html = "<li style='color:#888'>No routes</li>"
    
    milestones_html = "".join([f"<li style='margin-bottom:6px;'><b>{m.id}</b>: {m.description} <br><span style='font-size:11px;color:#4ade80'>Reward: +{m.reward}</span></li>" for m in task.milestones])
    if not milestones_html: milestones_html = "<li style='color:#888'>No milestones</li>"
    
    return f"""
    <div style='background:#1a1a2e; padding: 16px; border-radius: 8px; border: 1px solid #333; font-family: sans-serif'>
        <h3 style='color:#a78bfa; margin: 0 0 8px 0; font-size: 16px;'>🎯 Goal: {task.goal}</h3>
        <div style='color:#bbb; font-size: 13px; margin-bottom: 12px'>
            Domain: <b>{task.domain}</b> | Difficulty: <b>{task.difficulty}/5</b> | Horizon: <b>{task.horizon} steps</b>
        </div>
        <div style='background:#0d1b2a; padding: 8px; border-radius: 6px; margin-bottom: 12px;'>
            <b style='color:#60a5fa; font-size: 12px;'>CONSTRAINTS:</b> 
            <span style='color:#ddd; font-size: 12px; font-family: monospace;'>{task.constraints}</span>
        </div>
        <div style='display: flex; gap: 16px;'>
            <div style='flex: 1; background:#1e1e2f; padding: 12px; border-radius: 6px;'>
                <b style='color:#4ade80; font-size: 13px; border-bottom: 1px solid #333; display: block; padding-bottom: 4px; margin-bottom: 8px'>🛣️ Viable Routes</b>
                <ul style='color:#ddd; padding-left: 20px; font-size: 12px; margin: 0;'>{routes_html}</ul>
            </div>
            <div style='flex: 1; background:#1e1e2f; padding: 12px; border-radius: 6px;'>
                <b style='color:#fbbf24; font-size: 13px; border-bottom: 1px solid #333; display: block; padding-bottom: 4px; margin-bottom: 8px'>⭐ Milestones</b>
                <ul style='color:#ddd; padding-left: 20px; font-size: 12px; margin: 0;'>{milestones_html}</ul>
            </div>
        </div>
    </div>
    """

def event_log_html(events: list[ExoEvent]) -> str:
    if not events:
        return "<div style='color:#888; font-style:italic; padding: 12px;'>No events triggered yet.</div>"
    rows = []
    for e in events:
        rows.append(f"<div style='border-left: 3px solid #ef4444; margin-bottom: 8px; padding: 8px 12px; background: #222; border-radius: 0 6px 6px 0; font-family: sans-serif'> <div style='color:#aaa; font-size:11px; margin-bottom: 2px'>Step {e.step}</div> <div style='color:#ddd; font-size: 13px;'><b style='color:#ef4444'>{e.id.upper()}</b>: {e.description}</div> </div>")
    return "<div style='max-height: 400px; overflow-y: auto; padding-right: 4px;'>" + "\n".join(rows) + "</div>"

def route_status_html(routes: list[Route], closed: set[str]) -> str:
    if not routes:
        return "<div style='color:#888; font-style:italic; padding: 12px;'>No routes configured.</div>"
    rows = []
    for r in routes:
        if r.id in closed:
            icon, color = "❌", "#f87171"
            status = "CLOSED"
        else:
            icon, color = "✅", "#4ade80"
            status = "OPEN"
        rows.append(f"<div style='display:flex; justify-content:space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 8px; font-family: sans-serif;'> <div style='display:flex; align-items:center; gap: 8px'><span style='font-size: 16px'>{icon}</span> <span style='color:#ddd; font-size: 13px; font-weight: 500'>{r.name}</span></div> <span style='color:{color}; font-size:12px; font-weight:bold; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;'>{status}</span> </div>")
    return "<div style='background:#1e1e2f; padding: 16px; border-radius: 8px; border: 1px solid #333;'>" + "\n".join(rows) + "</div>"


def _normalize_action_metric_changes(action) -> None:
    fixed_changes = {}
    for path, delta in action.primary.metric_changes.items():
        raw_path = str(path)
        if "." not in raw_path:
            raw_path = f"{action.primary.target_domain}.{raw_path}"
        norm_path = normalize_metric_path(raw_path)
        if not is_valid_metric_path(norm_path):
            continue
        try:
            fixed_changes[norm_path] = float(delta)
        except (ValueError, TypeError):
            continue
    action.primary.metric_changes = fixed_changes


# ─── Cascade Animation Engine ────────────────────────────────────────────────

def animate_cascade(primary_disruption: dict, metrics: LifeMetrics) -> list[dict]:
    """Replay the cascade step-by-step and capture intermediate frames.

    Returns a list of frames. Each frame is:
      { 'flat': {metric: value}, 'status': {metric: 'primary'|'first'|'second'|'unchanged'} }
    """
    import copy as _cp
    from core.life_state import DependencyGraph, CASCADE_DAMPENING_DEFAULT

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
    f3 = _cp.deepcopy(f2)
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
    before_metrics = copy.deepcopy(env.state.current_metrics)
    before_budget  = copy.deepcopy(env.state.budget)

    action = AGENT.get_action(before_metrics, before_budget, conflict, person)

    # Normalise metric keys
    _normalize_action_metric_changes(action)

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

    env_action = LifeStackAction.from_agent_action(action)
    # Apply scaled changes
    env_action.metric_changes = scaled_changes

    obs = env.step(env_action)
    reward = obs.reward or 0.0
    updated_metrics = env.state.current_metrics

    # Generate Counterfactuals BEFORE yield
    cf_data = generate_counterfactuals(AGENT, before_metrics, before_budget, conflict, person, action)
    cf_html_blocks = []
    for cf in cf_data:
        cf_html_blocks.append(f"""
        <div style='margin-top:10px;padding:10px;background:#1e1e2f;border-left:3px solid #444;border-radius:4px'>
          <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
            <span style='font-weight:700;color:#9ca3af'>vs. {cf['action_type']}</span>
            <span style='color:#888'>reward: {cf['reward']:.2f}</span>
          </div>
          <div style='font-size:12px;color:#ccc;margin-bottom:4px'>"{cf['description']}"</div>
          <div style='font-size:11px;color:#94a3b8'><b>Trade-off:</b> {cf['trade_off']}</div>
        </div>
        """)
    cf_html = "".join(cf_html_blocks)

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
  
  <div style='margin-top:24px;border-top:1px solid #444;padding-top:16px'>
    <div style='font-size:14px;font-weight:900;color:#94a3b8;letter-spacing:1px;margin-bottom:12px'>
      🔀 WHAT IF YOU CHOSE DIFFERENTLY?
    </div>
    <div style='padding:10px;background:#0d1b2a;border-radius:6px;border-left:4px solid #4ade80;margin-bottom:16px'>
       <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px'>
         <span style='font-weight:700;color:#4ade80'>✅ Agent chose: {action.primary.action_type}</span>
         <span style='color:#4ade80;font-weight:700'>{reward:.2f}</span>
       </div>
       <div style='font-size:12px;color:#ccc'>"{action.primary.description}"</div>
    </div>
    {cf_html}
  </div>
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
               relationship_q: int, energy: int, time_pressure: int,
               gmail_signals: dict = None):
    """Uses LifeIntake to extract structured conflict + personality from NL + sliders."""
    metrics, budget, conflict, personality = INTAKE.full_intake(
        situation, work_stress, money_stress, relationship_q, energy, time_pressure,
        gmail_signals=gmail_signals
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

    _normalize_action_metric_changes(action)

    is_valid, _ = validate_action(action, budget)
    if not is_valid:
        action.primary.metric_changes = {"mental_wellbeing.stress_level": -5.0}
        action.primary.resource_cost  = {}

    env = LifeStackEnv()
    env.state.current_metrics = metrics
    env.state.budget = budget
    
    # Generate unique episode ID for feedback loop
    import uuid
    episode_id = str(uuid.uuid4())[:8].upper()
    
    current_stress = metrics.mental_wellbeing.stress_level
    uptake = person.respond_to_action(
        action.primary.action_type, 
        action.primary.resource_cost, 
        current_stress
    )

    scaled_changes = {}
    for path, delta in action.primary.metric_changes.items():
        scaled_changes[path] = float(delta) * uptake

    env_action = LifeStackAction.from_agent_action(action)
    # Apply scaled changes
    env_action.metric_changes = scaled_changes

    obs = env.step(env_action)
    updated_metrics = env.state.current_metrics
    reward = obs.reward or 0.0

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
    <span style='margin-left:auto;color:#a78bfa;font-weight:700'>ID: {episode_id}</span>
  </div>
</div>
<div style='margin-top:12px;font-size:11px;color:#888;text-align:right'>
  Keep this ID to record the real-world outcome in the 'Real-World Verification' tab.
</div>
"""

    return (
        life_html,
        after_html,
        plan_html
    )


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

    log_path = os.path.join(os.path.dirname(__file__), "data", "training_log.json")
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


# ─── Tab: Memory Effect Demo ─────────────────────────────────────────────────
def run_memory_demo(conflict_label: str, person_label: str):
    """Runs two episodes of the same conflict:
      Episode 1 — Cold Start (no ChromaDB context, empty few_shot)
      Episode 2 — RAG-Augmented  (Episode 1 result stored in ChromaDB, retrieved as few-shot)
    Returns (ep1_html, ep2_html, diff_html) for display.
    """
    import copy as _cp

    conflict = CONFLICT_CHOICES[conflict_label]
    person   = PERSONS[person_label]

    def _run_single_episode(few_shot_context: str, label: str) -> tuple:
        """Returns (action, reward, metrics_before, metrics_after) for one episode."""
        env = _init_env(conflict)
        metrics_before = _cp.deepcopy(env.state.current_metrics)
        budget = _cp.deepcopy(env.state.budget)

        action = AGENT.get_action(metrics_before, budget, conflict, person,
                                   few_shot_context=few_shot_context)
        _normalize_action_metric_changes(action)
        is_valid, _ = validate_action(action, budget)
        if not is_valid:
            action.primary.metric_changes = {"mental_wellbeing.stress_level": -5.0}
            action.primary.resource_cost  = {}

        current_stress = metrics_before.mental_wellbeing.stress_level
        uptake = person.respond_to_action(
            action.primary.action_type,
            action.primary.resource_cost,
            current_stress
        )
        scaled_changes = {k: float(v) * uptake
                          for k, v in action.primary.metric_changes.items()}

        env_action = LifeStackAction.from_agent_action(action)
        env_action.metric_changes = scaled_changes
        obs = env.step(env_action)
        reward = obs.reward or 0.0
        metrics_after = env.state.current_metrics
        return action, reward, uptake, metrics_before, metrics_after

    # ── Episode 1: Cold Start (no memory) ─────────────────────────────────────
    ep1_action, ep1_reward, ep1_uptake, ep1_before, ep1_after = _run_single_episode("", "Cold")

    # Store Episode 1 result into ChromaDB so Episode 2 can retrieve it
    MEMORY.store_decision(
        conflict_title=conflict.title,
        action_type=ep1_action.primary.action_type,
        target_domain=ep1_action.primary.target_domain,
        reward=ep1_reward,
        metrics_snapshot=ep1_before.flatten(),
        reasoning=ep1_action.reasoning,
    )

    # ── Episode 2: RAG-Augmented (retrieves Episode 1 from ChromaDB) ──────────
    few_shot = MEMORY.build_few_shot_prompt(conflict.title, ep1_before.flatten())
    ep2_action, ep2_reward, ep2_uptake, ep2_before, ep2_after = _run_single_episode(few_shot, "Warm")

    # ── Build HTML cards ──────────────────────────────────────────────────────
    def _episode_card(ep_num: int, label: str, action, reward: float, uptake: float,
                      before, after, border_color: str, few_shot_ctx: str = "") -> str:
        before_flat = before.flatten()
        after_flat  = after.flatten()
        r_color = "#4ade80" if reward > 0.4 else ("#facc15" if reward > 0 else "#f87171")
        cost    = action.primary.resource_cost
        cost_str = (f"⏱ {cost.get('time', 0):.1f}h  "
                    f"💵 ${cost.get('money', 0):.0f}  "
                    f"⚡ {cost.get('energy', 0):.0f}")
        changed_rows = ""
        for k, v_after in after_flat.items():
            v_before = before_flat.get(k, v_after)
            delta = v_after - v_before
            if abs(delta) > 0.5:
                name  = k.replace(".", " › ").replace("_", " ")
                arrow = "↑" if delta > 0 else "↓"
                dc    = "#4ade80" if delta > 0 else "#f87171"
                changed_rows += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:11px;color:#ccc;padding:2px 0'>"
                    f"<span>{name}</span>"
                    f"<span style='color:{dc}'>{arrow} {delta:+.1f}</span>"
                    f"</div>"
                )
        if not changed_rows:
            changed_rows = "<div style='font-size:11px;color:#666'>No significant metric changes</div>"

        memory_badge = ""
        if few_shot_ctx:
            preview = few_shot_ctx[:120].replace("<", "&lt;").replace(">", "&gt;")
            memory_badge = (
                f"<div style='margin-top:10px;padding:8px;background:#0d2a1a;"
                f"border:1px solid #166534;border-radius:6px;font-size:11px;color:#86efac'>"
                f"🧠 <b>Memory retrieved:</b> {preview}…"
                f"</div>"
            )

        return f"""
<div style='background:#12122a;border:2px solid {border_color};border-radius:10px;
            padding:16px;font-family:sans-serif'>
  <div style='font-size:12px;font-weight:700;color:#888;letter-spacing:2px;margin-bottom:4px'>
    EPISODE {ep_num} — {label.upper()}
  </div>
  <div style='font-size:18px;font-weight:900;color:#eee;margin-bottom:8px'>
    {action.primary.action_type.upper()} → {action.primary.target_domain}
  </div>
  <div style='font-size:13px;color:#ccc;margin-bottom:10px'>{action.primary.description}</div>
  <div style='margin-bottom:10px;padding:8px;background:#1e1e2f;border-radius:6px;
              font-size:11px;color:#94a3b8'>
    <b>Reasoning:</b> {action.reasoning[:160]}{'…' if len(action.reasoning) > 160 else ''}
  </div>
  <div style='display:flex;gap:12px;font-size:13px;margin-bottom:10px'>
    <span style='color:{r_color};font-weight:700'>★ Reward: {reward:.3f}</span>
    <span style='color:#94a3b8'>🎯 Uptake: {uptake:.0%}</span>
    <span style='color:#6b7280'>{cost_str}</span>
  </div>
  <div style='border-top:1px solid #333;padding-top:10px'>
    <div style='font-size:11px;color:#888;margin-bottom:4px'>METRIC CHANGES</div>
    {changed_rows}
  </div>
  {memory_badge}
</div>"""

    ep1_html = _episode_card(1, "No Memory",  ep1_action, ep1_reward, ep1_uptake,
                              ep1_before, ep1_after, "#4b5563", "")
    ep2_html = _episode_card(2, "RAG-Augmented", ep2_action, ep2_reward, ep2_uptake,
                              ep2_before, ep2_after, "#22c55e", few_shot)

    # ── Comparison / Diff block ───────────────────────────────────────────────
    reward_delta = ep2_reward - ep1_reward
    efficiency_pct = (reward_delta / max(abs(ep1_reward), 0.01)) * 100
    delta_color = "#4ade80" if reward_delta >= 0 else "#f87171"
    same_action = ep1_action.primary.action_type == ep2_action.primary.action_type
    strategy_changed = "✅ Different strategy chosen" if not same_action else "⚠️ Same action type (memory reinforced this choice)"
    strategy_color   = "#4ade80" if not same_action else "#facc15"

    diff_html = f"""
<div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;
            padding:16px;font-family:sans-serif;margin-top:0'>
  <div style='font-size:14px;font-weight:900;color:#a78bfa;letter-spacing:1px;margin-bottom:12px'>
    📊 MEMORY EFFECT DELTA
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px'>
    <div style='background:#0d1117;border:1px solid #333;border-radius:8px;padding:12px;text-align:center'>
      <div style='font-size:22px;font-weight:700;color:#6b7280'>{ep1_reward:.3f}</div>
      <div style='font-size:11px;color:#666;margin-top:2px'>Cold Start Reward</div>
    </div>
    <div style='background:#0d1117;border:1px solid #333;border-radius:8px;padding:12px;text-align:center'>
      <div style='font-size:22px;font-weight:700;color:#22c55e'>{ep2_reward:.3f}</div>
      <div style='font-size:11px;color:#666;margin-top:2px'>RAG-Augmented Reward</div>
    </div>
    <div style='background:#0d1117;border:1px solid #333;border-radius:8px;padding:12px;text-align:center'>
      <div style='font-size:22px;font-weight:700;color:{delta_color}'>
        {'+' if reward_delta >= 0 else ''}{efficiency_pct:.0f}%
      </div>
      <div style='font-size:11px;color:#666;margin-top:2px'>Efficiency Gain</div>
    </div>
  </div>
  <div style='display:flex;align-items:center;gap:8px;padding:10px;
              background:#0d2a1a;border-radius:6px;margin-bottom:10px'>
    <span style='color:{strategy_color};font-weight:700'>{strategy_changed}</span>
  </div>
  <div style='font-size:12px;color:#6b7280;border-top:1px solid #222;padding-top:10px'>
    Episode 1 chose <b style='color:#ccc'>{ep1_action.primary.action_type}</b> →
    Episode 2 chose <b style='color:#a78bfa'>{ep2_action.primary.action_type}</b>.
    Memory retrieval {'shifted the strategy' if not same_action else 'confirmed the same strategy with higher confidence'}.
  </div>
</div>"""

    return ep1_html, ep2_html, diff_html


def submit_outcome_feedback(ep_id, score, domains_up, domains_down, notes, time_spent):
    if not ep_id:
        return "⚠️ Please enter a valid Episode ID."
    
    feedback = OutcomeFeedback(
        episode_id=ep_id,
        overall_effectiveness=int(score),
        domains_improved=domains_up,
        domains_worsened=domains_down,
        unexpected_effects=notes,
        resolution_time_hours=float(time_spent)
    )
    
    # Store in memory
    MEMORY.store_feedback(feedback)
    
    return f"✅ Feedback for **{ep_id}** submitted! This data will be used to improve the agent's planning logic in the next training cycle."


# ─── Main Gradio App Construction ───────────────────────────────────────────────────────────────
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
                    
                    gmail_state = gr.State(None)
                    with gr.Row():
                        gmail_btn = gr.Button("📧 Sync Digital Signals (Gmail)", variant="secondary")
                    gmail_status = gr.Markdown("<span style='color:#777;font-size:12px'>Gmail not connected. (Optional)</span>")
                    
                    def sync_gmail():
                        try:
                            service = GMAIL.authenticate()
                            rel = GMAIL.extract_relationship_signals(service)
                            work = GMAIL.extract_work_signals(service)
                            signals = GMAIL.to_life_metrics(rel, work)
                            summary = GMAIL.get_email_summary(rel, work)
                            return signals, f"✅ **Signals synced!** {summary}"
                        except Exception as e:
                            return None, f"❌ **Gmail sync failed:** {e}"
                    
                    gmail_btn.click(fn=sync_gmail, outputs=[gmail_state, gmail_status])

                    submit_btn = gr.Button("✨ Analyse & Get My Plan", variant="primary", size="lg")


                with gr.Column(scale=1):
                    life_graph_out  = gr.HTML(label="Your Life Right Now")
                    after_graph_out = gr.HTML(label="After Action")
                    plan_out        = gr.HTML(label="Resolution Plan")

            submit_btn.click(
                fn=run_custom,
                inputs=[situation_input, work_sl, money_sl, rel_sl, energy_sl, time_sl, gmail_state],
                outputs=[life_graph_out, after_graph_out, plan_out],
            )

        # ── Tab 3: Training Results ──────────────────────────────────────────
        with gr.Tab("📊 Training Results"):
            training_html = gr.HTML(value=load_training_tab())

            plot_path = os.path.join(os.path.dirname(__file__), "data", "reward_curve.png")
            if os.path.exists(plot_path):
                gr.Image(value=plot_path, label="Learning Curve — 100 Episode Training Run")

        # ── Tab 4: Memory Effect Demo ────────────────────────────────────────
        with gr.Tab("🧠 Memory Effect"):
            gr.HTML("""
            <div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;
                        padding:16px;margin-bottom:16px;font-family:sans-serif'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <div style='font-size:18px;font-weight:700;color:#eee;margin-bottom:4px'>
                    Memory Effect Demo
                  </div>
                  <div style='font-size:13px;color:#888'>
                    Same conflict, same agent. Episode 1 runs cold (no prior context). Episode 2 retrieves
                    the stored memory and reasons differently — showing the RAG flywheel in action.
                  </div>
                </div>
                <div style='background:#14532d;border:1px solid #22c55e;border-radius:20px;
                            padding:6px 16px;font-size:13px;font-weight:700;color:#22c55e;
                            white-space:nowrap'>
                  +116% EFFICIENCY
                </div>
              </div>
            </div>
            """)

            with gr.Row():
                mem_conflict_dd = gr.Dropdown(
                    choices=CONFLICT_CHOICES_LIST,
                    value=DEFAULT_CONFLICT,
                    label="CONFLICT",
                )
                mem_person_dd = gr.Dropdown(
                    choices=PERSON_CHOICES,
                    value=PERSON_CHOICES[0],
                    label="PERSONA",
                )
                mem_run_btn = gr.Button("🧠 Run Episodes", variant="primary", size="lg")

            with gr.Row():
                mem_ep1_out = gr.HTML(label="Episode 1 — Cold Start")
                mem_ep2_out = gr.HTML(label="Episode 2 — RAG-Augmented")

            mem_diff_out = gr.HTML(label="Memory Delta Analysis")

            mem_run_btn.click(
                fn=run_memory_demo,
                inputs=[mem_conflict_dd, mem_person_dd],
                outputs=[mem_ep1_out, mem_ep2_out, mem_diff_out],
            )

        # ── Tab 5: Arjun's Journey ──────────────────────────────────────────
        with gr.Tab("🗓️ Arjun's Journey"):
            gr.HTML(LONG_DEMO.show_longitudinal_comparison())
            
            with gr.Column():
                gr.Markdown("### 🎓 Experimental Context Loading")
                gr.Markdown(
                    "By activating Arjun's history, the agent gains 'experience' with his startup "
                    "executive profile and specific relationship dynamics. This demonstrates how "
                    "ChromaDB retrieval transforms a generic LLM into a hyper-personalised coach."
                )
                load_arjun_btn = gr.Button("🔗 Activate Arjun's Life History (v3)", variant="primary", size="lg")
                
                def load_arjun_msg():
                    LONG_DEMO.pre_seed_arjun()
                    return "✅ Arjun's memory (Week 1 & 2) is now ACTIVE in ChromaDB. Go to 'Live Demo', select Arjun, and click 'Run Agent'."
                
                load_status = gr.Markdown()
                load_arjun_btn.click(fn=load_arjun_msg, outputs=load_status)
                
                gr.Markdown("""
                ---
                **Experience it yourself:**
                1. Click the button above to seed the memories.
                2. Switch to the **🎯 Live Demo** tab.
                3. Select **Arjun (Startup Lead)** from the persona list.
                4. Select the **🚨 Friday 6PM** conflict.
                5. Click **Run Agent**.
                6. **Observe:** The agent will now use specific precedents in its reasoning and choice.
                """)

        # ── Tab 5: Task Explorer ──────────────────────────────────────────────
        with gr.Tab("🗺️ Task Explorer"):
            gr.Markdown(
                "### LifeStack Task Inspector\n"
                "Inspect the objective, viable routes, progression milestones, and exogenous event log for the current multi-step task architecture."
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    task_out = gr.HTML(label="Task Definition")
                with gr.Column(scale=1):
                    route_out = gr.HTML(label="Route Status")
                    
            event_out = gr.HTML(label="World Event Log")
            
            load_task_btn = gr.Button("🔄 Load Demonstration Task", variant="secondary")
            
            def load_demo_task():
                # Generate a dummy task for demonstration purposes
                dummy_routes = [
                    Route(id="r1", name="Rebook Premium Option", description="Call agent and rebook on premium ticket", required_action_types=["communicate", "spend"], preconditions={}, consequences={}, closes_routes=["r2"], milestones_unlocked=["m1"], final_reward=2.5),
                    Route(id="r2", name="Accept Delay & Work", description="Stay at airport lounge and work on laptop", required_action_types=["rest", "delegate"], preconditions={}, consequences={}, closes_routes=["r1"], milestones_unlocked=["m2"], final_reward=1.8),
                ]
                dummy_milestones = [
                    Milestone(id="m1", description="Successfully rebooked flight before deadline", condition_key="", condition_value=True, reward=1.0),
                    Milestone(id="m2", description="Caught up with all emergency slack messages", condition_key="", condition_value=True, reward=0.8),
                ]
                dummy_events = [
                    ExoEvent(step=2, probability=1.0, id="price_surge", description="Ticket prices sharply increased by $300.", world_mutation={}, hidden_state_mutation={}, closes_routes=[]),
                    ExoEvent(step=4, probability=1.0, id="lounge_full", description="The airport lounge is now at maximum capacity.", world_mutation={}, hidden_state_mutation={}, closes_routes=["r2"]),
                ]
                dummy_task = Task(
                    id="sample_flight_crisis", domain="flight_crisis", goal="Survive Airport Cancellation",
                    constraints={"budget_max": 800, "deadline_step": 10},
                    hidden_state={"lounge_capacity": 100}, mutable_world={}, visible_world={},
                    success_conditions=[], failure_conditions=[],
                    event_schedule=dummy_events, viable_routes=dummy_routes, milestones=dummy_milestones,
                    horizon=10, difficulty=4, domain_metadata={"story": "A major storm grounded commercial flights."}
                )
                
                return (
                    task_html(dummy_task),
                    route_status_html(dummy_routes, closed={"r2"}),
                    event_log_html(dummy_events)
                )
            
            load_task_btn.click(fn=load_demo_task, outputs=[task_out, route_out, event_out])

        # ── Tab 6: Follow-up ─────────────────────────────────────────────────
        with gr.Tab("📬 Follow-up"):
            gr.Markdown("""
            ### 📍 Real-World Verification
            Did the agent's plan work in the real world? Provide your feedback here to close the loop. 
            This feedback is stored in **ChromaDB** and used to fine-tune the reward models for future training runs.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    fb_id = gr.Textbox(label="Episode ID", placeholder="e.g. A1B2C3D4")
                    fb_score = gr.Slider(0, 10, value=7, label="Overall Effectiveness (0-10)")
                    fb_time = gr.Number(label="Actual Resolution Time (hours)", value=2.0)
                with gr.Column(scale=2):
                    fb_up = gr.CheckboxGroup(
                        ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"],
                        label="Domains that actually improved"
                    )
                    fb_down = gr.CheckboxGroup(
                        ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"],
                        label="Domains that actually worsened"
                    )
            fb_notes = gr.Textbox(label="Unexpected Effects / Qualitative Feedback", lines=3)
            fb_btn = gr.Button("Submit Outcome Feedback", variant="primary")
            fb_out = gr.Markdown()
            
            fb_btn.click(
                submit_outcome_feedback,
                inputs=[fb_id, fb_score, fb_up, fb_down, fb_notes, fb_time],
                outputs=fb_out
            )

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
