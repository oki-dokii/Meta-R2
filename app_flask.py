"""
app_flask.py — LifeStack Flask Portal (FULL FEATURE PARITY)
Complete migration of the Gradio demo to a Flask-native architecture.
Includes: Live Demo, Custom Situations, Gmail Sync, Longitudinal Analysis, Task Explorer.
"""

import os
import json
import copy
import uuid
import datetime
from collections import deque
from flask import Flask, render_template, request, jsonify, session
from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
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
from intake.calendar_intake import CalendarIntake
from core.task import Task, ExoEvent, Route, Milestone
from core.feedback import OutcomeFeedback, compute_human_feedback_reward
from core.cascade_utils import animate_cascade

app = Flask(__name__)
app.secret_key = "lifestack_secret_key_2026"

# ─── Global Instances ───
AGENT  = LifeStackAgent()
MEMORY = LifeStackMemory(silent=True)
INTAKE = LifeIntake()
USER_HEALTH_OVERRIDES: dict = {}          # persisted health/calendar metric deltas
EPISODE_HISTORY: deque = deque(maxlen=5)  # ring buffer, most recent first

@app.route('/api/history', methods=['GET'])
@app.route('/api/history/list', methods=['GET'])
def get_history():
    summaries = [
        {
            "id":        ep.get("action", {}).get("id", ""),
            "conflict":  ep.get("conflict", {}).get("title", "Unknown"),
            "person":    ep.get("conflict", {}).get("person", "Unknown"),
            "reward":    ep.get("action", {}).get("reward", 0.0),
            "timestamp": ep.get("timestamp", ""),
        }
        for ep in EPISODE_HISTORY
    ]
    return jsonify(summaries)

@app.route('/api/history/replay/<episode_id>', methods=['GET'])
def replay_episode(episode_id):
    for ep in EPISODE_HISTORY:
        if ep.get("action", {}).get("id", "") == episode_id:
            return jsonify(ep)
    return jsonify({"error": "Episode not found"}), 404

GMAIL    = GmailIntake()
CALENDAR = CalendarIntake()
LONG_DEMO = LongitudinalDemo()
DEMO_PREDICTOR = ConflictPredictor()

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

CONFLICT_CHOICES = {t.title: t for t in TEMPLATES}

# ─── Visual Helpers ───
DOMAIN_EMOJI = {
    "career": "💼", "finances": "💰", "relationships": "❤️",
    "physical_health": "💪", "mental_wellbeing": "🧠", "time": "📅",
}
INVERTED_METRICS = {"stress_level", "debt_pressure", "workload", "commute_burden", "admin_overhead"}

_DOMAINS = ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"]

def compute_domain_health(metrics_flat: dict) -> dict:
    """Compute per-domain health score (0-100) from flat metrics. Inverted metrics are flipped."""
    health = {}
    for dom in _DOMAINS:
        subs = {k: v for k, v in metrics_flat.items() if k.startswith(dom + ".")}
        if not subs:
            health[dom] = 50.0
            continue
        scores = []
        for k, v in subs.items():
            sub = k.split(".")[1]
            scores.append((100.0 - v) if sub in INVERTED_METRICS else float(v))
        health[dom] = round(sum(scores) / len(scores), 1)
    return health

def _normalize_action_metric_changes(action) -> None:
    fixed_changes = {}
    for path, delta in action.primary.metric_changes.items():
        raw_path = str(path)
        if "." not in raw_path:
            raw_path = f"{action.primary.target_domain}.{raw_path}"
        norm_path = normalize_metric_path(raw_path)
        if not is_valid_metric_path(norm_path): continue
        try:
            fixed_changes[norm_path] = float(delta)
        except (ValueError, TypeError): continue
    action.primary.metric_changes = fixed_changes

# ─── Routes ───
@app.route('/')
def index():
    return render_template('index.html', 
                          persons=list(PERSONS.keys()), 
                          conflicts=list(CONFLICT_CHOICES.keys()))

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    data = request.json
    conflict_label = data.get('conflict')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    base_metrics = LifeMetrics()
    # Apply any uploaded health/calendar overrides
    for path, delta in USER_HEALTH_OVERRIDES.items():
        if '.' in path:
            dom, sub = path.split('.', 1)
            dom_obj = getattr(base_metrics, dom, None)
            if dom_obj and hasattr(dom_obj, sub):
                setattr(dom_obj, sub, max(0.0, min(100.0, getattr(dom_obj, sub) + delta)))
    flat = base_metrics.flatten()
    DEMO_PREDICTOR.add_snapshot(base_metrics)
    return jsonify({
        "status": "success",
        "metrics": flat,
        "prediction": {
            "summary": DEMO_PREDICTOR.get_prediction_summary(),
            "risk_score": DEMO_PREDICTOR.get_risk_score()
        }
    })

@app.route('/api/simulation/cascade', methods=['POST'])
def get_cascade_frames():
    data = request.json
    conflict_label = data.get('conflict')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    base = LifeMetrics()
    frames = animate_cascade(conflict.primary_disruption, base)
    # Feed each cascade frame into the predictor so trajectory is populated
    for frame in frames:
        snap = LifeMetrics()
        for path, val in frame['flat'].items():
            if '.' in path:
                dom, sub = path.split('.', 1)
                dom_obj = getattr(snap, dom, None)
                if dom_obj and hasattr(dom_obj, sub):
                    setattr(dom_obj, sub, float(val))
        DEMO_PREDICTOR.add_snapshot(snap)
    return jsonify({"frames": frames})

@app.route('/api/simulation/graph', methods=['GET'])
def get_dependency_graph():
    graph = DependencyGraph()
    nodes = []
    edges = []
    
    # Flatten metrics to get all nodes
    metrics = LifeMetrics().flatten()
    for path in metrics.keys():
        dom, sub = path.split('.')
        nodes.append({
            "id": path,
            "label": sub.replace('_', ' '),
            "group": dom
        })
        
    for src, targets in graph.edges.items():
        for target, weight in targets:
            edges.append({
                "from": src,
                "to": target,
                "value": abs(weight),
                "arrows": "to",
                "color": {"color": "#4ade80" if weight > 0 else "#ef4444", "opacity": 0.2}
            })
            
    return jsonify({"nodes": nodes, "edges": edges})

@app.route('/api/simulation/action', methods=['POST'])
def perform_action():
    data = request.json
    person_label = data.get('person')
    conflict_label = data.get('conflict')
    memory_enabled = data.get('use_memory', False)
    
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    person = PERSONS.get(person_label, PERSONS["Alex (Executive) — driven, high-stress"])
    
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
    
    before_metrics = copy.deepcopy(env.state.current_metrics)
    before_budget = copy.deepcopy(env.state.budget)
    
    # RAG: Build few-shot context from ChromaDB if enabled
    few_shot = ""
    retrieved = []
    if memory_enabled:
        few_shot = MEMORY.build_few_shot_prompt(conflict.title, before_metrics.flatten())
        retrieved = MEMORY.retrieve_similar(conflict.title, before_metrics.flatten())
        
    action = AGENT.get_action(before_metrics, before_budget, conflict, person, few_shot_context=few_shot)
    _normalize_action_metric_changes(action)
    
    uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost, 
                                     before_metrics.mental_wellbeing.stress_level)
    
    env_action = LifeStackAction.from_agent_action(action)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
    
    obs = env.step(env_action)
    
    episode_id = "".join(str(uuid.uuid4()).split("-")[:2]).upper()

    # Store decision in memory for future RAG
    MEMORY.store_decision(
        conflict_title=conflict.title,
        action_type=action.primary.action_type,
        target_domain=action.primary.target_domain,
        reward=obs.reward,
        metrics_snapshot=before_metrics.flatten(),
        reasoning=action.reasoning,
        episode_id=episode_id,
    )
    
    cf_data = generate_counterfactuals(AGENT, before_metrics, before_budget, conflict, person, action)
    
    result = {
        "metrics": obs.metrics,
        "domain_health": compute_domain_health(obs.metrics),
        "action": {
            "type": action.primary.action_type,
            "target": action.primary.target_domain,
            "description": action.primary.description,
            "reasoning": action.reasoning,
            "reward": obs.reward,
            "uptake": uptake,
            "cost": action.primary.resource_cost,
            "id": episode_id,
            "memories_retrieved": retrieved
        },
        "counterfactuals": cf_data,
        "prediction": {
            "summary": DEMO_PREDICTOR.get_prediction_summary(),
            "risk_score": DEMO_PREDICTOR.get_risk_score()
        },
        "conflict": {
            "title": conflict.title,
            "person": person.name
        },
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }
    
    # Store in history
    EPISODE_HISTORY.appendleft(result)
        
    return jsonify(result)

# ─── 7-Day Trajectory ───
@app.route('/api/simulation/trajectory', methods=['POST'])
def get_trajectory():
    """
    Run the agent action then perform a 7-step rollout.
    Returns per-day metric snapshots for the forecast panel.
    """
    data = request.json
    conflict_label = data.get('conflict')
    person_label   = data.get('person')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    person   = PERSONS.get(person_label, PERSONS["Alex (Executive) — driven, high-stress"])

    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})

    before_metrics = copy.deepcopy(env.state.current_metrics)
    before_budget  = copy.deepcopy(env.state.budget)

    action = AGENT.get_action(before_metrics, before_budget, conflict, person)
    _normalize_action_metric_changes(action)
    uptake = person.respond_to_action(
        action.primary.action_type, action.primary.resource_cost,
        before_metrics.mental_wellbeing.stress_level,
    )
    env_action = LifeStackAction.from_agent_action(action)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}

    obs = env.step(env_action)
    day0_metrics = dict(obs.metrics)

    # ── 7-day drift simulation ────────────────────────────────────────────────
    # env.rollout() fires null actions (actions_taken=0) which trigger the
    # INACTION_PENALTY every step → constant reward and flat metric lines.
    # Instead: simulate realistic daily drift as the unresolved conflict
    # continues to erode metrics, modulated by persona neuroticism.
    GAMMA = 0.9
    trajectory = []
    cumulative = 0.0

    # Per-day erosion: 1/10 of the original conflict disruption per day
    # (representing the unresolved portion gradually worsening).
    daily_erosion = {k: v * 0.10 for k, v in conflict.primary_disruption.items()}

    # Neuroticism amplifier: anxious people suffer more from unresolved stress
    neuroticism_amp = 1.0 + (person.neuroticism - 0.5) * 0.6   # 0.7–1.3 range

    current = copy.deepcopy(day0_metrics)   # plain dict, mutated each day

    for t in range(1, 8):
        prev = dict(current)
        changes = {}

        for metric_key, base_delta in daily_erosion.items():
            # Inverted metrics (stress, workload, etc.) erode UP; others erode DOWN
            _INVERTED = {"stress_level", "workload", "debt_pressure",
                         "commute_burden", "admin_overhead"}
            sub = metric_key.split(".")[-1]
            direction = 1 if sub in _INVERTED else -1
            magnitude = abs(base_delta) * neuroticism_amp * direction
            changes[metric_key] = magnitude

        # Small natural recovery on domains NOT hit by the conflict
        # (life keeps going; effort in other areas pays small dividends)
        hit_domains = {k.split(".")[0] for k in conflict.primary_disruption}
        for dom in ["career", "finances", "relationships",
                    "physical_health", "mental_wellbeing", "time"]:
            if dom not in hit_domains:
                for k in [k for k in current if k.startswith(dom + ".")]:
                    changes[k] = changes.get(k, 0) + 0.5   # tiny passive gain

        for k, delta in changes.items():
            current[k] = round(max(0.0, min(100.0, current.get(k, 70.0) + delta)), 2)

        # Compute reward as avg change across all metrics
        total_delta = sum(current[k] - prev.get(k, 70.0) for k in current)
        avg_delta = total_delta / max(len(current), 1)
        step_reward = round(max(-1.0, min(1.0, avg_delta / 50.0)), 5)

        disc = round(GAMMA ** t * step_reward, 5)
        cumulative += disc
        trajectory.append({
            "step": t,
            "reward": step_reward,
            "metrics": dict(current),
            "discounted_contribution": disc,
        })

    return jsonify({
        "action": {
            "type": action.primary.action_type,
            "target": action.primary.target_domain,
            "reasoning": action.reasoning,
            "reward": obs.reward,
        },
        "day0_metrics": day0_metrics,
        "discounted_reward": round(cumulative, 5),
        "trajectory": trajectory,
    })


# ─── Custom Situation Entry ───
@app.route('/api/custom/run', methods=['POST'])
def run_custom():
    data = request.json
    situation_input = data.get('situation', "")
    
    # Map sliders to metrics
    m = LifeMetrics()
    m.career.stress_level = float(data.get('work_stress', 5)) * 10
    m.finances.debt_pressure = float(data.get('money_stress', 5)) * 10
    m.relationships.conflict_frequency = (10 - float(data.get('rel_quality', 5))) * 10
    m.physical_health.energy_level = float(data.get('energy_level', 5)) * 10
    m.time.free_time = (10 - float(data.get('time_pressure', 5))) * 10
    
    # Apply uploaded health/calendar overrides to custom metrics
    for path, delta in USER_HEALTH_OVERRIDES.items():
        if '.' in path:
            dom, sub = path.split('.', 1)
            dom_obj = getattr(m, dom, None)
            if dom_obj and hasattr(dom_obj, sub):
                setattr(dom_obj, sub, max(0.0, min(100.0, getattr(dom_obj, sub) + delta)))

    gmail_signals = data.get('gmail_signals')
    if gmail_signals:
        # Merge digital signals if provided
        for k, v in gmail_signals.items():
            parts = k.split(".")
            if len(parts) == 2:
                dom = getattr(m, parts[0], None)
                if dom and hasattr(dom, parts[1]):
                    setattr(dom, parts[1], v)

    # Extract conflict from text using LLM
    conflict = INTAKE.extract_conflict(situation_input, m)
    pers_dict = INTAKE.get_personality_from_description(situation_input)
    person = SimPerson(
        name=pers_dict.get("name", "Inferred Self"),
        openness=pers_dict.get("openness", 0.5),
        conscientiousness=pers_dict.get("conscientiousness", 0.5),
        extraversion=pers_dict.get("extraversion", 0.5),
        agreeableness=pers_dict.get("agreeableness", 0.5),
        neuroticism=pers_dict.get("neuroticism", 0.5)
    )
    
    budget = ResourceBudget(time=24, money=1000, energy=100)
    action = AGENT.get_action(m, budget, conflict, person)
    _normalize_action_metric_changes(action)
    
    uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost, 
                                     m.mental_wellbeing.stress_level)
    
    env = LifeStackEnv()
    env.state.current_metrics = copy.deepcopy(m)
    env.state.budget = budget
    
    env_action = LifeStackAction.from_agent_action(action)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
    obs = env.step(env_action)
    
    return jsonify({
        "before_metrics": m.flatten(),
        "after_metrics": obs.metrics,
        "domain_health": compute_domain_health(obs.metrics),
        "action": {
            "type": action.primary.action_type,
            "target": action.primary.target_domain,
            "description": action.primary.description,
            "reasoning": action.reasoning,
            "id": "".join(str(uuid.uuid4()).split("-")[:2]).upper()
        },
        "person": {"name": person.name or "Inferred Self"}
    })

@app.route('/api/gmail/sync', methods=['POST'])
def sync_gmail():
    signals, metric_deltas, summary, is_demo = GMAIL.sync()
    return jsonify({
        "status": "success",
        "signals": metric_deltas,
        "raw": signals,
        "summary": summary,
        "is_demo": is_demo,
    })


@app.route('/api/digital/sync', methods=['POST'])
def digital_sync():
    """
    Unified Digital Sync — Gmail + Google Calendar + Fitness (demo payload).
    Tries real OAuth for Gmail and Calendar; falls back to demo_signals.json on failure.
    Fitness is always served from the demo payload (no first-party fitness API scope).
    Returns merged metric deltas, per-source raw signals, and a demo flag per source.
    """
    import json as _json
    demo_path = os.path.join(os.path.dirname(__file__), 'data', 'demo_signals.json')

    with open(demo_path) as f:
        demo_full = _json.load(f)

    # Gmail
    gmail_signals, gmail_deltas, gmail_summary, gmail_is_demo = GMAIL.sync()

    # Calendar
    cal_signals, cal_deltas, cal_is_demo = CALENDAR.sync()

    # Fitness — always demo (no live fitness API)
    fitness_signals = demo_full['fitness']
    fitness_deltas = {
        "physical_health.sleep_quality": demo_full['derived_metric_deltas']['physical_health.sleep_quality'],
        "physical_health.energy_level": demo_full['derived_metric_deltas']['physical_health.energy_level'],
        "physical_health.exercise_consistency": demo_full['derived_metric_deltas']['physical_health.exercise_consistency'],
        "mental_wellbeing.stress_level": demo_full['derived_metric_deltas']['mental_wellbeing.stress_level'],
    }
    fitness_is_demo = True

    # Merge all deltas (last writer wins — Calendar > Gmail for overlapping keys)
    merged_deltas = {}
    merged_deltas.update(gmail_deltas)
    merged_deltas.update(cal_deltas)
    merged_deltas.update(fitness_deltas)

    return jsonify({
        "status": "success",
        "merged_deltas": merged_deltas,
        "sources": {
            "gmail": {
                "signals": gmail_signals if isinstance(gmail_signals, dict) else {},
                "summary": gmail_summary,
                "is_demo": gmail_is_demo,
            },
            "calendar": {
                "signals": cal_signals,
                "summary": cal_signals.get("summary", ""),
                "is_demo": cal_is_demo,
            },
            "fitness": {
                "signals": fitness_signals,
                "summary": fitness_signals.get("summary", ""),
                "is_demo": True,
            },
        },
        "persona_note": demo_full.get("persona", "Jordan (PM at Series-B startup)"),
    })

@app.route('/api/arjun/activate', methods=['POST'])
def activate_arjun():
    LONG_DEMO.pre_seed_arjun()
    return jsonify({"status": "success", "message": "Arjun's memory (Week 1 & 2) is now ACTIVE in ChromaDB."})

@app.route('/api/task/demo', methods=['GET'])
def get_demo_task():
    conflict_label = request.args.get('conflict')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)

    # Keyword → action types mapping for route inference
    _ROUTE_ACTION_MAP = {
        "leave": ["rest", "delegate"], "rest": ["rest"], "sleep": ["rest"],
        "medical": ["rest", "spend"], "specialist": ["spend", "communicate"],
        "diet": ["rest", "execute"], "negotiate": ["negotiate", "communicate"],
        "call": ["communicate"], "pay": ["spend", "execute"],
        "hire": ["spend", "delegate"], "delegate": ["delegate"],
        "plan": ["execute", "communicate"], "date": ["spend", "communicate"],
        "talk": ["communicate"], "network": ["communicate", "negotiate"],
        "remote": ["communicate", "reschedule"], "reschedule": ["reschedule"],
        "work": ["execute"], "book": ["spend", "execute"], "repair": ["spend"],
        "bus": ["spend", "execute"], "rideshare": ["spend"], "carpool": ["communicate"],
        "disconnect": ["rest", "deprioritize"], "refuse": ["negotiate", "communicate"],
        "deny": ["negotiate"], "scan": ["inspect", "execute"], "lawyer": ["spend", "delegate"],
    }

    def _infer_action_types(decision_text: str) -> list:
        lower = decision_text.lower()
        for keyword, actions in _ROUTE_ACTION_MAP.items():
            if keyword in lower:
                return actions
        return ["execute", "communicate"]

    # Route reward scales with difficulty
    base_reward = round(1.0 + conflict.difficulty * 0.3, 1)
    routes = [
        {
            "id": f"r{i+1}",
            "name": dec,
            "description": f"Approach: {dec}",
            "required_action_types": _infer_action_types(dec),
            "final_reward": base_reward,
        }
        for i, dec in enumerate(conflict.decisions_required)
    ]

    # Milestones — one per disrupted metric, reward proportional to hit magnitude
    _METRIC_LABELS = {
        "physical_health.fitness": "Restore fitness baseline",
        "physical_health.sleep_quality": "Recover healthy sleep",
        "physical_health.energy": "Rebuild energy reserves",
        "finances.liquidity": "Stabilise immediate cash flow",
        "finances.long_term_health": "Protect long-term financial health",
        "finances.debt_pressure": "Reduce debt pressure below danger zone",
        "relationships.romantic": "Repair romantic relationship",
        "relationships.family": "Resolve family situation",
        "relationships.social": "Restore social connections",
        "relationships.professional_network": "Protect professional reputation",
        "career.workload": "Bring workload back to manageable level",
        "career.stability": "Secure career stability",
        "career.satisfaction": "Recover career satisfaction",
        "career.growth_trajectory": "Get growth trajectory back on track",
        "mental_wellbeing.stress_level": "Reduce stress to safe level",
        "mental_wellbeing.motivation": "Recover motivation",
        "mental_wellbeing.clarity": "Restore mental clarity",
        "mental_wellbeing.emotional_stability": "Regain emotional stability",
        "time.free_hours_per_week": "Reclaim personal time",
        "time.commute_burden": "Resolve commute disruption",
        "time.admin_overhead": "Clear admin backlog",
    }
    milestones = [
        {
            "id": f"m{idx+1}",
            "metric": metric_key,
            "description": _METRIC_LABELS.get(
                metric_key,
                f"Recover {metric_key.split('.')[-1].replace('_', ' ')}"
            ),
            "impact": delta,
            "reward": round(abs(delta) / 40.0, 1),
        }
        for idx, (metric_key, delta) in enumerate(conflict.primary_disruption.items())
    ]

    # World events scaled by difficulty (display only)
    _EVENTS_BY_DIFF = {
        1: [],
        2: [{"step": 3, "id": "minor_complication", "description": "Situation slightly worsened — act soon."}],
        3: [
            {"step": 2, "id": "cascade_trigger", "description": "Cascading effect detected — secondary metric dropping."},
            {"step": 5, "id": "window_closing", "description": "Optimal resolution window is closing."},
        ],
        4: [
            {"step": 1, "id": "immediate_pressure", "description": "Immediate action required — every step costs resources."},
            {"step": 3, "id": "escalation_risk", "description": "Risk of escalation if no action taken."},
            {"step": 6, "id": "resource_drain", "description": "Budget shrinking faster than expected."},
        ],
        5: [
            {"step": 1, "id": "critical_state", "description": "CRITICAL: Multiple systems failing simultaneously."},
            {"step": 2, "id": "cascade_chain", "description": "Cascade chain activated — metrics affecting each other."},
            {"step": 4, "id": "last_window", "description": "Last viable window for intervention."},
            {"step": 7, "id": "terminal_risk", "description": "Terminal failure risk if not resolved this step."},
        ],
    }
    events = _EVENTS_BY_DIFF.get(conflict.difficulty, [])

    # Primary domain = worst-hit metric
    top_metric = max(conflict.primary_disruption.items(), key=lambda x: abs(x[1]))[0]
    domain_tag = top_metric.split(".")[0]

    return jsonify({
        "goal": f"Resolve '{conflict.title}' without cascading failure",
        "difficulty": conflict.difficulty,
        "story": conflict.story,
        "domain": domain_tag,
        "budget": conflict.resource_budget,
        "disruption": conflict.primary_disruption,
        "routes": routes,
        "milestones": milestones,
        "events": events,
        "conflict_id": conflict.id,
    })

@app.route('/api/task/list', methods=['GET'])
def list_tasks():
    """Return all conflict templates formatted as browsable task cards."""
    result = []
    for t in TEMPLATES:
        disruption_events = [
            {
                "step": 1,
                "id": k.replace('.', '_'),
                "description": f"{k.split('.')[-1].replace('_', ' ').title()}: {'+' if v > 0 else ''}{v:.0f}",
                "probability": 1.0
            }
            for k, v in (t.primary_disruption or {}).items()
        ]
        decision_routes = [
            {
                "name": d,
                "description": f"Approach: {d}",
                "actions": [],
                "reward": round(0.8 + t.difficulty * 0.3, 1)
            }
            for d in (t.decisions_required or [])
        ]
        result.append({
            "id": t.id,
            "title": t.title,
            "story": t.story,
            "difficulty": t.difficulty,
            "resource_budget": t.resource_budget or {},
            "routes": decision_routes,
            "milestones": [{"id": f"resolve_{t.id}", "description": f"Resolve '{t.title}' without cascading failure", "reward": round(0.5 + t.difficulty * 0.25, 1)}],
            "events": disruption_events,
        })
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    from collections import Counter
    stats = MEMORY.get_stats()
    all_records = []
    try:
        raw = MEMORY.collection.get(include=["metadatas"])
        all_records = raw.get("metadatas", [])
    except Exception:
        pass

    stats["feedback_count"] = len([m for m in all_records if m.get("type") == "feedback"])

    # Sort by timestamp so the reward chart is chronological
    timed = [m for m in all_records if "reward" in m and "timestamp" in m]
    timed.sort(key=lambda m: m.get("timestamp", ""))
    untimed = [m for m in all_records if "reward" in m and "timestamp" not in m]
    ordered = timed + untimed
    rewards = [round(float(m["reward"]), 4) for m in ordered]
    stats["reward_history"] = rewards[-20:] if rewards else []

    # Average reward
    stats["average_reward"] = round(sum(rewards) / len(rewards), 3) if rewards else None

    # Action type distribution (top 5)
    action_counts = Counter(m["action_type"] for m in all_records if m.get("action_type"))
    stats["action_distribution"] = dict(action_counts.most_common(5))

    # Most targeted domain
    domain_counts = Counter(m["target_domain"] for m in all_records if m.get("target_domain"))
    stats["top_domain"] = domain_counts.most_common(1)[0][0] if domain_counts else None

    # Last agent insight — derive from most recent high-reward records
    recent_high = sorted(
        [m for m in all_records if m.get("reward", 0) > 0.3 and m.get("reasoning")],
        key=lambda m: m.get("timestamp", ""), reverse=True
    )
    stats["last_insight"] = recent_high[0]["reasoning"][:120] if recent_high else None

    return jsonify(stats)

@app.route('/api/history/reset', methods=['POST'])
def reset_history():
    """Wipe all memories and feedback from ChromaDB."""
    try:
        MEMORY.collection.delete(where={}) # Delete everything
        return jsonify({"status": "success", "message": "History and memories cleared."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/feedback/submit', methods=['POST'])
def submit_feedback():
    data = request.json
    try:
        feedback = OutcomeFeedback(
            episode_id=data.get('episode_id'),
            submitted_at=datetime.datetime.now(),
            overall_effectiveness=int(data.get('score', 7)),
            domains_improved=data.get('improved', []),
            domains_worsened=data.get('worsened', []),
            unexpected_effects=data.get('notes', ""),
            resolution_time_hours=float(data.get('time', 1.0))
        )
        MEMORY.store_feedback(feedback)
        return jsonify({"status": "success", "message": f"Feedback stored for episode {feedback.episode_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ─── Feature F1 helper: random action baseline ───
_ACTION_TYPES = ["negotiate", "communicate", "delegate", "spend", "reschedule", "rest", "deprioritize", "execute"]

def _random_action(conflict, person):
    """Purely random action baseline — worst possible agent, used for ablation floor."""
    import random as _r
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
    flat = env.state.current_metrics.flatten()
    atype = _r.choice(_ACTION_TYPES)
    dom = _r.choice(_DOMAINS)
    key = f"{dom}.stress_level" if dom in ("career", "mental_wellbeing") else f"{dom}.liquidity" if dom == "finances" else f"{dom}.energy_level"
    mc = {key: _r.uniform(-20, 20)}
    rc = {"time": _r.uniform(0.5, 3.0), "energy": _r.uniform(5, 30)}
    uptake = person.respond_to_action(atype, rc, flat.get("mental_wellbeing.stress_level", 70))
    env_action = LifeStackAction(action_type=atype, target=dom,
                                  metric_changes={k: v * uptake for k, v in mc.items()},
                                  resource_cost=rc, reasoning="Random baseline.", actions_taken=1)
    obs = env.step(env_action)
    return {"metrics": obs.metrics, "action": {"type": atype, "target": dom,
            "description": "Random action (ablation floor).",
            "reasoning": "Random baseline.", "reward": obs.reward, "cost": rc}}


# ─── Feature A: Trained vs Untrained Comparison ───
BASELINE_ACTION_MAP = {
    "career":           ("negotiate",    {"career.workload": -12.0, "mental_wellbeing.stress_level": -4.0},    {"time": 1.5, "energy": 20.0}, "Negotiate workload with manager."),
    "finances":         ("spend",        {"finances.liquidity": -200.0, "mental_wellbeing.stress_level": -8.0}, {"time": 1.0, "energy": 10.0}, "Spend to resolve financial pressure."),
    "relationships":    ("communicate",  {"relationships.romantic": 8.0, "mental_wellbeing.stress_level": -5.0},{"time": 0.5, "energy": 8.0},  "Call partner to check in."),
    "physical_health":  ("rest",         {"physical_health.energy_level": 12.0, "mental_wellbeing.stress_level": -6.0}, {"time": 1.0}, "Rest to recover energy."),
    "mental_wellbeing": ("rest",         {"mental_wellbeing.stress_level": -15.0, "physical_health.sleep_quality": 5.0},  {"time": 1.0}, "Take a break to reduce stress."),
    "time":             ("reschedule",   {"time.free_hours_per_week": 6.0, "career.workload": -8.0},            {"time": 1.5, "energy": 12.0}, "Reschedule non-critical tasks."),
}

def _run_baseline(conflict, person):
    """Rule-based baseline: pick the action for the worst-scoring domain."""
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
    flat = env.state.current_metrics.flatten()

    domain_scores = {}
    for dom in ["career", "finances", "relationships", "physical_health", "mental_wellbeing", "time"]:
        subs = {k: v for k, v in flat.items() if k.startswith(dom + ".")}
        domain_scores[dom] = sum(subs.values()) / len(subs) if subs else 70.0

    worst_dom = min(domain_scores, key=domain_scores.get)
    atype, mc, rc, desc = BASELINE_ACTION_MAP.get(worst_dom, BASELINE_ACTION_MAP["mental_wellbeing"])

    uptake = person.respond_to_action(atype, rc, flat.get("mental_wellbeing.stress_level", 70))
    scaled_mc = {k: v * uptake for k, v in mc.items()}

    env_action = LifeStackAction(
        action_type=atype,
        target=worst_dom,
        metric_changes=scaled_mc,
        resource_cost=rc,
        reasoning=f"Rule-based: {worst_dom} scored {domain_scores[worst_dom]:.1f} — lowest domain.",
        actions_taken=1,
    )
    obs = env.step(env_action)
    return {
        "metrics": obs.metrics,
        "action": {
            "type": atype,
            "target": worst_dom,
            "description": desc,
            "reasoning": env_action.reasoning,
            "reward": obs.reward,
            "cost": rc,
        }
    }

def _run_agent_comparison_side(conflict, person, api_only: bool):
    """Run one side of the comparison: api_only=True → untrained LLM, False → GRPO-trained."""
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
    before_metrics = copy.deepcopy(env.state.current_metrics)
    before_budget = copy.deepcopy(env.state.budget)
    action = AGENT.get_action(before_metrics, before_budget, conflict, person, api_only=api_only)
    _normalize_action_metric_changes(action)
    uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost,
                                      before_metrics.mental_wellbeing.stress_level)
    env_action = LifeStackAction.from_agent_action(action)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
    obs = env.step(env_action)
    return {
        "metrics": obs.metrics,
        "action": {
            "type": action.primary.action_type,
            "target": action.primary.target_domain,
            "description": action.primary.description,
            "reasoning": action.reasoning,
            "reward": obs.reward,
            "cost": action.primary.resource_cost,
        }
    }


@app.route('/api/comparison/run', methods=['POST'])
def run_comparison():
    """Run same conflict through untrained LLM (no RL) AND GRPO-trained LifeStack agent."""
    data = request.json
    conflict_label = data.get('conflict')
    person_label = data.get('person')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    person = PERSONS.get(person_label, PERSONS["Alex (Executive) — driven, high-stress"])

    # Untrained LLM path — forces Groq API, no GRPO optimization
    try:
        baseline = _run_agent_comparison_side(conflict, person, api_only=True)
    except Exception as e:
        baseline = {"error": str(e)}

    # GRPO-trained agent path — uses local model if available, lazy-loaded
    try:
        trained = _run_agent_comparison_side(conflict, person, api_only=False)
    except Exception as e:
        trained = {"error": str(e)}

    return jsonify({"baseline": baseline, "trained": trained})


# ─── Feature E: Memory Effect Comparison ───
@app.route('/api/memory/compare', methods=['POST'])
def memory_compare():
    """Show the same conflict resolved cold (no memory) vs warm (with RAG memory)."""
    try:
        data = request.json
        conflict_label = data.get('conflict')
        person_label = data.get('person')
        conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
        person = PERSONS.get(person_label, PERSONS["Alex (Executive) — driven, high-stress"])

        def _run_episode(use_memory: bool):
            env = LifeStackEnv()
            env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
            before_metrics = copy.deepcopy(env.state.current_metrics)
            before_budget = copy.deepcopy(env.state.budget)
            few_shot = ""
            retrieved = []
            if use_memory:
                few_shot = MEMORY.build_few_shot_prompt(conflict.title, before_metrics.flatten())
                retrieved = MEMORY.retrieve_similar(conflict.title, before_metrics.flatten())
            action = AGENT.get_action(before_metrics, before_budget, conflict, person, few_shot_context=few_shot)
            _normalize_action_metric_changes(action)
            uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost,
                                              before_metrics.mental_wellbeing.stress_level)
            env_action = LifeStackAction.from_agent_action(action)
            env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
            obs = env.step(env_action)
            MEMORY.store_decision(
                conflict_title=conflict.title,
                action_type=action.primary.action_type,
                target_domain=action.primary.target_domain,
                reward=obs.reward,
                metrics_snapshot=before_metrics.flatten(),
                reasoning=action.reasoning,
            )
            return {
                "metrics": obs.metrics,
                "action": {
                    "type": action.primary.action_type,
                    "target": action.primary.target_domain,
                    "description": action.primary.description,
                    "reasoning": action.reasoning,
                    "reward": obs.reward,
                    "memories_retrieved": retrieved,
                }
            }

        cold = _run_episode(use_memory=False)
        warm = _run_episode(use_memory=True)
        return jsonify({"cold": cold, "warm": warm})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── F2: /api/cascade/frames alias ───
@app.route('/api/cascade/frames', methods=['POST'])
def cascade_frames_alias():
    """Alias route for /api/simulation/cascade — same handler."""
    return get_cascade_frames()


# ─── F4: Personality Comparison with OCEAN scores ───
@app.route('/api/personality/compare', methods=['POST'])
def personality_compare():
    data = request.json
    conflict_label = data.get('conflict')
    person_a_label = data.get('person_a')
    person_b_label = data.get('person_b')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)

    def _run_person(person_label):
        person = PERSONS.get(person_label, list(PERSONS.values())[0])
        env = LifeStackEnv()
        env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
        before_m = copy.deepcopy(env.state.current_metrics)
        before_b = copy.deepcopy(env.state.budget)
        action = AGENT.get_action(before_m, before_b, conflict, person)
        _normalize_action_metric_changes(action)
        uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost,
                                          before_m.mental_wellbeing.stress_level)
        env_action = LifeStackAction.from_agent_action(action)
        env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
        obs = env.step(env_action)
        return {
            "name": person.name,
            "ocean": {
                "openness": round(person.openness * 100),
                "conscientiousness": round(person.conscientiousness * 100),
                "extraversion": round(person.extraversion * 100),
                "agreeableness": round(person.agreeableness * 100),
                "neuroticism": round(person.neuroticism * 100),
            },
            "action": {
                "type": action.primary.action_type,
                "target": action.primary.target_domain,
                "description": action.primary.description,
                "reasoning": action.reasoning,
                "reward": obs.reward,
                "uptake": uptake,
            },
            "metrics": obs.metrics,
            "domain_health": compute_domain_health(obs.metrics),
        }

    try:
        return jsonify({"a": _run_person(person_a_label), "b": _run_person(person_b_label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── F6: Dedicated Counterfactual Generation ───
@app.route('/api/counterfactuals/generate', methods=['POST'])
def counterfactuals_generate():
    data = request.json
    conflict_label = data.get('conflict')
    person_label = data.get('person')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    person = PERSONS.get(person_label, list(PERSONS.values())[0])

    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
    before_m = copy.deepcopy(env.state.current_metrics)
    before_b = copy.deepcopy(env.state.budget)
    action = AGENT.get_action(before_m, before_b, conflict, person)
    _normalize_action_metric_changes(action)

    env_action = LifeStackAction.from_agent_action(action)
    uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost,
                                      before_m.mental_wellbeing.stress_level)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
    obs = env.step(env_action)

    cf_data = generate_counterfactuals(AGENT, before_m, before_b, conflict, person, action)
    return jsonify({
        "counterfactuals": cf_data,
        "actual_action": {
            "type": action.primary.action_type,
            "target": action.primary.target_domain,
            "description": action.primary.description,
            "reasoning": action.reasoning,
            "reward": obs.reward,
            "metrics": obs.metrics,
            "cost": action.primary.resource_cost,
        },
    })


# ─── F7: Memory Ablation Study ───
@app.route('/api/memory/ablation', methods=['POST'])
def memory_ablation():
    """Memory ablation: cold (0 memories) vs warm (RAG-augmented). Surfaces ablation delta."""
    data = request.json
    conflict_label = data.get('conflict')
    person_label = data.get('person')
    conflict = CONFLICT_CHOICES.get(conflict_label, DEMO_CONFLICT)
    person = PERSONS.get(person_label, list(PERSONS.values())[0])

    def _run(use_memory):
        env = LifeStackEnv()
        env.reset(conflict=conflict.primary_disruption, budget={"time": max((conflict.resource_budget or {}).get("time", 20.0), 4.0), "money": max((conflict.resource_budget or {}).get("money", 500.0), 500.0), "energy": max((conflict.resource_budget or {}).get("energy", 100.0), 20.0)})
        before_m = copy.deepcopy(env.state.current_metrics)
        before_b = copy.deepcopy(env.state.budget)
        few_shot, retrieved = "", []
        if use_memory:
            few_shot = MEMORY.build_few_shot_prompt(conflict.title, before_m.flatten())
            retrieved = MEMORY.retrieve_similar(conflict.title, before_m.flatten())
        action = AGENT.get_action(before_m, before_b, conflict, person, few_shot_context=few_shot)
        _normalize_action_metric_changes(action)
        uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost,
                                          before_m.mental_wellbeing.stress_level)
        env_action = LifeStackAction.from_agent_action(action)
        env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
        obs = env.step(env_action)
        MEMORY.store_decision(conflict_title=conflict.title, action_type=action.primary.action_type,
                              target_domain=action.primary.target_domain, reward=obs.reward,
                              metrics_snapshot=before_m.flatten(), reasoning=action.reasoning)
        return {"metrics": obs.metrics, "action": {
            "type": action.primary.action_type, "target": action.primary.target_domain,
            "description": action.primary.description, "reasoning": action.reasoning,
            "reward": obs.reward, "memories_retrieved": retrieved,
        }}

    cold = _run(use_memory=False)
    warm = _run(use_memory=True)
    delta = warm["action"]["reward"] - cold["action"]["reward"]
    return jsonify({"cold": cold, "warm": warm,
                    "ablation_delta": round(delta, 4),
                    "memory_count": len(warm["action"]["memories_retrieved"])})


# ─── F10: Health + Calendar Data Upload ───
@app.route('/api/data/health/upload', methods=['POST'])
def upload_health_data():
    """Accept health/fitness JSON signals and return metric deltas."""
    data = request.json or {}
    sleep = float(data.get('sleep_hours', 7.0))
    hr = float(data.get('resting_heart_rate', 70))
    steps = float(data.get('daily_steps', 8000))
    deltas = {
        "physical_health.sleep_quality": round(min(100, sleep / 8 * 100) - 50, 1),
        "physical_health.energy_level": round(min(100, steps / 10000 * 100) - 50, 1),
        "physical_health.exercise_consistency": round(min(100, steps / 8000 * 70), 1),
        "mental_wellbeing.stress_level": round(max(0.0, 80.0 - hr), 1),
    }
    summary = f"Sleep {sleep:.1f}h | HR {hr:.0f}bpm | Steps {int(steps):,}/day"
    # Persist overrides so future simulations use the uploaded health data
    USER_HEALTH_OVERRIDES.update(deltas)
    return jsonify({"status": "success", "deltas": deltas, "summary": summary,
                    "signals": {"avg_sleep_hours": sleep, "resting_heart_rate": hr, "daily_steps_avg": steps}})


@app.route('/api/data/calendar/upload', methods=['POST'])
def upload_calendar_data():
    """Accept calendar JSON signals and return metric deltas."""
    data = request.json or {}
    occupancy = float(data.get('week_occupancy_pct', 50))
    btb = int(data.get('back_to_back_blocks', 0))
    deadlines = data.get('upcoming_deadlines', [])
    critical_count = sum(1 for d in deadlines if d.get('priority') == 'critical')
    deltas = {
        "time.free_hours_per_week": round(-((occupancy - 50) / 5), 1),
        "time.schedule_control": round(-(occupancy / 10), 1),
        "mental_wellbeing.stress_level": round((occupancy / 10) + (btb * 2), 1),
        "career.workload": round((occupancy - 50) / 2 + critical_count * 5, 1),
    }
    summary = f"Occupancy {occupancy:.0f}% | {len(deadlines)} deadlines ({critical_count} critical)"
    return jsonify({"status": "success", "deltas": deltas, "summary": summary,
                    "signals": {"week_occupancy_pct": occupancy, "back_to_back_blocks": btb,
                                "upcoming_deadlines": deadlines}})


# ─── Global Error Handlers ───
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Slow down!", "details": str(e)}), 429

@app.errorhandler(500)
def server_error_handler(e):
    return jsonify({"error": "Internal server error. The agent might be overwhelmed.", "details": str(e)}), 500

if __name__ == '__main__':
    LONG_DEMO.pre_seed_arjun()
    app.run(host='0.0.0.0', port=7860, debug=True)
