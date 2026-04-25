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
from flask import Flask, render_template, request, jsonify, session
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
from core.cascade_utils import animate_cascade

app = Flask(__name__)
app.secret_key = "lifestack_secret_key_2026"

# ─── Global Instances ───
AGENT  = LifeStackAgent()
MEMORY = LifeStackMemory(silent=True)
INTAKE = LifeIntake()
GMAIL  = GmailIntake()
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
    flat = base_metrics.flatten()
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
    frames = animate_cascade(conflict.primary_disruption, LifeMetrics())
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
    env.reset(conflict=conflict.primary_disruption, budget=conflict.resource_budget)
    
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
    
    # Store decision in memory for future RAG
    MEMORY.store_decision(
        conflict_title=conflict.title,
        action_type=action.primary.action_type,
        target_domain=action.primary.target_domain,
        reward=obs.reward,
        metrics_snapshot=before_metrics.flatten(),
        reasoning=action.reasoning
    )
    
    cf_data = generate_counterfactuals(AGENT, before_metrics, before_budget, conflict, person, action)
    episode_id = "".join(str(uuid.uuid4()).split("-")[:2]).upper()
    
    return jsonify({
        "metrics": obs.metrics,
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
        }
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
    try:
        service = GMAIL.authenticate()
        rel = GMAIL.extract_relationship_signals(service)
        work = GMAIL.extract_work_signals(service)
        signals = GMAIL.to_life_metrics(rel, work)
        summary = GMAIL.get_email_summary(rel, work)
        return jsonify({"status": "success", "signals": signals, "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/arjun/activate', methods=['POST'])
def activate_arjun():
    LONG_DEMO.pre_seed_arjun()
    return jsonify({"status": "success", "message": "Arjun's memory (Week 1 & 2) is now ACTIVE in ChromaDB."})

@app.route('/api/task/demo', methods=['GET'])
def get_demo_task():
    dummy_routes = [
        Route(id="r1", name="Rebook Premium Option", description="Call agent and rebook on premium ticket", required_action_types=["communicate", "spend"], milestones_unlocked=["m1"], final_reward=2.5),
        Route(id="r2", name="Accept Delay & Work", description="Stay at airport lounge and work on laptop", required_action_types=["rest", "delegate"], milestones_unlocked=["m2"], final_reward=1.8),
    ]
    dummy_milestones = [
        Milestone(id="m1", description="Successfully rebooked flight before deadline", reward=1.0),
        Milestone(id="m2", description="Caught up with all emergency slack messages", reward=0.8),
    ]
    dummy_events = [
        ExoEvent(step=2, probability=1.0, id="price_surge", description="Ticket prices sharply increased by $300."),
        ExoEvent(step=4, probability=1.0, id="lounge_full", description="The airport lounge is now at maximum capacity."),
    ]
    task = Task(
        id="sample_flight_crisis", domain="flight_crisis", goal="Survive Airport Cancellation",
        event_schedule=dummy_events, viable_routes=dummy_routes, milestones=dummy_milestones,
        horizon=10, difficulty=4
    )
    return jsonify({
        "goal": task.goal,
        "difficulty": task.difficulty,
        "routes": [{"name": r.name, "description": r.description} for r in dummy_routes],
        "milestones": [{"id": m.id, "description": m.description} for m in dummy_milestones],
        "events": [{"step": e.step, "id": e.id, "description": e.description} for e in dummy_events],
        "story": "A major storm grounded commercial flights."
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = MEMORY.get_stats()
    return jsonify(stats)

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

if __name__ == '__main__':
    LONG_DEMO.pre_seed_arjun()
    app.run(host='0.0.0.0', port=7860, debug=True)
