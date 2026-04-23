"""
app_flask.py — LifeStack Flask Portal
Complete migration of the Gradio demo to a Flask-native architecture.
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

def _metric_color(key: str, val: float) -> str:
    sub = key.split(".")[-1]
    if sub in INVERTED_METRICS:
        return "#f87171" if val > 70 else ("#facc15" if val >= 40 else "#4ade80")
    return "#4ade80" if val > 70 else ("#facc15" if val >= 40 else "#f87171")

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
    person_label = data.get('person')
    conflict_label = data.get('conflict')
    
    conflict = CONFLICT_CHOICES[conflict_label]
    base_metrics = LifeMetrics()
    
    # Simple direct return for start frame
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
    conflict = CONFLICT_CHOICES[conflict_label]
    
    from app import animate_cascade
    frames = animate_cascade(conflict.primary_disruption, LifeMetrics())
    return jsonify({"frames": frames})

@app.route('/api/simulation/action', methods=['POST'])
def perform_action():
    data = request.json
    person_label = data.get('person')
    conflict_label = data.get('conflict')
    
    conflict = CONFLICT_CHOICES[conflict_label]
    person = PERSONS[person_label]
    
    env = LifeStackEnv()
    env.reset(conflict=conflict.primary_disruption, budget=conflict.resource_budget)
    
    before_metrics = copy.deepcopy(env.state.current_metrics)
    before_budget = copy.deepcopy(env.state.budget)
    
    action = AGENT.get_action(before_metrics, before_budget, conflict, person)
    _normalize_action_metric_changes(action)
    
    uptake = person.respond_to_action(action.primary.action_type, action.primary.resource_cost, 
                                     before_metrics.mental_wellbeing.stress_level)
    
    env_action = LifeStackAction.from_agent_action(action)
    env_action.metric_changes = {k: v * uptake for k, v in action.primary.metric_changes.items()}
    
    obs = env.step(env_action)
    
    # Counterfactuals
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
            "id": episode_id
        },
        "counterfactuals": cf_data,
        "prediction": {
            "summary": DEMO_PREDICTOR.get_prediction_summary(),
            "risk_score": DEMO_PREDICTOR.get_risk_score()
        }
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
            overall_effectiveness=int(data.get('score')),
            domains_improved=data.get('improved', []),
            domains_worsened=data.get('worsened', []),
            unexpected_effects=data.get('notes', ""),
            resolution_time_hours=float(data.get('time'))
        )
        MEMORY.store_feedback(feedback)
        return jsonify({"status": "success", "message": f"Feedback stored for episode {feedback.episode_id}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    LONG_DEMO.pre_seed_arjun()
    app.run(host='0.0.0.0', port=7860, debug=True)
