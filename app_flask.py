import os
import json
from flask import Flask, render_template_string, request, jsonify
from core.lifestack_env import LifeStackEnv, LifeStackAction
from core.life_state import LifeMetrics

app = Flask(__name__)
env = LifeStackEnv()

# Premium CSS for "WOW" factor
MODERN_CSS = """
:root {
    --primary: #6366f1;
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --accent: #22d3ee;
}
body { 
    background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; 
    margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh;
}
.container { width: 90%; max-width: 1000px; padding: 40px 0; }
.header { text-align: center; margin-bottom: 40px; }
.card { background: var(--card); border-radius: 16px; padding: 24px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); margin-bottom: 24px; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 16px; }
.metric-box { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px; text-align: center; }
.metric-val { font-size: 1.5rem; font-weight: bold; color: var(--accent); }
.btn { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1rem; transition: 0.2s; }
.btn:hover { opacity: 0.9; transform: translateY(-1px); }
.reasoning { font-style: italic; color: #94a3b8; margin-top: 10px; }
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LifeStack Engine</title>
    <style>{{css}}</style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🪐 LifeStack Premium</h1>
            <p>Meta OpenEnv Grand Finale 2026</p>
        </div>

        <div class="card">
            <h2>📊 Current Vital Metrics</h2>
            <div id="metrics" class="metrics-grid">
                {% for k, v in metrics.items() %}
                <div class="metric-box">
                    <div style="font-size: 0.8rem; color: #94a3b8">{{ k }}</div>
                    <div class="metric-val">{{ "%.1f"|format(v) }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="card">
            <h2>🛠️ Take Action</h2>
            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                <input id="action_type" placeholder="Action Type (e.g. leisure)" style="flex: 1; padding: 10px; border-radius: 8px;">
                <button class="btn" onclick="takeStep()">Execute Step</button>
            </div>
            <div id="log" class="card" style="background: rgba(0,0,0,0.2); min-height: 100px;">
                <p style="color: #94a3b8">Simulation Log...</p>
            </div>
        </div>
        
        <div style="text-align: center;">
            <button class="btn" style="background: #ef4444;" onclick="resetEnv()">Reset Simulation</button>
        </div>
    </div>

    <script>
        async function updateUI(data) {
            const metricsDiv = document.getElementById('metrics');
            metricsDiv.innerHTML = Object.entries(data.metrics).map(([k, v]) => `
                <div class="metric-box">
                    <div style="font-size: 0.8rem; color: #94a3b8">${k}</div>
                    <div class="metric-val">${v.toFixed(1)}</div>
                </div>
            `).join('');
            
            if (data.reward !== undefined) {
                const log = document.getElementById('log');
                log.innerHTML = `<p>Step Result: Reward <b>${data.reward.toFixed(2)}</b></p>`;
                if (data.metadata && data.metadata.breakdown) {
                   log.innerHTML += `<div class="reasoning">${data.reasoning || 'Action applied.'}</div>`;
                }
            }
        }

        async function takeStep() {
            const type = document.getElementById('action_type').value || 'inaction';
            const res = await fetch('/step', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action_type: type})
            });
            const data = await res.json();
            updateUI(data);
        }

        async function resetEnv() {
            const res = await fetch('/reset', {method: 'POST'});
            const data = await res.json();
            updateUI(data);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    metrics = env.state.current_metrics.flatten()
    return render_template_string(HTML_TEMPLATE, css=MODERN_CSS, metrics=metrics)

@app.route('/reset', methods=['POST'])
def reset():
    obs = env.reset()
    return jsonify({"metrics": obs.metrics})

@app.route('/step', methods=['POST'])
def step():
    data = request.json
    action = LifeStackAction(
        action_type=data.get('action_type', 'inaction'),
        target='career', # Default for demo
        metric_changes={}, # Simplified for demo
        resource_cost={},
        reasoning="Flask UI Interaction"
    )
    obs = env.step(action)
    return jsonify({
        "metrics": obs.metrics,
        "reward": obs.reward,
        "reasoning": action.reasoning
    })

if __name__ == '__main__':
    # HF Spaces use port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
