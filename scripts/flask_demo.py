from flask import Flask, request, jsonify
from core.lifestack_env import LifeStackEnv, LifeStackAction
import os

app = Flask(__name__)

# Initialize the LifeStack Engine
# Note: In a production Flask app, you'd handle session-based env storage
# For this demo, we'll use a globally shared instance
env = LifeStackEnv()

@app.route('/reset', methods=['POST'])
def reset_simulation():
    """Starts a new simulation episode."""
    # Reset to a fresh state (optionally with a custom task)
    obs = env.reset()
    return jsonify({
        "status": "success",
        "current_metrics": obs.metrics,
        "message": "LifeStack simulation reset successfully."
    })

@app.route('/step', methods=['POST'])
def take_action():
    """Executes a single step in the life simulation."""
    data = request.json
    
    # Construct the LifeStack Action from the request
    action = LifeStackAction(
        action_type=data.get('action_type', 'inaction'),
        target=data.get('target', 'leisure'),
        metric_changes=data.get('metric_changes', {}),
        resource_cost=data.get('resource_cost', {}),
        reasoning=data.get('reasoning', "Flask API Request")
    )
    
    # Execute the step in the engine
    obs = env.step(action)
    
    return jsonify({
        "metrics_after": obs.metrics,
        "reward": obs.reward,
        "done": obs.done,
        "metadata": obs.metadata
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current state of the engine."""
    return jsonify({
        "metrics": env.state.current_metrics.flatten(),
        "step_count": env.step_count
    })

if __name__ == '__main__':
    print("\n🚀 LifeStack Engine is now running via Flask!")
    print("Endpoints: /reset [POST], /step [POST], /status [GET]")
    app.run(host='0.0.0.0', port=5000)
