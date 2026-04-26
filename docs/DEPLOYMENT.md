# Deployment

**Source files:** `start.sh`, `server.py`, `app_flask.py`, `Dockerfile`

---

## Architecture

LifeStack runs as two services inside a single Docker container:

| Service | File | Port | Role |
|---------|------|------|------|
| Flask demo UI | `app_flask.py` | 7860 | HF health-check target, always foreground |
| OpenEnv server | `server.py` | 8000 | EnvClient endpoint, background process |

The Flask process must stay alive for HuggingFace Spaces to mark the Space as healthy. The OpenEnv server can crash without affecting the Space's health status.

---

## `start.sh`

The Docker `CMD`. Starts both services:

```bash
#!/bin/bash
set -e

# Start OpenEnv server in background on port 8000
python server.py 2>&1 | sed 's/^/[openenv] /' &
OPENENV_PID=$!
echo "[start.sh] OpenEnv server started (PID $OPENENV_PID) on port 8000"

# Wait 3s for OpenEnv to bind its port
sleep 3

# Start Flask UI in foreground on port 7860
exec python app_flask.py
```

The `sed` prefix (`[openenv]`) routes OpenEnv logs into HuggingFace's log viewer distinctly from Flask logs. `exec python app_flask.py` replaces the shell process with Flask, so HF's process supervisor sees Flask directly.

---

## `server.py` — crash-safe design

```python
def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        print(f"[openenv-server] uvicorn not installed: {exc}. Skipping OpenEnv server.")
        return      # <-- return, not raise SystemExit
    try:
        from openenv.core import create_app
    except Exception as exc:
        print(f"[openenv-server] openenv.core unavailable ({exc}). Skipping OpenEnv server.")
        return      # <-- return, not raise SystemExit
    ...
```

Every failure path in `server.py` calls `return` rather than `raise SystemExit`. If `openenv-core` isn't installed, if `create_app()` crashes, or if any import fails, `server.py` prints a message and exits the `main()` function quietly. The background process started by `start.sh` exits with code 0, and the Flask foreground process is unaffected.

This is what prevents the HuggingFace restart loop: when the Space first starts and dependencies are being resolved, the OpenEnv server can fail gracefully while Flask handles health checks.

---

## `Dockerfile`

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENENV_PORT=8000
ENV OPENENV_HOST=0.0.0.0
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh
EXPOSE 7860
EXPOSE 8000
CMD ["bash", "start.sh"]
```

Both ports are exposed. HuggingFace Spaces maps port 7860 automatically; port 8000 is available for direct `EnvClient` connections from external scripts.

---

## Environment variables

| Variable | Default | Set by | Purpose |
|----------|---------|--------|---------|
| `OPENENV_HOST` | `0.0.0.0` | Dockerfile / env | OpenEnv server bind address |
| `OPENENV_PORT` | `8000` | Dockerfile / env | OpenEnv server port |
| `OPENENV_MAX_SESSIONS` | `4` | env | Max concurrent `LifeStackEnv` instances |
| `PORT` | `7860` | HF Spaces | Flask listen port (HF sets this automatically) |
| `GROQ_API_KEY` | unset | HF Secrets | Enables Groq API in `LifeStackAgent` |
| `HF_TOKEN` | unset | HF Secrets | Loads private model checkpoints |
| `LIFESTACK_NO_UNSLOTH` | unset | env | Set `1` to skip Unsloth in training |

---

## Running locally

```bash
# Option 1: Docker (mirrors HF Spaces exactly)
docker build -t lifestack .
docker run -p 7860:7860 -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  lifestack

# Option 2: Direct (two terminals)
# Terminal 1 — OpenEnv server
python server.py

# Terminal 2 — Flask UI
python app_flask.py

# Both services with start.sh
bash start.sh
```

---

## Connecting via OpenEnv client

Once the server is running on port 8000:

```python
from openenv import EnvClient

client = EnvClient("http://localhost:8000")

# Reset the environment
obs = client.reset()
print(obs.metrics)          # 23 metric values
print(obs.resources)        # {"time": 20.0, "money": 500.0, "energy": 100.0}

# Step with an action
action = {
    "action_type": "communicate",
    "target_domain": "relationships",
    "metric_changes": {"relationships.romantic": 10.0},
    "resource_cost": {"time": 0.5, "energy": 10.0},
    "reasoning": "A reassuring call prevents relationship decay during the crisis.",
    "actions_taken": 1
}
obs = client.step(action)
print(obs.reward)
print(obs.metadata["breakdown"])
```

The OpenEnv server's web interface at `http://localhost:8000/web` provides a browser-based client for manual testing.

---

## HuggingFace Spaces specifics

HF Spaces mounts the GitHub repository via the `.github/workflows/deploy.yml` action. The deploy workflow strips binary files (`plots/*.png`, `train_run_v1.log`, etc.) from the Space snapshot to avoid HF's non-LFS binary rejection. These files are hosted separately on the HF model repositories (`jdsb06/lifestack-grpo-v4/plots/`).

The `README.md` front matter block at the root of the repo sets Space metadata (title, emoji, SDK, port, tags). This block must remain at the top of `README.md`; HF Spaces reads it at deploy time.

---

## Related files

- `start.sh` — dual-service startup
- `server.py` — crash-safe OpenEnv entry point
- `server/app.py` — full `create_app()` wrapper with console output
- `app_flask.py` — Flask UI on port 7860
- `Dockerfile` — container definition
- `.github/workflows/deploy.yml` — HF Spaces CI deploy
