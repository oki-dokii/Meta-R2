# configuration.md — Configuration Reference

Environment variables, secrets, and server configuration for LifeStack.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For agent/training | API key for the LLM agent and GRPO reward function |
| `GROQ_API_KEY` | Optional | Alternative fast-inference backend |
| `GMAIL_CREDENTIALS_PATH` | Optional | Path to Gmail OAuth2 credentials JSON |

> **Never commit `.env`** — it is listed in `.gitignore`.

---

## `openenv.yaml`

Defines the OpenEnv service manifest for MCP / REST integration.

```yaml
name: lifestack
version: "1.1.0"
entry: server.py
port: 8000
```

Edit this file if you rename the server entry point or change the port.

---

## Gradio App

Configured in `app.py` `__main__` block:

```python
app.launch(
    share=False,
    server_port=7860,
    show_error=True,
)
```

Change `server_port` or set `share=True` for a public Gradio link.

---

## Docker

```bash
docker build -t lifestack:latest .
docker run -p 7860:7860 --env-file .env lifestack:latest
```

The `Dockerfile` installs `requirements.txt` and runs `python app.py`.

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-23 | Initial doc created |
