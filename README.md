# LifeStack
### An RL Environment for Multi-Domain Life Conflict Resolution
**Built for Meta × HuggingFace PyTorch OpenEnv Hackathon — Grand Finale 2026**

---

LifeStack is the first OpenEnv-compatible environment that trains agents to resolve cascading real-life conflicts across 6 interconnected domains simultaneously, under finite resource constraints, with Pareto-optimal reward shaping.

---

## Quick Start

```bash
git clone https://github.com/oki-dokii/Meta-R2
cd Meta-R2
bash setup.sh
source .venv/bin/activate
python app.py          # Launch Gradio demo  →  http://127.0.0.1:7860
python scripts/train.py        # Run 50-episode curriculum training
```

> **Verify openenv installed:** `pip3 show openenv-core` — should show `Version: 0.2.3`  
> **Note:** For compatibility with older environments, LifeStack supports both `openenv.core` and `openenv.env` import paths.

---

## Environment

| Item | Detail |
|---|---|
| **Python** | 3.9+ |
| **RL Environment** | `openenv-core >= 0.2.3` — `LifeStackEnv` targets the OpenEnv-style `reset()` / `step()` / `render()` API |
| **LLM Backend** | Groq Cloud API (`llama-3.1-8b-instant`) via OpenAI-compatible client |
| **Vector Memory** | ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` |
| **Demo UI** | Gradio 4.x — 3-tab interface |
| **Training Notebook** | Google Colab T4 GPU (`LifeStack_Training.ipynb`) |

> **API Key:** Store your `GROQ_API_KEY` as an env variable or in a `.env` file in the project root. The agent reads it automatically on startup.

---

## Environment Overview

LifeStack models human life as a directed dependency graph spanning 6 domains — **career, finances, relationships, physical health, mental wellbeing, and time** — connected by 20 weighted edges that propagate impact across nodes using a dampened BFS traversal. 

1.  **OpenEnv Core**: Inherits from `openenv.core.Environment` with Pydantic-based action and observation schemas.
2.  **Cascade Engine**: A graph-based propagator using BFS traversal with distance-decay to model how a "work crisis" bleeds into relationships or health.
3.  **Rubric Reward**: Normalized `LifeStackRubric` computing rewards in the [0.0, 1.0] range with hard-floor penalty triggers. 

When a crisis disrupts one domain (e.g. a sudden workload spike), the cascade engine propagates secondary effects through connected nodes (stress rises, sleep degrades, motivation falls, career growth slows). The agent operates under hard resource constraints: time, money, and finite energy units that cannot be manufactured — only managed. Each episode runs for up to 5 steps, during which the agent must find a sequence of actions that resolves the crisis without triggering critical collapses (any metric hitting zero terminates the episode with a severe penalty).

---

## Reward Function

```
reward = (0.40 × outcome_score)
       + (0.25 × cascade_containment)
       + (0.20 × resource_efficiency)
       + (0.15 × relationship_preservation)
       + penalties
```

| Component | Description |
|---|---|
| `outcome_score` | Weighted improvement across all 23 sub-metrics relative to the state before the action |
| `cascade_containment` | Bonus for preventing negative cascade spread into domains not targeted by the action |
| `resource_efficiency` | Reward scaled by how much of the resource budget was preserved after the action |
| `relationship_preservation` | Additional weight on maintaining social and romantic metrics, which degrade silently |

**Penalties applied when:**
- Any metric drops below the critical floor (< 20) — `CRITICAL_FLOOR` penalty
- The agent takes an action with insufficient resources — `INSUFFICIENT_RESOURCES` penalty
- No meaningful action is taken (inaction on a high-severity conflict) — `INACTION` penalty
- Relationship metrics fall by more than 15 points in a single step — `RELATIONSHIP_COLLAPSE` penalty

## Deployment (OpenEnv Native)

LifeStack is now a fully qualified OpenEnv project. You can serve it as an environment service or manage it via the OpenEnv CLI.

**Launch Environment Service:**
```bash
python3 server.py      # Starts the environment server on port 8000
```
- **Web Interface:** `http://localhost:8000/web`
- **MCP Tool List:** `http://localhost:8000/mcp`
- **EnvClient Target:** `sync_client = SyncEnvClient("http://localhost:8000")`

**CLI Manifest:** See [openenv.yaml](file:///Users/sohambanerjee/meta/openenv.yaml) for integration details.

---

## Key Features

- **SimPerson — OCEAN Personality Model:** Each episode samples a person from a diverse personality pool. The Big Five trait scores directly modulate action uptake — how effectively an action's metric changes translate into real outcomes. An anxious introvert (high neuroticism, low extraversion) responds poorly to delegation; a highly agreeable person responds well to communication-based actions.

- **Conflict Generator — 15 Templates × 5 Difficulty Levels:** Conflicts are drawn from a structured template library spanning career, financial, relationship, health, and time crises. Templates support escalation mid-episode: a manageable workload crunch can become a compounded crisis by step 3, forcing the agent to adapt dynamically.

- **RAG Memory — ChromaDB Persistent Store:** Every high-quality decision (reward > threshold) is embedded using `sentence-transformers` and stored in ChromaDB. On each new episode, semantically similar past decisions are retrieved and injected into the agent's prompt as few-shot context. The agent genuinely improves across sessions — not from weight updates, but from retrieved experience.

- **Multimodal Action Space:** The agent outputs both a primary action (type, target domain, metric changes, resource cost) and an optional communication action (recipient, tone, message content). This models how real life conflict resolution requires both internal action and external communication simultaneously.

- **Pareto-Optimal Planning:** The reward structure makes single-metric exploitation impossible. An agent cannot dump all resources into one metric without paying a cascade containment penalty and a resource efficiency penalty simultaneously. It must find balanced resolutions.

---

## Results

| Phase | Episodes | Avg Reward |
|---|---|---|
| Early | 1–15 | 2.438 |
| Mid | 16–35 | 2.533 |
| Late | 36–50 | 2.443 |
| **Overall** | **1–50** | **2.478** |

Memory growth: **0 → 928 decisions** stored across 50 training episodes.  
Best single episode reward: **2.579** (Episode 7).  
Memory advantage: agents with retrieval-augmented memory outperform blind agents on compound difficulty-5 crises.

---

## Research Foundations

| Paper | Influence |
|---|---|
| Russell & Norvig — *Constraint Satisfaction* | Informed the resource budget system and hard constraint enforcement during validation |
| Roijers et al. (2013) — *Multi-Objective RL Survey* | Foundation for the multi-component reward function and Pareto-optimality framing |
| Wang et al. (2024) — *Pareto-Optimal Treatment Under Conflicting Outcomes* | Shaped the cascade containment and relationship preservation reward components |
| Mullainathan & Shafir (2013) — *Scarcity: Why Having Too Little Means So Much* | Grounded the resource-constraint design and the stress-amplification penalty system |

---

## OpenEnv Compliance

- Uses **OpenEnv v0.2.3** when installed (`pip install openenv-core`)
- `LifeStackEnv` exposes the standard `reset()` / `step()` / `render()` API and falls back to a local shim if OpenEnv is unavailable
- Training script: **`LifeStack_Training.ipynb`** — runs end-to-end on Colab T4 GPU, Unsloth/TRL compatible
- HuggingFace blog post: [BLOG.md](./BLOG.md)

---

## File Structure

| File | Description |
|---|---|
| `lifestack_env.py` | Core OpenEnv environment — `reset()`, `step()`, `render()` with cascade propagation |
| `life_state.py` | 6-domain, 23-sub-metric `LifeMetrics` dataclass + `DependencyGraph` with 20 edges |
| `reward.py` | 4-component reward function with Pareto weighting and hard penalties |
| `agent.py` | LLM-powered agent (Groq / llama-3.1-8b) with retry logic and JSON action parsing |
| `conflict_generator.py` | 15 conflict templates across 5 difficulty levels with mid-episode escalation |
| `action_space.py` | `AgentAction` / `PrimaryAction` / `CommunicationAction` dataclasses + `apply_action()` |
| `simperson.py` | OCEAN personality model with `respond_to_action()` uptake scoring and drift |
| `memory.py` | ChromaDB RAG memory — stores high-reward decisions, retrieves few-shot context |
| `run_episode.py` | Full episode orchestrator — conflict → agent → apply → reward → memory loop |
| `train.py` | 50-episode curriculum trainer with `training_log.json` and `reward_curve.png` output |
| `app.py` | 3-tab Gradio demo — Live Demo, Try Your Situation, Training Results |
| `LifeStack_Training.ipynb` | Google Colab notebook — full training pipeline, runs on free T4 GPU |

---

## Team

**Team of 3 — Scaler School of Technology, Bangalore**

---

*LifeStack: We built the gym. Now any model can train in it.*
