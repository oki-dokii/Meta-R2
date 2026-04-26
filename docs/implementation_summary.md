# LifeStack — implementation summary

**Repo:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)  
**HF Space:** [https://huggingface.co/spaces/jdsb06/meta-r2](https://huggingface.co/spaces/jdsb06/meta-r2)

---

## What LifeStack is

A multi-domain life-management RL environment built on OpenEnv 0.2.3. It models a human life as 23 interdependent metrics across 6 domains with a 32-edge directed dependency graph. Actions are structured JSON — the trained model allocates time, money, and energy across competing priorities without collapsing any metric to zero.

**Policy:** `Qwen2.5-1.5B-Instruct` (QLoRA / bf16) + LoRA adapters via GRPO fine-tuning.

**Output:** `{"action_type": ..., "target_domain": ..., "metric_changes": {...}, "resource_cost": {...}, "reasoning": "..."}` or episodic `{"actions": [...]}`.

---

## Training timeline

| Run | Key change | Outcome |
|-----|-----------|---------|
| v1 Run 1 | Initial — `max_completion_length` too large | Parse failures → mean **−0.944** |
| v1 Run 2 | Shorter completions | Minimal learning → **−0.266** |
| v1 Run 3 | JSON extraction fix (`raw_decode` + greedy regex) | First positive mean ≈ **+0.023** |
| v1 Run 4 | Longer single-step run | Plateau ≈ **−0.1** eval |
| v1 Run 5 | Episodic GRPO, horizon=3 | Best ≈ **+0.734**, curriculum 1→2 |
| v3 | Episodic curriculum 1→4 with `reward_compact_fn` | `reward_compact_fn` constant → bad logged mean |
| v4 | Removed compact head; weights `[1, 0.5, 0.5, 2]` on format/EOS/plausibility/return | Peak reward ≈ **+0.856** |

**HF Hub:** `jdsb06/lifestack-grpo` (v1), `jdsb06/lifestack-grpo-v3`, `jdsb06/lifestack-grpo-v4`.

---

## Code map

| File | What it does |
|------|-------------|
| `core/lifestack_env.py` | `LifeStackEnv` — reset, step, rollout, WorldEngine |
| `core/life_state.py` | `LifeMetrics` (23 values), `DependencyGraph` (32 edges), `ResourceBudget` |
| `core/reward.py` | `compute_task_reward`, `compute_reward`, 4 standalone scoring functions |
| `core/task.py` | `Task`, `Route`, `Milestone`, `ExoEvent`, `FlightCrisisTask`, `CodeMergeCrisisTask` |
| `core/verifier.py` | `LifeStackVerifier` — success/failure/milestone checks |
| `core/cascade_utils.py` | `animate_cascade()` — frame-by-frame visualization of propagation |
| `core/action_space.py` | `AgentAction`, `PrimaryAction`, `apply_action` |
| `agent/agent.py` | `LifeStackAgent` — GRPO model + Groq fallback, prompt building |
| `agent/memory.py` | `LifeStackMemory` — ChromaDB episodic memory, similarity retrieval |
| `agent/conflict_generator.py` | `TaskGenerator` (8 domains), `ConflictEvent` templates |
| `agent/conflict_predictor.py` | `ConflictPredictor` — pattern matching on episode history |
| `agent/counterfactuals.py` | What-if reasoning over metric snapshots |
| `intake/simperson.py` | `SimPerson` — Big Five personality, action uptake scaling |
| `scripts/train_trl.py` | Full GRPO training — `LifeStackGRPOTrainer`, 5-stage curriculum |
| `scripts/eval.py` | Random-policy baseline for reward floor measurement |
| `app_flask.py` | Flask demo UI — port 7860, 10 tabs, Chart.js, vis-network |
| `server.py` | Crash-safe OpenEnv server entry point — port 8000 |
| `start.sh` | Docker CMD — starts both services |

---

## Known limitations

`reward_timeout_check()` in `core/reward.py` has a known bug: it only fires when `step_count >= max_steps AND done == False`. Since `done` is `True` by the time the check runs, this function returns 0.0 in normal operation. The timeout penalty in `compute_task_reward()` is applied directly rather than through this function.

The holdout evaluation reuses the same `TaskGenerator` as training. This limits how confidently we can claim generalization to novel conflict types.

---

## Related docs

- [lifestack_env.md](lifestack_env.md) — environment reference
- [reward.md](reward.md) — all reward functions
- [train_trl.md](train_trl.md) — training reference
- [DEPLOYMENT.md](DEPLOYMENT.md) — deployment guide
