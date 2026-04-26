# LifeStack — implementation summary (final)

This document replaces the older sprint checklist with a **concise record** of what **LifeStack** is today: **environment**, **training**, **rewards**, **known bugs**, and **where the code lives**.

**Repo:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)

---

## Product definition

- **Policy:** Qwen2.5-1.5B-Instruct (4-bit Unsloth bundle) + **LoRA** GRPO fine-tuning.
- **Output:** structured JSON actions (`action_type`, `target_domain`, `metric_changes`, `resource_cost`, `reasoning`) or episodic **`{"actions": [...]}`** completions.
- **World:** **LifeStackEnv** — eight domains, metrics, resources, tasks, **DependencyGraph** cascades (e.g. job loss → stress → sleep → health).

---

## Training timeline (engineering)

| Phase | Outcome |
|-------|---------|
| Run 1 | Truncated JSON (`max_completion_length` too large) → parse failures → **−0.944** |
| Run 2 | Shorter completions → **−0.266** (minimal learning) |
| Run 3 | **JSON + trailing prose** fix (greedy `\{.*\}` / `raw_decode`) → first **positive** mean ~**+0.023** (~**97%** vs Run 1) |
| Run 4 | Longer single-step run → **plateau** (~**−0.1** eval) |
| Run 5 | **Episodic** GRPO, horizon 3 → best ~**+0.734**, curriculum **1→2** |
| v3 | Episodic curriculum **1→4**; **`reward_compact_fn`** constant → bad logged mean |
| v4 | **Removed** compact head; weights **1 / 0.5 / 0.5 / 2** on format, EOS, plausibility, return — **in training** |

**Hub:** `jdsb06/lifestack-grpo`, `jdsb06/lifestack-grpo-v3`, `jdsb06/lifestack-grpo-v4`.

---

## Code map

| Area | Path |
|------|------|
| Environment | `core/lifestack_env.py`, `openenv.yaml` |
| Metrics / state | `core/life_state.py`, `core/metric_schema.py` |
| Tasks / routes | `core/task.py`, `agent/conflict_generator.py` |
| Rewards (shared) | `core/reward.py` |
| GRPO trainer + heads | `scripts/train_trl.py` (`LifeStackGRPOTrainer`, curriculum loops) |
| Eval / plotting | `scripts/train_trl.py` (`evaluate_and_plot`), `scripts/eval.py` |
| Memory (optional) | `agent/memory.py`, `core/feedback.py` |
| Demos | `app.py`, `app_flask.py` |

---

## Technical stack (reference)

- **TRL 0.15.1** (pinned; avoids mergekit import issues in newer TRL)
- **Unsloth** optional via **`LIFESTACK_NO_UNSLOTH=1`** when Torch/kernels break
- **Shims** in `train_trl.py` for `mergekit`, `llm_blender`, `weave` imports

---

## Bugs fixed (condensed)

1. Documentation typo `GRPOTrainer\(`.
2. Unsloth compiled cache / **Torch 2.10** → `LIFESTACK_NO_UNSLOTH=1`.
3. TRL / Transformers **sampler** mismatch → runtime patch.
4. **Completion length** vs truncated JSON.
5. **Naive JSON parse** on completions with trailing natural language.
6. **Non-greedy regex** breaking nested JSON.
7. **`reward_compact_fn`** zero-variance penalty → removed v4.
8. Optional dependency **import** failures → shims + TRL pin.

---

## Documentation

- [README.md](../README.md) — repository overview and quick start  
- [blog.md](blog.md) — narrative post  
- [model_card.md](model_card.md) — Hugging Face model card  
- [README.md](README.md) (this folder) — full documentation index  

---

## Hackathon alignment

- **OpenEnv** manifest present (`openenv.yaml`).
- **Minimal training path:** `scripts/train_trl.py` with TRL/Unsloth.
- **Story + models:** blog draft + three Hub repos.

---

*Last updated: 2026-04-26.*
