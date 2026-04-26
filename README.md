# LifeStack

**LifeStack** trains **Qwen2.5-1.5B-Instruct** with **GRPO** (Group Relative Policy Optimization) using **Hugging Face TRL 0.15.1** and **Unsloth** so the model can propose **structured JSON action plans** for everyday crises across **eight domains**.

**Repository:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)

**OpenEnv:** This project ships an `openenv.yaml` manifest at the repo root (`core.lifestack_env:LifeStackEnv`) for hackathon and tooling compatibility.

---

## Domains and output format

Domains include: **career**, **finances**, **relationships**, **physical_health**, **mental_wellbeing**, **time**, **transport_crisis**, and **code_merge_crisis**.

The policy is trained to emit JSON like:

```json
{
  "action_type": "negotiate|communicate|delegate|spend|reschedule|rest|deprioritize|execute",
  "target_domain": "career|finances|relationships|...",
  "metric_changes": {"domain.submetric": 0},
  "resource_cost": {"time": 0, "money": 0, "energy": 0},
  "reasoning": "brief explanation"
}
```

The **LifeStack** simulator applies actions and propagates consequences through a **dependency graph** (for example: job loss → stress → sleep → health).

---

## Hugging Face models

| Model | Role |
|--------|------|
| [jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) | Run 4 — strong **single-step** checkpoint |
| [jdsb06/lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3) | **Episodic** v3 — curriculum advanced every stage; mean reward held back by a dead-weight term |
| [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) | **Episodic** v4 — **current / best target** (training in progress); `reward_compact_fn` removed, return signal up-weighted |

---

## Training progression (what we actually ran)

Honest summary: rewards stayed **near zero** for a long time; the big jump came from **parsing and length fixes**, then **episodic training** improved peak metrics. The model is **still learning**, not a solved product.

| Milestone | Setup | What happened | Status |
|-----------|--------|----------------|--------|
| **Run 1** | Single-step GRPO, 3×50 prompts, `max_completion_length=256` | JSON truncated → broken parses → `clipped_ratio≈1.0`, `frac_reward_zero_std≈0.75`, reward **−0.944** | Failed |
| **Run 2** | Single-step, 3×50, shorter completions (e.g. 128) | Reward **−0.266** (stage 1); eval mean **−0.41**, ~8/50 episodes at 0 | Minimal signal |
| **Run 3** | Single-step, 3×50 + **greedy JSON extraction** + stage-1 format warm-up | Root cause: valid JSON plus **trailing explanation** broke `json.loads` → constant **−0.5**; **greedy** `re.search(r'\{.*\}', text, re.DOTALL)` (not non-greedy) fixes nested objects; first **positive** mean reward **+0.023**, `frac_zero_std≈0.05`; eval **−0.010**, ~20/50 at 0 | **~97% gain vs Run 1** |
| **Run 4** | Single-step, 5×100 + `reward_clean_eos_fn` | Eval reward **−0.100**, ~45/50 episodes hitting 0 — consistent but **plateau** | Good single-step baseline |
| **Run 5** | Episodic GRPO, horizon **3**, 2 episode stages + new return / route rewards | Best **single-step-style** signal **+0.734**, `frac_zero_std=0.00`; curriculum **1→2** | Breakthrough |
| **v3** | Episodic, **3** episode stages, difficulty **1→2→3→4**; weights format **1.0**, eos **1.5**, plausibility **0.75**, return **1.0**, plus **`reward_compact_fn`** | Curriculum worked, but **`reward_compact_fn` stuck at −0.5** (zero variance) and dragged the logged mean (**−0.694 to −0.771**) | Curriculum ✅, reward scale ❌ |
| **v4** | Episodic, 3 stages, **60** episodes/stage, horizon **3**; **removed** `reward_compact_fn`; weights **format 1.0 \| eos 0.5 \| plausibility 0.5 \| return 2.0** | **In training**; expect roughly **+0.6 to +0.8** mean reward if the return signal dominates as in analysis | **Current** |

---

## Episodic GRPO (why it exists)

- **Single-step** training scores one completion per prompt.
- **Episodic** training asks for a short **sequence** of actions (horizon **3**). The environment **re-simulates** after each step; the learner is rewarded for **discounted return** and terminal outcomes, not just the first JSON blob.

This matches how “one bad Monday decision” compounds across domains in the dependency graph.

---

## Reward stack (v4 — episodic main training)

| Component | Weight | Typical range (per function) |
|-----------|--------|------------------------------|
| `reward_episode_format_fn` | 1.0 | about **−0.5 … +1.0** (schema / keys / enums) |
| `reward_clean_eos_fn` | 0.5 | stops cleanly after JSON (penalizes long trailing text) |
| `reward_episode_plausibility_fn` | 0.5 | penalizes “free lunch” metric jumps |
| `reward_episode_return_fn` | 2.0 | **trajectory** return from the real env (primary positive signal) |

**Single-step warm-up** (early curriculum) combines `reward_format_fn`, `reward_clean_eos_fn`, and `reward_route_target_fn` with weights tuned for format-first learning.

**Removed in v4:** `reward_compact_fn` — empirically **constant −0.5**, **zero standard deviation**, so it contributed **no learning signal** and only depressed logged reward.

---

## Quick start: train v4 (T4 / Colab-style)

Pinned stack we used: **Unsloth 2026.4.8**, **Transformers 5.5.0**, **Torch 2.10+cu128**, and **`trl==0.15.1`** (pin avoids mergekit / optional import churn in newer TRL).

On Torch **2.10**, Unsloth’s compiled cache can crash (`ref_hidden_states=None`). Training sets:

```bash
export LIFESTACK_NO_UNSLOTH=1
```

Example **v4** command:

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py \
  --episode-train \
  --stages 3 \
  --episodes-per-stage 60 \
  --episode-horizon 3 \
  --episode-warmup-stages 1 \
  --prompts-per-stage 60 \
  --num-train-epochs 3 \
  --max-prompt-length 4096 \
  --output-dir ./lifestack_model_v4 \
  --push-to-hub \
  --hub-repo-id jdsb06/lifestack-grpo-v4
```

**Resume** after a disconnect:

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py --episode-train --resume \
  --output-dir ./lifestack_model_v4 \
  # ...same flags as above...
```

---

## Tech summary

| Item | Value |
|------|--------|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit` |
| Method | GRPO via TRL |
| LoRA | r=16, α=16; modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Trainable params | ~18.5M / ~1.56B (**~1.18%**) |
| Hardware | Tesla **T4 15GB** (typical) |

---

## Known bugs we fixed (engineering notes)

1. **Stray backslash** in docs/snippets: `GRPOTrainer\(` → `GRPOTrainer(`.
2. **Unsloth + Torch 2.10** compiled path → use **`LIFESTACK_NO_UNSLOTH=1`**.
3. **TRL 0.15.1 + Transformers**: `_get_train_sampler` mismatch → **monkey-patch** with `inspect.signature` guard (see `scripts/train_trl.py`).
4. **`max_completion_length` too large** → truncated JSON; reduced for single-step (**128**), episodic uses **larger** budgets (script default **224** single-step; episodic auto-scales with horizon).
5. **Main bug — JSON + prose**: model returned valid JSON then English; naive `json.loads(full_text)` failed → **greedy brace extraction**; **non-greedy** `\{.*?\}` **breaks** on nested `metric_changes` objects.
6. **`reward_compact_fn`**: constant penalty, **zero variance** → **removed** in v4.
7. **TRL version drift**: optional **mergekit** imports in TRL **0.17+** → **pin `trl==0.15.1`** for this repo’s GRPO path.
8. **`weave` / optional deps**: local **shims** for mergekit / llm_blender / weave so imports succeed on Colab.

---

## Repository layout

```text
Meta-R2/
├── README.md              # This file — overview and quick start
├── openenv.yaml           # OpenEnv manifest (LifeStackEnv)
├── requirements.txt
├── core/                  # Environment, life state, tasks, rewards
├── agent/                 # Memory, conflict generation, agent loop
├── intake/                # SimPerson and intake helpers
├── scripts/               # train_trl.py, eval.py, smoke tests, …
├── docs/                  # All documentation — start at docs/README.md
├── notebooks/             # Jupyter notebooks (see notebooks/README.md)
├── data/                  # Small JSON fixtures (see .gitignore rules)
├── app.py / app_flask.py  # Demos
└── …
```

Local training checkpoints (`lifestack_model/`, `lifestack_model_v4/`, …) are listed in **`.gitignore`** so they do not clutter commits.

---

## Documentation

All detailed docs are in **[docs/](docs/)** — start at **[docs/README.md](docs/README.md)**.

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/training_guide.md](docs/training_guide.md) | Install → train → push |
| [docs/train_trl.md](docs/train_trl.md) | Training CLI reference |
| [docs/reward.md](docs/reward.md) | Reward functions |
| [docs/configuration.md](docs/configuration.md) | GRPOConfig |
| [docs/blog.md](docs/blog.md) | Hugging Face–style writeup |
| [docs/model_card.md](docs/model_card.md) | Model card |
| [docs/mentor_pitch.md](docs/mentor_pitch.md) | Short pitch |

---

## License and credits

See repository files for license and attribution. Built for a hackathon track that rewards **OpenEnv**, a **minimal TRL/Unsloth** training story, and a short **blog or video**.
