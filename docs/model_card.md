---
language:
- en
tags:
- grpo
- trl
- unsloth
- reinforcement-learning
- life-planning
- structured-output
license: other  # set to match the repository LICENSE when you publish
---

# Model Card: LifeStack GRPO (Qwen2.5-1.5B-Instruct)

## Summary

**LifeStack** fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit via Unsloth) with **GRPO** (Group Relative Policy Optimization) using **Hugging Face TRL 0.15.1**. The policy generates **JSON action plans** for multi-domain life conflicts inside the **LifeStackEnv** simulator (OpenEnv manifest: `openenv.yaml`).

**Checkpoints:**

| Hub ID | Description |
|--------|-------------|
| `jdsb06/lifestack-grpo` | Run 4 — best **single-step** curriculum checkpoint |
| `jdsb06/lifestack-grpo-v3` | **Episodic** v3 — curriculum **1→4**; mean reward depressed by a constant penalty head |
| `jdsb06/lifestack-grpo-v4` | **Episodic** v4 — **current** training; removed dead-weight reward, up-weighted trajectory return |

**Code:** [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)

---

## Intended use

- **Research / demo:** structured decision support in a **simulated** environment.
- **Not** for medical, legal, or financial advice in the real world.
- Outputs must be **validated** by application logic; treat the model as a **proposal generator**.

---

## Model details

| Field | Value |
|-------|--------|
| Base | `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit` |
| Fine-tuning | GRPO (`trl==0.15.1`), LoRA r=16, α=16 |
| Target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Trainable parameters | ~18.5M / ~1.56B total (~1.18%) |
| Domains | career, finances, relationships, physical_health, mental_wellbeing, time, transport_crisis, code_merge_crisis |

---

## Training procedure

1. **Single-step curriculum** (early runs): format warm-up, then richer reward stack (task success, milestones, long-horizon rollout reward, etc.).
2. **Episodic curriculum** (v3+): horizon **3** actions per episode; prompts ask for compact JSON with an **`actions`** list; rewards combine format, EOS cleanliness, plausibility, and **discounted environment return**.
3. **Difficulty curriculum:** stages advance when logged mean reward ≥ **0** (see `curriculum_state.json` in training output).

**Environment:** set `LIFESTACK_NO_UNSLOTH=1` when Unsloth’s compiled cache conflicts with **Torch 2.10** (`ref_hidden_states=None`).

**Hardware reference:** NVIDIA **T4 15GB**; training time depends on `--stages`, `--episodes-per-stage`, and `--num-train-epochs`.

---

## Reward functions (v4 episodic)

| Function | Weight | Role |
|----------|--------|------|
| `reward_episode_format_fn` | 1.0 | Valid JSON schema, required keys, allowed enums |
| `reward_clean_eos_fn` | 0.5 | Penalizes text after the closing `}` |
| `reward_episode_plausibility_fn` | 0.5 | Penalizes implausible metric jumps vs resource cost |
| `reward_episode_return_fn` | 2.0 | Discounted sum of env step rewards + terminal shaping |

**Removed in v4:** `reward_compact_fn` (empirically constant, zero variance — no gradient).

---

## Example input / output

**User / system prompt (abridged):** crisis description, current metrics, valid routes — see `scripts/train_trl.py` prompt builders.

**Target completion (single-step shape):**

```json
{
  "action_type": "communicate",
  "target_domain": "relationships",
  "metric_changes": {"relationships.partner_trust": 5},
  "resource_cost": {"time": 2, "money": 0, "energy": 10},
  "reasoning": "Short calm conversation before the deadline reduces escalation."
}
```

**Episodic shape:** JSON with an `"actions"` array of objects in the same schema.

---

## Evaluation

Reported **mean reward** during development ranged from about **−0.94** (broken baseline) to **+0.73** (best episodic stage) depending on run and metric mix. **Evaluation means often stayed near zero** even when training improved — this is a **hard** simulation with multiple penalty terms, not a toy sentiment task.

Metrics to monitor in logs:

- **`reward` / `train/reward`:** scalar GRPO objective (weighted sum of heads).
- **`clipped_ratio`:** near 1.0 can indicate policy updates hitting PPO/GRPO limits or “talkative” completions wasting tokens.
- **`frac_reward_zero_std`:** fraction of reward components with **zero** standard deviation across the group — high values mean weak relative preferences.

---

## Limitations

- **Simulation only:** metrics and cascades are **hand-crafted**, not real life.
- **Small model:** reasoning depth is limited; long horizons need careful prompt and length budgets.
- **JSON brittleness:** without robust parsing, trailing prose breaks naive `json.loads` — production systems must use **safe extraction** (greedy outer braces or incremental `JSONDecoder.raw_decode`).
- **Reward design sensitivity:** a constant penalty term with **no variance** can **flatten** the logged reward without teaching anything (the v3 → v4 lesson).

---

## Ethical considerations

Do not deploy for high-stakes decisions without human oversight. The model may emit **plausible-looking** JSON that is **unfair, incomplete, or unsafe** if applied literally.

---

## Citation / links

- Repository: [https://github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)
- Models: [lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) · [lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3) · [lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4)
