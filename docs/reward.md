# Reward functions — LifeStack GRPO

Rewards are implemented in **`scripts/train_trl.py`** (GRPO heads) and **`core/reward.py`** (format, plausibility, env rollouts). GRPO combines multiple scalar heads with **`config.reward_weights`**.

---

## JSON extraction (the main bug)

**Symptom:** the model prints valid JSON and then adds friendly explanation. A naive `json.loads(entire_completion)` **fails** → format rewards collapse to **−0.5** → **no useful gradient**.

**Fixes:**

1. **Incremental decode:** `JSONDecoder.raw_decode` from each `{` — see `_load_first_json_object` in `core/reward.py`.
2. **Greedy outer braces:** `re.search(r'\{.*\}', text, re.DOTALL)` matches the **outermost** object. **Do not** use non-greedy `\{.*?\}` for nested payloads — it stops too early inside `metric_changes`.

This change alone moved the project from **~−0.94** mean reward (broken Run 1) to **first positive** means (~**+0.023**, Run 3) — about **97%** improvement from one diagnosis.

---

## `frac_reward_zero_std` (why it matters)

GRPO is **group-relative**: within a batch of completions for the same prompt, it compares rewards. If a reward head returns **the same value for every sample** (standard deviation **0** across the group), that head provides **no relative ranking**.

**`frac_reward_zero_std`** (when logged) estimates how many heads are “flat.” High values → weak learning signal. Removing **`reward_compact_fn`** in v4 was driven by this: it sat at **−0.5** with **zero variance**, so it was **dead weight** and only made the scalar `reward` look worse.

---

## Single-step warm-up (curriculum stage 1)

Used in `train_curriculum` **stage == 1**:

| Function | Weight (stage 1) | Range / notes |
|----------|------------------|---------------|
| `reward_format_fn` | **1.0** | Uses `reward_format_compliance` — roughly **−1.0 … +1.0** (empty/refusal **−1**, bad parse **−0.5**, partial **0.2–0.5**, full **+1.0**) |
| `reward_clean_eos_fn` | **1.5** (warm-up) | Rewards stopping near end of JSON; trailing prose → penalties (see docstring in `train_trl.py`) |
| `reward_route_target_fn` | **1.0** | **0 … ~0.3** — alignment with route ids / execute bonus |

Later single-step stages add simulation-heavy heads (`reward_task_success_fn`, `reward_milestone_fn`, `reward_longterm_fn`, etc.) with their own scales; see `train_curriculum` in `train_trl.py`.

---

## Episodic training (v4 — main stack)

| Function | Weight | Range / notes |
|----------|--------|---------------|
| `reward_episode_format_fn` | **1.0** | Mean format score over each action in the sequence |
| `reward_clean_eos_fn` | **0.5** | Same EOS logic as single-step |
| `reward_episode_plausibility_fn` | **0.5** | Mean `reward_plausibility_check` per action (**0 … −0.30** typical) |
| `reward_episode_return_fn` | **2.0** | **Trajectory** return from `LifeStackEnv` (**clamped** roughly to **[-1, 1]** in `get_episode_evaluation`) |

**Removed:** `reward_compact_fn` — in v3 it was effectively **constant −0.5**, **std = 0**, so it did not rank completions and only depressed the logged mean.

---

## `reward_compact_fn` (legacy — why removed)

Designed to penalize long trailing text after JSON. In practice (v3 episodic runs) it **always** returned **−0.5** with **no spread** across GRPO groups → **zero contribution to the relative policy gradient**. v4 drops it and **increases** `reward_episode_return_fn` weight to **2.0**.

---

## Reward weight progression (high level)

| Phase | Stack |
|-------|--------|
| Early single-step | Format-only emphasis; length bugs dominated |
| Run 4 single-step | Added / emphasized **clean EOS** to fight “JSON + essay” |
| Run 5 episodic | Introduced **episode return** + route-style shaping |
| v3 episodic | Strong curriculum; **compact** head poisoned metrics |
| v4 episodic | **Compact out**; weights **1.0 / 0.5 / 0.5 / 2.0** |

---

## Plausibility (`reward_plausibility_check`)

Penalizes **large `metric_changes` with zero `resource_cost`** (normalized time/money/energy). Prevents “free lunch” plans.

---

## Episode return (`get_episode_evaluation`)

1. Parse `actions` from model JSON.  
2. `LifeStackEnv.reset` from prompt metadata.  
3. For each action up to **horizon**: `env.step`, accumulate **γ**-discounted `obs.reward`.  
4. Add terminal success / failure shaping.

This is the **main signal** that reflects **multi-step** consequences.

---

## See also

- [train_trl.md](train_trl.md) — CLI and trainer flow  
- [lifestack_env.md](lifestack_env.md) — what `obs.reward` means  
- [configuration.md](configuration.md) — `GRPOConfig` and weights  
