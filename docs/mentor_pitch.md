# Mentor pitch — LifeStack (60–90 seconds)

## One-liner

**LifeStack** fine-tunes **Qwen2.5-1.5B** with **GRPO** so it outputs **valid JSON action plans** for messy, multi-domain life crises inside a **simulated** environment with **cascading consequences** — not generic chat advice.

---

## The problem

Chat models give plausible paragraphs. They do not reliably emit **structured actions** your app can execute, and they do not see **second-order effects** (money stress → sleep → health). We needed an **MDP-style** world plus a **training loop** that rewards **parseable** plans and **trajectory** outcomes.

---

## What we built

- **LifeStackEnv** (OpenEnv: `openenv.yaml`) — **eight domains**, sub-metrics, resource budgets, scripted disruptions, **dependency graph** propagation.
- **GRPO via TRL 0.15.1 + Unsloth** — group-relative updates without a critic; **LoRA** ~**1.18%** trainable params.
- **Episodic GRPO (key differentiator)** — horizon **3**: the model proposes a **short action sequence**; the env **steps** between actions; reward includes **discounted return**, not just the first JSON blob.

---

## Results (honest numbers)

- **Run 1 (broken):** truncated JSON + parse failures → reward about **−0.94**, `clipped_ratio≈1.0`, learning stalled.
- **Run 3 (fix):** discovered the model returned **JSON + English**; fixing extraction gave roughly **97% improvement** in mean reward vs Run 1 (first **positive** training mean ~**+0.02**).
- **Run 5 / episodic:** peak stage reward about **+0.73**; curriculum advanced past difficulty **1** for the first time.
- **v3:** curriculum **1→4** worked, but **`reward_compact_fn`** was **constant −0.5** with **zero variance** — it **flattened** the logged objective.
- **v4 (current):** removed the dead head, **doubled** episode-return weight — **target** mean roughly **+0.6 to +0.8** while training; **eval** often stays **near zero** because the sim is strict.

We are **not** claiming human-level life coaching — we **are** claiming a **repeatable RLHF-ish** pipeline with a **real env** and a **documented** debugging story.

---

## The “one bug” story (memorable)

**The model was often right and the parser was wrong.** Valid JSON followed by *“This is a simple JSON object…”* made `json.loads` fail → format reward **−0.5** every time → **no gradient**. **Greedy** outer-brace extraction (and incremental `raw_decode`) fixed it. **Non-greedy** regex **breaks** nested `{}` in `metric_changes`.

---

## Artifacts

- **Code:** [github.com/oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)
- **Models:** [jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) (best single-step), [jdsb06/lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3), [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) (current episodic)

---

## Ask / feedback we want

- Does the **episodic GRPO** story land as the main research contribution?
- Where would you **tighten** the reward (fewer heads vs richer sim)?
- **Safety / product:** when is structured JSON **worse** than refusing to act?
