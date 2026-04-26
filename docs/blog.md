# LifeStack: Teaching an LLM to Manage Daily Life Crises with GRPO

*A short post for Hugging Face — repo: [oki-dokii/Meta-R2](https://github.com/oki-dokii/Meta-R2)*

---

## 1. What LifeStack is

LifeStack is a small RL-style fine-tuning project. We take **Qwen2.5-1.5B-Instruct**, run **GRPO** through **Hugging Face TRL 0.15.1** and **Unsloth**, and ask the model to output **one JSON object per decision** — action type, target domain, metric deltas, resource cost, and a short reason. The goal is not poetry; it is **reliable structure** plus **plausible** life management under pressure.

## 2. The environment — eight domains, cascading consequences

Behind the trainer sits **LifeStackEnv**: eight domains (**career, finances, relationships, physical_health, mental_wellbeing, time, transport_crisis, code_merge_crisis**), sub-metrics, budgets for time/money/energy, scripted crises, and a **dependency graph** so shocks propagate. A job loss can raise stress, wreck sleep, then hit health — the model is scored on whether its JSON action **makes sense** and how the **simulated trajectory** evolves.

## 3. Training journey — what broke, what fixed

We did not “sweep hyperparameters” into a miracle. Most of the graph is **honest failure** plus a few sharp fixes.

| Milestone | Lesson |
|-----------|--------|
| Run 1 | **Too many tokens** → JSON **truncated** → parse failures → GRPO **clipped_ratio≈1.0**, reward **−0.944**. |
| Run 2 | Shorter completions helped (**−0.266**) but learning was still thin. |
| Run 3 | **The real bug:** the model often returned **valid JSON and then English**. `json.loads` on the **whole string** failed → format reward **−0.5** almost always. **Greedy** brace extraction with `re.search(r'\{.*\}', text, re.DOTALL)` fixed it (non-greedy `\{.*?\}` **fails** on nested objects like `"metric_changes": {}`). First **positive** mean **+0.023** — about **97% better** than Run 1. |
| Run 4 | Longer single-step run; **plateau** near zero on eval. |
| Run 5 | **Episodic GRPO** (horizon 3) + return shaping → best single-stage mean **+0.734**, curriculum finally moved past difficulty 1. |
| v3 | Full episodic curriculum **1→2→3→4**, but **`reward_compact_fn`** sat at **−0.5** with **zero variance** — pure drag. Logged means **−0.69 to −0.77** despite curriculum working. |
| v4 | **Dropped** `reward_compact_fn`, **doubled** weight on **episode return** (the only consistently informative signal). **Training now** toward **[jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4)**. |

## 4. The main bug — why JSON extraction mattered more than “more epochs”

Instruction-tuned models **like to explain themselves**. That habit is lethal if your reward code assumes the completion is **only** JSON. The fix was not “train harder”; it was **parse like a production system** — isolate the first JSON object (we use greedy outer braces for the narrative case above; the trainer also uses incremental `JSONDecoder.raw_decode` in places). Once rewards reflected real structure, GRPO had a gradient worth following.

## 5. Episodic GRPO — planning three moves ahead

Single-step training scores **one** answer. **Episodic** training asks for a **short sequence** of actions; the env **updates state between steps** and we score **discounted return**. That matches the design: crises **compound**. Episodic training is the main story after Run 5.

## 6. Results — reward tables (v4)

**Episodic (v4)**

| Function | Weight |
|----------|--------|
| Episode format | 1.0 |
| Clean EOS | 0.5 |
| Plausibility | 0.5 |
| Episode return | 2.0 |

**Warm-up single-step** uses format + EOS + route targeting before episodic fine-tuning.

We watch **frac_reward_zero_std** — when too many reward heads have **zero standard deviation**, GRPO gets weak group-relative signal. Removing the dead **compact** term was as important as any learning rate tweak.

## 7. What’s next

Ship **v4**, compare against **[jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo)** (best single-step) and **[jdsb06/lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3)**, and keep expectations **real**: rewards cluster near **zero**, not +1.0. The win is **reliable JSON**, **multi-step reasoning**, and an environment that **punishes magical thinking** about money and energy.

**Models:** [lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo) · [lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3) · [lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4)

---

*Word count: under 1000 — conversational, not a sales deck.*
