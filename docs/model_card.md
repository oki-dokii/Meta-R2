# Model Card

**HuggingFace repos:**
- v1: [jdsb06/lifestack-grpo](https://huggingface.co/jdsb06/lifestack-grpo)
- v3: [jdsb06/lifestack-grpo-v3](https://huggingface.co/jdsb06/lifestack-grpo-v3)
- v4: [jdsb06/lifestack-grpo-v4](https://huggingface.co/jdsb06/lifestack-grpo-v4) (best)

---

## What these models are

LoRA adapters trained on top of `Qwen/Qwen2.5-1.5B-Instruct`. Given a life conflict scenario (a JSON prompt describing a crisis, metric states, and available routes), the model outputs a structured JSON action:

```json
{
  "action_type": "communicate",
  "target_domain": "relationships",
  "metric_changes": {"relationships.romantic": 12.0, "mental_wellbeing.stress_level": -5.0},
  "resource_cost": {"time": 0.5, "money": 0, "energy": 10.0},
  "reasoning": "A reassuring call prevents relationship erosion while stress is high."
}
```

The adapter is not a general-purpose assistant. It is specialized for the LifeStack action space (`VALID_ACTION_TYPES = frozenset({"negotiate", "communicate", "delegate", "spend", "reschedule", "rest", "deprioritize", "execute", "prepare", "self_care"})`).

---

## Training details

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Library | HuggingFace TRL 0.15.1 |
| Efficiency | Unsloth 2026.4.8 (4-bit QLoRA) or plain HF+PEFT (bf16) |
| LoRA rank | r=16, lora_alpha=16 |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Environment | `LifeStackEnv` — 23 metrics across 6 life-metric domains + 2 crisis task domains, 32-edge dependency graph |
| Training hardware | A100 80GB |

**v1** was trained with `train_curriculum()` — 5 stages of single-step GRPO, 100 prompts per stage, with 10 reward functions starting from stage 2. Training log: [`train_run_v1.log`](../train_run_v1.log).

**v3** and **v4** were trained with `train_episodic_curriculum()` — episodic GRPO using `LifeStackGRPOTrainer` with JSON-boundary gradient masking, `episode_horizon=3`, 4 reward functions. v4 removed `reward_compact_fn` (zero variance throughout v3 training). Training log for v4: [`jdsb06/lifestack-grpo-v4/train_run_v4.log`](https://huggingface.co/jdsb06/lifestack-grpo-v4/blob/main/train_run_v4.log).

---

## Training evidence

The training logs are real runs. Key metrics from v4 (final episodic stage, sourced from `train_run_v4.log`):

| Metric | Value |
|--------|-------|
| Peak GRPO reward | ~0.856 |
| Format reward (peak) | +0.660 |
| Episode return (avg) | +0.140 |
| Training time | ~2 hrs (3 stages on A100 80GB) |
| Natural EOS terminations | First appeared in v4 (model self-terminating after JSON close) |

Plots (reward curve, loss curve, per-component breakdown) are in [`plots/`](../plots/) and mirrored at [`jdsb06/lifestack-grpo-v4/plots/`](https://huggingface.co/jdsb06/lifestack-grpo-v4/tree/main/plots).

The re-runnable Colab notebook is at [`notebooks/Colab_GRPO_Training.ipynb`](../notebooks/Colab_GRPO_Training.ipynb).

---

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("jdsb06/lifestack-grpo-v4")
model = PeftModel.from_pretrained(base, "jdsb06/lifestack-grpo-v4")
model.eval()
```

Or use `LifeStackAgent` directly — it handles model loading, fallback, and inference:

```python
from agent.agent import LifeStackAgent
agent = LifeStackAgent()   # loads jdsb06/lifestack-grpo-v4 by default
```

---

## Known limitations

The model is specialized for the LifeStack action space. Using it outside of LifeStack prompts will produce well-formed JSON that may be semantically nonsensical for other domains.

`reward_human_feedback_fn` requires a populated `LifeStackMemory` (ChromaDB). On fresh environments it abstains (returns 0.0). This means the human feedback signal was sparse during training on Colab/Kaggle runs where ChromaDB was not installed.

The holdout evaluation in `scripts/train_trl.py` uses the same `TaskGenerator` as training, which limits how confidently we can claim generalization. The training data is diverse across 8 domains and 5 difficulty levels, but it's procedurally generated from the same templates.

---

## Related files

- `agent/agent.py` — `LifeStackAgent` with lazy model loading
- `scripts/train_trl.py` — full training code
- `notebooks/Colab_GRPO_Training.ipynb` — Colab notebook
- `docs/HF_MODEL_CARD_V4.md` — full model card for HF Hub upload
- `docs/HF_MODEL_CARD_V1.md` — v1 model card for HF Hub upload
