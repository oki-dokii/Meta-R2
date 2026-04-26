# Training Guide — from zero to Hub

**Stack:** `Qwen2.5-1.5B-Instruct`, TRL 0.15.1, Unsloth 2026.4.8 (optional), PyTorch 2.10+

---

## Prerequisites

- Python 3.10+ (3.12 OK with compatible wheel builds)
- NVIDIA GPU with at least 15 GB VRAM (T4 works; A100 80GB recommended for bf16 HF+PEFT path)
- HuggingFace account if using `--push-to-hub`

Pinned stack for reproducibility:

```text
trl==0.15.1
transformers==5.5.0
torch==2.10+cu128
unsloth            # optional; skip with LIFESTACK_NO_UNSLOTH=1
```

---

## Setup

```bash
git clone https://github.com/oki-dokii/Meta-R2.git
cd Meta-R2
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For GPU PyTorch, install the CUDA build matching your driver before `pip install -r requirements.txt`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Log in to HuggingFace:

```bash
huggingface-cli login
```

---

## Running training

```bash
# Full 5-stage single-step curriculum (recommended on A100, ~2h)
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py

# With Unsloth 4-bit (T4 GPU, lower VRAM)
python scripts/train_trl.py

# Episodic training only (uses LifeStackGRPOTrainer, horizon=3)
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py --episode-train

# Dry run (CPU, validates pipeline, ~60s, no checkpoint)
python scripts/train_trl.py --dry-run

# Resume from checkpoint
python scripts/train_trl.py --resume

# Start from specific stage
python scripts/train_trl.py --start-stage 3

# Push final model to Hub
python scripts/train_trl.py --push-to-hub --hub-model-id jdsb06/lifestack-grpo
```

---

## Unsloth vs HF+PEFT

`LIFESTACK_NO_UNSLOTH=1` skips Unsloth entirely and uses plain `AutoModelForCausalLM` + `get_peft_model()`. This is recommended on A100 80GB because:

- Unsloth's 4-bit checkpoint bakes in `bnb_4bit_compute_dtype=float16`, which clashes with LoRA gradient precision in hard-to-diagnose ways
- The HF+PEFT path is bf16 end-to-end — no AMP surprises, no GradScaler
- 1.5B × 2 bytes ≈ 3 GB, comfortably fits alongside the optimizer state on 80 GB

On T4 (15 GB), the Unsloth path is the only practical option due to VRAM.

---

## Checkpoint layout

```
./lifestack_model/
├── curriculum_state.json     # {"completed_stage": 3, "next_difficulty": 2}
├── stage_1/
│   ├── checkpoint-5/         # mid-stage checkpoint
│   └── ...                   # final stage model
├── stage_2/
│   └── ...
└── stage_5/
    └── ...                   # final model here after train_curriculum()
```

`--resume` reads `curriculum_state.json` to find where to restart, then finds the latest `checkpoint-N` folder in that stage directory for the mid-stage resume.

---

## Loading a trained checkpoint

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./lifestack_model")
model = PeftModel.from_pretrained(base, "./lifestack_model")
model.eval()
```

For inference via Unsloth (faster generation):

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lifestack_model",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

---

## Common errors

**`GradScaler` crash with Unsloth:** Set `LIFESTACK_NO_UNSLOTH=1`. This happens when Unsloth's float16 4-bit kernel collides with gradient precision settings.

**`_get_train_sampler` TypeError:** Already patched in `train_trl.py` via `_patched_get_train_sampler`. Appears with TRL 0.15.1 + Transformers 4.56.2 — the patch makes the method signature flexible.

**`mergekit`/`llm_blender`/`weave` ImportError:** Shims are installed by `_install_trl_optional_dependency_shims()` before TRL imports. If you see these errors, ensure you're running from the repo root and the shim installation ran.

**Out of VRAM on T4:** Reduce `max_prompt_length` to 1024 and `max_completion_length` to 128. The episodic setting uses `min(1024, max(512, 256 * horizon))` by default; pass `--episodic-max-completion 256` to reduce.

---

## Related files

- `scripts/train_trl.py` — full training code
- `docs/train_trl.md` — detailed CLI reference
- `docs/configuration.md` — `GRPOConfig` parameters
- `notebooks/Colab_GRPO_Training.ipynb` — Colab notebook version
