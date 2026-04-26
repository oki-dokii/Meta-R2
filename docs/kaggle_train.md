# Kaggle / Colab Training Notes

These are notes specific to training on Kaggle notebooks or Google Colab where the environment differs from a dedicated A100 server.

---

## GPU options

| Platform | GPU | VRAM | Recommended path |
|----------|-----|------|-----------------|
| Colab Free | T4 | 15 GB | Unsloth 4-bit path (default) |
| Colab Pro | A100 40GB | 40 GB | `LIFESTACK_NO_UNSLOTH=1` + bf16 |
| Kaggle | P100 / T4 | 16 GB | Unsloth 4-bit path |

---

## Install on Colab/Kaggle

```bash
# Install the LifeStack stack
!pip install unsloth trl==0.15.1 transformers datasets accelerate peft chromadb sentence_transformers

# Clone the repo
!git clone https://github.com/oki-dokii/Meta-R2.git
%cd Meta-R2
```

Or open the ready-made notebook: [`notebooks/Colab_GRPO_Training.ipynb`](../notebooks/Colab_GRPO_Training.ipynb)

---

## Session length limits

Colab free sessions time out after ~90 minutes. LifeStack training uses `curriculum_state.json` + per-step checkpoints (`save_steps=5`) to support resume. When the session dies mid-stage:

```bash
# On next session
!python scripts/train_trl.py --resume
```

The `--resume` flag reads `curriculum_state.json` to determine which stage to restart, then calls `find_latest_checkpoint()` to find the last saved `checkpoint-N` folder.

---

## Memory pressure

If you hit OOM on T4:

```bash
# Reduce prompt and completion length
!python scripts/train_trl.py --max-prompt-length 1024 --max-completion-length 128
```

For episodic mode with horizon=3, set `--episodic-max-completion 384` (default is `min(1024, max(512, 256×horizon)) = 768`).

---

## Unsloth on Kaggle

Unsloth must be installed before importing TRL. The training script handles this with an early import at the top (`import unsloth` before any TRL import). If Unsloth fails for any reason, the script falls back to plain HF+PEFT automatically. Check that Unsloth's version is compatible with your CUDA build:

```bash
!pip show unsloth | grep Version
```

---

## After training

Push to Hub from Colab/Kaggle:

```bash
from huggingface_hub import login
login(token="your_token")  # or set HF_TOKEN env var

!python scripts/train_trl.py --push-to-hub --hub-model-id your_username/lifestack-grpo
```

Or push manually:

```bash
!huggingface-cli upload your_username/lifestack-grpo ./lifestack_model
```

---

## Related files

- `notebooks/Colab_GRPO_Training.ipynb` — complete Colab notebook
- `docs/training_guide.md` — full training guide
- `docs/train_trl.md` — `train_trl.py` CLI reference
