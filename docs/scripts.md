# Scripts reference

**Directory:** `scripts/`

---

## `scripts/train_trl.py` — main training

Full GRPO training — single-step curriculum and episodic curriculum. See [train_trl.md](train_trl.md) for complete reference.

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py            # full 5-stage curriculum
python scripts/train_trl.py --dry-run                         # pipeline check, CPU OK
python scripts/train_trl.py --resume                          # resume from checkpoint
python scripts/train_trl.py --episode-train                   # episodic mode
python scripts/train_trl.py --push-to-hub --hub-model-id ...  # upload after training
```

---

## `scripts/eval.py` — random baseline

Runs a uniform random policy for N episodes and reports mean/std reward. No trained model, no GPU, no API key needed. Use this to establish a reward floor before GRPO runs or to verify env correctness after code changes.

```bash
python scripts/eval.py                              # 10 episodes, all domains
python scripts/eval.py --episodes 20 --domain flight_crisis
python scripts/eval.py --episodes 5 --verbose       # per-step output
```

---

## `scripts/plot_training.py` — plot generation

Parses training logs (`train_run_v*.log`) and generates matplotlib plots: reward curve, loss curve, per-component reward breakdown, and 4-panel summary. Supports multiple log formats.

```bash
python scripts/plot_training.py --log train_run_v4.log --output-dir plots/
```

---

## `scripts/smoke_test.py`

Fast pipeline check: import validation + one `reset()` + one `step()`. No GPU, no downloads.

```bash
python scripts/smoke_test.py
```

Also called by `setup.sh` at the end of the install process.

---

## `scripts/upload_hf_model_cards.py`

Uploads model cards and training artifacts to HuggingFace model repositories:

- `docs/HF_MODEL_CARD_V4.md` → `jdsb06/lifestack-grpo-v4/README.md`
- `docs/HF_MODEL_CARD_V1.md` → `jdsb06/lifestack-grpo/README.md`
- `train_run_v1.log` → `jdsb06/lifestack-grpo/`
- Relevant plots → `jdsb06/lifestack-grpo-v4/plots/`

```bash
python scripts/upload_hf_model_cards.py   # requires HF_TOKEN or huggingface-cli login
```

---

## `scripts/run_episode.py`

Runs one full episode with the `LifeStackAgent` (GRPO model or Groq API fallback). Requires credentials configured.

```bash
python scripts/run_episode.py
python scripts/run_episode.py --difficulty 3 --verbose
```

---

## Related files

- `docs/train_trl.md` — full train_trl.py reference
- `docs/eval.md` — eval.py reference
- `docs/training_guide.md` — end-to-end training guide
