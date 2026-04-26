# `scripts/train_trl.py` — GRPO training (TRL + Unsloth)

Trains **Qwen2.5-1.5B-Instruct** for **LifeStack** using **`GRPOTrainer`** from **TRL 0.15.1** (pinned for reproducibility and to avoid optional-import breakage in TRL **≥0.17**).

**Source:** `scripts/train_trl.py`  
**Hub examples:** `jdsb06/lifestack-grpo`, `jdsb06/lifestack-grpo-v3`, `jdsb06/lifestack-grpo-v4`

---

## Environment variable: `LIFESTACK_NO_UNSLOTH`

When set to `1`, `true`, or `yes`, training **skips** `import unsloth` so **`trl.GRPOTrainer`** stays the stock HF trainer. Use this on **Torch 2.10** if Unsloth’s compiled path crashes (e.g. `ref_hidden_states=None`).

```bash
export LIFESTACK_NO_UNSLOTH=1
```

---

## CLI reference

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | One training step on a tiny batch (smoke test; CPU OK). |
| `--stages` | `5` | Number of curriculum **stages** (single-step **or** episodic, depending on mode). |
| `--prompts-per-stage` | `100` | Prompts per stage in **single-step** `train_curriculum`. Also used for **episode warm-up** stages when `--episode-train` is set. |
| `--output-dir` | `./lifestack_model` | Root directory for checkpoints and `curriculum_state.json`. |
| `--resume` | off | Resume from `curriculum_state.json` and latest `checkpoint-*` in the current stage directory. |
| `--start-stage` | none | Force 1-based stage index; ignores saved curriculum position unless combined with `--resume` behavior as implemented. |

### Episode / multi-step

| Flag | Default | Description |
|------|---------|-------------|
| `--episode-train` | off | Run **`train_episodic_curriculum`** instead of single-step curriculum. |
| `--episode-horizon` | `3` | Max actions per episode completion. |
| `--episodes-per-stage` | `40` | Episodes (prompts) per episodic stage. |
| `--episode-warmup-stages` | `1` | Number of **single-step** warm-up stages before episodic training (skipped when `--resume`). |

### Lengths and optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--max-prompt-length` | `2048` | Left-truncated prompt token budget for GRPO. |
| `--max-completion-length` | `224` | Max **new** tokens per completion in **single-step** training. |
| `--episodic-max-completion` | `0` | Override episodic completion budget; **`0` = auto** (`min(1024, max(512, 256 * horizon))`). |
| `--num-train-epochs` | `1` | Epochs **per stage** in episodic training (and used where wired in curriculum). Increase (e.g. **3**) for longer runs without more prompts. |

### Hugging Face Hub

| Flag | Default | Description |
|------|---------|-------------|
| `--push-to-hub` | off | Push `model` + `tokenizer` after training (or after `--full-episode` when applicable). |
| `--hub-repo-id` | `lifestack-grpo` | Target repo id (`username/name`). |

### Post-training

| Flag | Description |
|------|-------------|
| `--full-episode` | Roll out multi-step episodes with a saved model (evaluation path). |

---

## Modes

### 1) Single-step curriculum (default)

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py \
  --stages 5 \
  --prompts-per-stage 100 \
  --output-dir ./lifestack_model
```

Stage 1 uses **format + clean EOS + route target** rewards; later stages add simulation-heavy heads (task success, milestones, long-term rollout, etc.).

### 2) Episodic curriculum

Runs **`--episode-warmup-stages`** single-step stages (unless `--resume`), then episodic stages with weights:

**`[reward_episode_format_fn, reward_clean_eos_fn, reward_episode_plausibility_fn, reward_episode_return_fn]`**  
Weights: **`[1.0, 0.5, 0.5, 2.0]`** (v4; `reward_compact_fn` removed).

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

---

## Resume behavior

1. **`curriculum_state.json`** in `--output-dir` stores `completed_stage` and `next_difficulty`.
2. On `--resume`, training reads the file and continues at **`completed_stage + 1`** with saved difficulty.
3. Within a stage, **`find_latest_checkpoint(stage_dir)`** picks the newest `checkpoint-*` and passes `resume_from_checkpoint=` to `trainer.train`.

**Single-step stage dirs:** `{output_dir}/stage_{n}`  
**Episodic stage dirs:** `{output_dir}/episode_stage_{n}`

If you change `--stages` or output layout manually, verify paths before resuming.

---

## Implementation notes

- **Optional dependency shims:** before importing TRL, the script registers stub modules for **`mergekit`**, **`llm_blender`**, and **`weave`** so Colab/Kaggle installs do not fail on unused imports.
- **Custom trainer:** episodic training uses **`LifeStackGRPOTrainer`** for JSON-aware masking / alignment with reward boundaries (see source).
- **JSON parsing:** rewards prefer **`_load_first_json_object`** (`JSONDecoder.raw_decode`) in `core/reward.py`; UI code may use greedy `re.search(r'\{.*\}', text, re.DOTALL)` — **non-greedy** patterns break nested objects.

---

## Bugs fixed (reference)

| Issue | Mitigation |
|--------|------------|
| Docs typo `GRPOTrainer\(` | Use correct constructor name. |
| Unsloth compiled cache / Torch 2.10 | `LIFESTACK_NO_UNSLOTH=1` |
| TRL `_get_train_sampler` signature drift | Monkey-patch in `train_trl.py` |
| `max_completion_length` too large | Truncated JSON; use **128–224** single-step; episodic auto/override |
| Valid JSON + trailing prose → `json.loads` fail | Greedy brace extract or `raw_decode` |
| `reward_compact_fn` constant −0.5, std=0 | Removed from v4 stack |
| TRL 0.17+ mergekit import | Pin **`trl==0.15.1`** for this project path |
| `weave` / W&B tracing import | Stub `weave` and submodules |

---

## See also

- [reward.md](reward.md) — reward math and heads  
- [training_guide.md](training_guide.md) — end-to-end setup  
- [configuration.md](configuration.md) — `GRPOConfig` fields  
