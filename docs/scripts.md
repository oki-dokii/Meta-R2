# `scripts/` — auxiliary scripts

Scripts **not** fully covered by [eval.md](eval.md) or [train_trl.md](train_trl.md).

---

## `scripts/run_episode.py`

Runs one episode with the **LLM agent** (requires API credentials configured for your stack).

```bash
python scripts/run_episode.py
python scripts/run_episode.py --difficulty 3 --verbose
```

---

## `scripts/train.py`

**Legacy** policy-gradient loop (pre-TRL). Prefer **`scripts/train_trl.py`** for GRPO + curriculum + Hub push.

---

## `scripts/smoke_test.py`

Import check + single `reset` / `step`. No GPU.

```bash
python scripts/smoke_test.py
```

---

## `scripts/test_lifestack.py`

Edge-case tests (can run as a script or with pytest). Tests that need `OPENAI_API_KEY` skip when unset.

```bash
python scripts/test_lifestack.py
pytest scripts/test_lifestack.py -v
```

---

## `scripts/longitudinal_demo.py` (if present)

Longer rollout demos — see file docstring.

---

## Primary training entry

**GRPO / TRL / episodic training:** [train_trl.md](train_trl.md)

```bash
LIFESTACK_NO_UNSLOTH=1 python scripts/train_trl.py --episode-train --dry-run
```

---

## See also

- [training_guide.md](training_guide.md)  
- [README.md](README.md)  
