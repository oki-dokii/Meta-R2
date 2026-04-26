# `scripts/eval.py` — evaluation runner

Standalone **random-policy** baseline against **`LifeStackEnv`**. No trained model, no GPU, no API key.

Use it to:

- Verify the simulator after code changes  
- Establish a **reward floor** before GRPO runs  
- Quick CI / smoke checks  

---

## Usage

```bash
python scripts/eval.py
python scripts/eval.py --episodes 20 --domain flight_crisis
python scripts/eval.py --episodes 5 --verbose
```

Run from the **repo root** so `core` imports resolve.

---

## CLI

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | `10` | Number of episodes |
| `--domain` | `None` | Optional filter for `TaskGenerator` / task domain (e.g. `flight_crisis`, `code_merge_crisis`, or `transport_crisis` if wired) |
| `--verbose` | off | Per-step action, reward, `done` |

---

## Output

- Per-episode table (mean reward, steps, domain)  
- Aggregate mean / std across episodes  

Interpret **trained** models with `scripts/train_trl.py --full-episode` or app demos — `eval.py` is intentionally **random**.

---

## Relation to GRPO training

| Tool | Policy |
|------|--------|
| `eval.py` | Uniform random actions |
| `train_trl.py` | GRPO-trained LLM completions |
| `train_trl.py --full-episode` | Roll out **multi-step** episodes with a **saved** checkpoint |

---

## See also

- [train_trl.md](train_trl.md)  
- [lifestack_env.md](lifestack_env.md)  
