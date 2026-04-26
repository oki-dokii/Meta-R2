# `agent/memory.py` — LifeStackMemory

ChromaDB-backed storage for **trajectories** and **human outcome feedback**. Used by the demo apps and optionally by **`reward_human_feedback_fn`** in `scripts/train_trl.py` (abstains with **0.0** if Chroma or the DB is unavailable — training never hard-fails on memory).

---

## Collections

| Collection | Contents |
|------------|----------|
| Trajectories | Successful episode decisions — action, reward, reasoning snippets |
| `feedback_collection` | Structured **OutcomeFeedback** from user follow-ups (domains improved/worsened, effectiveness) |

Only high-quality trajectories are persisted (module enforces a **reward threshold** to limit noise — see source for the current cutoff).

---

## API

### Construction

```python
from agent.memory import LifeStackMemory

memory = LifeStackMemory(silent=True)
memory = LifeStackMemory(silent=True, path="./my_memory")
```

### `store_trajectory(...)`

Persists a completed episode summary for later RAG-style retrieval.

### `store_feedback(OutcomeFeedback)`

Stores human labels used by `core/feedback.py:compute_human_feedback_reward`.

### Embeddings

`LifeStackMemory` uses **sentence-transformers** (when installed) for embedding text. Training’s `reward_human_feedback_fn` queries by **embedding** to avoid reward hacking on raw completion text.

---

## Training integration

- If **chromadb** / models are missing, `reward_human_feedback_fn` returns **zeros** and prints a **one-time** warning.  
- This keeps **Colab/Kaggle** runs reproducible without seeding memory first.

---

## See also

- [reward.md](reward.md) — `reward_human_feedback_fn`  
- [conflict_generator.md](conflict_generator.md) — task context for stored episodes  
