# Episodic Memory

**Source file:** `agent/memory.py`

---

## Overview

`LifeStackMemory` gives the agent access to a history of past decisions and their outcomes. On each new conflict, it retrieves the most relevant past high-reward decisions and injects them as few-shot context into the prompt. This is not RAG over a knowledge base — it's a replay buffer that shapes the agent's prior before the model even generates a token.

The 116.81% reward improvement reported in `data/before_after_comparison.json` compares the memory-augmented agent against the same model without memory context. It is not a trained-vs-untrained comparison.

---

## Storage

Three ChromaDB collections, all in `./lifestack_memory/` by default:

| Collection | Content |
|------------|---------|
| `decisions` | Per-step action records: conflict title, action type, domain, reward, reasoning |
| `trajectories` | Per-episode summaries: task ID, route taken, total reward, milestone hits |
| `feedback` | Human outcome feedback: episode ID, effectiveness rating, improved/worsened domains |

```python
memory = LifeStackMemory(path="./lifestack_memory")
# Persistent ChromaDB; falls back to in-memory client on path failure
```

On first initialization, if the `decisions` collection is empty, `LifeStackMemory` auto-hydrates from `data/preseeded_memory*.json` (3 partitioned files). This ensures the agent has something useful to retrieve on a fresh deployment without requiring prior training runs.

---

## Embedding

`LifeStackMemory` uses `SentenceTransformer('all-MiniLM-L6-v2')` for 384-dim embeddings. If the model isn't available locally (`local_files_only=True`), it falls back to a deterministic hash-based embedding:

```python
# Hash fallback: adler32 hash of each token, bucketed into 384 dimensions
for token in text.lower().split():
    idx = zlib.adler32(token.encode()) % len(buckets)
    buckets[idx] += 1.0
```

The hash fallback preserves semantic retrieval quality well enough for the lexically consistent LifeStack vocabulary (conflict titles, domain names, action types appear repeatedly).

---

## Storing decisions

```python
memory.store_decision(
    conflict_title="Friday 6PM",
    action_type="communicate",
    target_domain="relationships",
    reward=0.72,
    metrics_snapshot={"relationships.romantic": 55, "mental_wellbeing.stress_level": 80},
    reasoning="A quick call prevents relationship erosion during high-stress periods."
)
```

The stored text is `f"{conflict_title} Action: {action_type} Domain: {target_domain} Reward: {reward:.2f} {reasoning[:100]}"`. This text is embedded and stored with the full metadata dict. Only the embedding is used at retrieval time.

---

## Retrieving similar decisions

```python
similar = memory.retrieve_similar(
    conflict_title="The Perfect Storm",
    current_metrics={"career.workload": 90, "mental_wellbeing.stress_level": 85},
    n=3
)
```

Query construction: embeds `f"{conflict_title} <top_3_most_stressed_metrics>"` — the most stressed metrics are the 3 with lowest values after sorting `current_metrics.items()`. This grounds the query in the agent's current situation rather than just the conflict name.

Results are filtered to `reward >= 0.05` before returning. The function retrieves `n*2` candidates and selects the top `n` by cosine similarity after filtering.

Return format:
```python
[{
    "action_type": "communicate",
    "target_domain": "relationships",
    "reward": 0.72,
    "reasoning": "...",
    "similarity_score": 0.87,
    ...
}]
```

---

## Few-shot prompt injection

```python
few_shot = memory.build_few_shot_prompt("Friday 6PM", current_metrics)
# Output:
# --- PAST EXPERIENCE & HUMAN VERIFICATION ---
# - Action Taken: [COMMUNICATE] on RELATIONSHIPS
#   Agent's Initial Reasoning: A quick call prevents relationship erosion...
#   HUMAN FEEDBACK: Rated 8/10. Notes: Partner appreciated the transparency.
```

`build_few_shot_prompt()` calls `retrieve_similar()`, then for each retrieved decision checks whether its `episode_id` has stored feedback in the `feedback` collection. If feedback exists, it appends the effectiveness rating and unexpected effects as additional context. This is the mechanism that brings human feedback into the agent's prompt without any fine-tuning.

---

## Trajectory storage and retrieval

```python
memory.store_trajectory(
    task_id="flight_crisis_task_main",
    route_taken="rebook_premium",
    total_reward=2.5,
    trajectory_summary={"milestones_hit": ["m1"], "steps": 8}
)

similar_trajectories = memory.retrieve_similar_trajectories(
    task_domain="flight_crisis",
    current_world={"lounge_access": True, "flight_rebooked": False},
    n=3
)
```

Trajectories are stored in the separate `traj_collection`. Query construction uses `f"TaskDomain: {task_domain} <top_3_most_stressed_world_values>"`. These are surfaced to the agent in `LifeStackAgent.plan()` but are not currently injected into the main GRPO training prompt (the training prompt uses `decisions` only).

---

## Human feedback storage

```python
from core.feedback import OutcomeFeedback
from datetime import datetime

feedback = OutcomeFeedback(
    episode_id="ep_12345",
    overall_effectiveness=8,
    domains_improved=["relationships", "mental_wellbeing"],
    domains_worsened=[],
    unexpected_effects="Partner called back and offered help with finances.",
    resolution_time_hours=2.5
)
memory.store_feedback(feedback)
```

Stored at doc ID `f"fb_{episode_id}"`. Retrieved by `reward_human_feedback_fn` during GRPO training via embedding similarity on the prompt text. This is what closes the loop between real-world outcomes and training signal — if a human reports that the agent's relationship actions worked well, future training batches for similar conflicts will reward those actions more.

---

## Memory stats

```python
stats = memory.get_stats()
# {"total_memories": 145, "average_reward": 0.623, "by_action_type": {"communicate": 38, ...}}
```

---

## Related files

- `agent/agent.py` — `LifeStackAgent` uses `LifeStackMemory.build_few_shot_prompt()`
- `core/feedback.py` — `OutcomeFeedback` dataclass, `compute_human_feedback_reward()`
- `scripts/train_trl.py` — `reward_human_feedback_fn` queries the feedback collection during training
- `data/preseeded_memory*.json` — initial hydration data (decisions collection)
