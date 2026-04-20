import chromadb
from sentence_transformers import SentenceTransformer
import json
import uuid
import os
from datetime import datetime
from collections import defaultdict


class LifeStackMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path='./lifestack_memory')
        self.collection = self.client.get_or_create_collection(name='decisions')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Memory system initialized")

    def store_decision(
        self,
        conflict_title: str,
        action_type: str,
        target_domain: str,
        reward: float,
        metrics_snapshot: dict,
        reasoning: str
    ) -> None:
        """Only stores high-reward decisions (reward >= 0.5) for good example learning."""
        if reward < 0.5:
            print(f"Skipped (low reward {reward:.2f}): {action_type} → {target_domain}")
            return

        text = f"{conflict_title} {action_type} {target_domain} {reasoning[:100]}"
        embedding = self.encoder.encode(text).tolist()

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "conflict_title": conflict_title,
                "action_type": action_type,
                "target_domain": target_domain,
                "reward": reward,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat()
            }]
        )
        print(f"Stored decision: {action_type} → {target_domain} (reward: {reward:.2f})")

    def retrieve_similar(self, conflict_title: str, current_metrics: dict, n: int = 3) -> list[dict]:
        """Retrieves the n most similar past high-reward decisions using semantic search."""
        if self.collection.count() == 0:
            return []

        # Build query from conflict title + 3 most stressed metrics (lowest values)
        sorted_metrics = sorted(current_metrics.items(), key=lambda x: x[1])
        top_stressed = " ".join(f"{k}:{v:.0f}" for k, v in sorted_metrics[:3])
        query_text = f"{conflict_title} {top_stressed}"

        query_embedding = self.encoder.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.collection.count())
        )

        output = []
        for i, meta in enumerate(results['metadatas'][0]):
            distance = results['distances'][0][i]
            similarity = round(1.0 / (1.0 + distance), 4)
            output.append({
                "action_type": meta["action_type"],
                "target_domain": meta["target_domain"],
                "reward": meta["reward"],
                "reasoning": meta["reasoning"],
                "similarity_score": similarity
            })

        return output

    def build_few_shot_prompt(self, conflict_title: str, current_metrics: dict) -> str:
        """Formats retrieved memories into a few-shot prompt block for the LLM."""
        memories = self.retrieve_similar(conflict_title, current_metrics)
        if not memories:
            return ""

        lines = ["Past successful decisions in similar situations:\n"]
        for m in memories:
            short_reason = m['reasoning'][:60]
            lines.append(
                f"  [{m['action_type']}] on [{m['target_domain']}] → reward {m['reward']:.2f} "
                f"(reasoning: {short_reason}...)"
            )

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Returns memory stats: total count, average reward, and action_type breakdown."""
        if self.collection.count() == 0:
            return {"total_memories": 0, "average_reward": 0.0, "by_action_type": {}}

        all_records = self.collection.get(include=["metadatas"])
        metadatas = all_records["metadatas"]

        total = len(metadatas)
        avg_reward = sum(m["reward"] for m in metadatas) / total

        by_type = defaultdict(int)
        for m in metadatas:
            by_type[m["action_type"]] += 1

        return {
            "total_memories": total,
            "average_reward": round(avg_reward, 3),
            "by_action_type": dict(by_type)
        }


def main():
    memory = LifeStackMemory()

    # --- Synthetic Decisions: mix of high and low reward ---
    synthetic = [
        {
            "conflict_title": "Friday 6PM",
            "action_type": "negotiate",
            "target_domain": "career",
            "reward": 0.72,
            "metrics_snapshot": {"career.workload": 100, "mental_wellbeing.stress_level": 95},
            "reasoning": "Negotiating the deadline directly reduced workload pressure quickly."
        },
        {
            "conflict_title": "Friday 6PM",
            "action_type": "rest",
            "target_domain": "mental_wellbeing",
            "reward": 0.61,
            "metrics_snapshot": {"mental_wellbeing.stress_level": 95, "physical_health.energy": 40},
            "reasoning": "A short rest during peak stress restored energy before tackling logistics."
        },
        {
            "conflict_title": "The Perfect Storm",
            "action_type": "communicate",
            "target_domain": "relationships",
            "reward": 0.58,
            "metrics_snapshot": {"relationships.romantic": 45, "mental_wellbeing.emotional_stability": 50},
            "reasoning": "A quick reassuring call prevented relationship collapse under crisis."
        },
        {
            "conflict_title": "The Perfect Storm",
            "action_type": "delegate",
            "target_domain": "career",
            "reward": 0.38,  # Below threshold — should NOT be stored
            "metrics_snapshot": {"career.workload": 90, "career.stability": 55},
            "reasoning": "Attempted to delegate but the neurotic profile made it ineffective."
        },
        {
            "conflict_title": "Health Scare",
            "action_type": "rest",
            "target_domain": "physical_health",
            "reward": 0.80,
            "metrics_snapshot": {"physical_health.energy": 20, "mental_wellbeing.stress_level": 90},
            "reasoning": "Aggressive rest protocol dramatically recovered energy and clarity."
        },
        {
            "conflict_title": "Check Engine Light",
            "action_type": "spend",
            "target_domain": "finances",
            "reward": 0.33,  # Below threshold — should NOT be stored
            "metrics_snapshot": {"finances.liquidity": 40, "time.commute_burden": 80},
            "reasoning": "Overspent on premium repair, draining liquidity buffer dangerously."
        },
    ]

    print("\n--- STORING SYNTHETIC DECISIONS ---")
    for d in synthetic:
        memory.store_decision(**d)

    # --- Retrieve similar decisions ---
    print("\n--- RETRIEVING SIMILAR DECISIONS ---")
    test_metrics = {
        "career.workload": 95,
        "mental_wellbeing.stress_level": 90,
        "finances.liquidity": 35,
        "physical_health.energy": 50,
        "relationships.romantic": 70
    }
    similar = memory.retrieve_similar("Friday 6PM", test_metrics, n=3)
    for s in similar:
        print(f"  [{s['action_type']}] → {s['target_domain']} | reward: {s['reward']:.2f} | similarity: {s['similarity_score']:.4f}")
        print(f"  Reasoning: {s['reasoning'][:80]}...")

    # --- Few-shot prompt ---
    print("\n--- FEW-SHOT PROMPT OUTPUT ---")
    prompt = memory.build_few_shot_prompt("Friday 6PM", test_metrics)
    print(prompt if prompt else "(No relevant memories found)")

    # --- Stats ---
    print("\n--- MEMORY STATS ---")
    stats = memory.get_stats()
    print(f"Total Memories : {stats['total_memories']}")
    print(f"Average Reward : {stats['average_reward']}")
    print(f"By Action Type : {stats['by_action_type']}")


if __name__ == "__main__":
    main()
