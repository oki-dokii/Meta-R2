import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import math
from datetime import datetime
from collections import defaultdict
from typing import Optional


class LifeStackMemory:
    def __init__(self, silent: bool = False, path: str = "./lifestack_memory"):
        self.silent = silent
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(name='decisions')
            self.traj_collection = self.client.get_or_create_collection(name='trajectories')
            self.feedback_collection = self.client.get_or_create_collection(name='feedback')
        except Exception as e:
            if not self.silent:
                print(f"⚠️ Memory persistence failed ({e}). Falling back to episodic (in-memory) mode.")
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name='decisions')
            self.traj_collection = self.client.get_or_create_collection(name='trajectories')
            self.feedback_collection = self.client.get_or_create_collection(name='feedback')
        
        self.encoder = self._load_encoder()
        if not self.silent:
            print("Memory system initialized")
        
        # Auto-hydrate if empty
        if self.collection.count() == 0:
            self._hydrate_from_preseeded()

    def _hydrate_from_preseeded(self):
        import json
        sources = ["./data/preseeded_memory.json", "./data/preseeded_memory_p1.json", "./data/preseeded_memory_p2.json"]
        
        if not self.silent:
            print(f"🧬 Empty memory detected. Hydrating from partitioned volumes...")
            
        total_decisions = 0
        for path in sources:
            if not os.path.exists(path):
                continue
            
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Hydrate decisions
                d = data.get("decisions", {})
                if d.get("ids"):
                    self.collection.add(
                        ids=d["ids"],
                        documents=d["documents"],
                        metadatas=d["metadatas"],
                        embeddings=d["embeddings"]
                    )
                    total_decisions += len(d["ids"])
            except Exception as e:
                if not self.silent:
                    print(f"⚠️ Hydration failed for {path}: {e}")
        
        if not self.silent:
            print(f"✅ Hydration complete: {total_decisions} memories restored.")

    def _load_encoder(self):
        try:
            return SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception as exc:
            if not self.silent:
                print(f"Falling back to local hash embeddings: {exc}")
            return None

    def _embed_text(self, text: str) -> list[float]:
        if self.encoder is not None:
            return self.encoder.encode(text).tolist()

        import zlib
        buckets = [0.0] * 384
        for token in text.lower().split():
            idx = zlib.adler32(token.encode()) % len(buckets)
            buckets[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]

    def store_decision(
        self,
        conflict_title: str,
        action_type: str,
        target_domain: str,
        reward: float,
        metrics_snapshot: dict,
        reasoning: str,
        trajectory: list[dict] = None,
        route_outcome: str = None,
        episode_id: str = None,
        personality_type: str = None
    ) -> None:
        """Stores individual decision for longitudinal tracking."""

        text = f"{conflict_title} Action: {action_type} Domain: {target_domain} Reward: {reward:.2f} {reasoning[:100]}"
        embedding = self._embed_text(text)

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "conflict_title": conflict_title,
                "action_type": action_type,
                "target_domain": target_domain,
                "reward": float(reward),
                "reasoning": reasoning,
                "route_outcome": route_outcome or "",
                "episode_id": episode_id or "",
                "personality_type": personality_type or "unknown",
                "timestamp": datetime.now().isoformat()
            }]
        )

    def store_trajectory(
        self,
        conflict_title: str = None,
        route_taken: str = None,
        total_reward: float = 0.0,
        metrics_diff_str: str = None,
        reasoning: str = None,
        task_id: str = None,
        trajectory_summary: dict = None
    ) -> None:
        """Stores a full trajectory summary."""

        if trajectory_summary is not None and task_id is not None:
            import json
            text = f"Task: {task_id} Route: {route_taken} Reward: {total_reward:.2f} Hits: {len(trajectory_summary.get('milestones_hit', []))}"
            embedding = self._embed_text(text)
            doc_id = str(uuid.uuid4())
            self.traj_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "task_id": task_id,
                    "route_taken": route_taken,
                    "reward": total_reward,
                    "summary": json.dumps(trajectory_summary),
                    "timestamp": datetime.now().isoformat()
                }]
            )
            if not self.silent:
                print(f"Stored task trajectory: {route_taken} (reward: {total_reward:.2f})")
            return

        # Fallback to older signature logic
        text = f"{conflict_title} Route: {route_taken} Diff: {metrics_diff_str} {reasoning[:100]}"
        embedding = self._embed_text(text)

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "conflict_title": conflict_title,
                "route_taken": route_taken,
                "metrics_diff": metrics_diff_str,
                "reward": total_reward,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat()
            }]
        )
        if not self.silent:
            print(f"Stored trajectory fallback: {route_taken} (reward: {total_reward:.2f})")

    def store_feedback(self, feedback) -> None:
        """Stores OutcomeFeedback linked to a specific episode."""
        import json
        text = f"Episode: {feedback.episode_id} Effectiveness: {feedback.overall_effectiveness} Resolution: {feedback.resolution_time_hours}h"
        embedding = self._embed_text(text)
        
        doc_id = f"fb_{feedback.episode_id}"
        self.feedback_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "episode_id": feedback.episode_id,
                "effectiveness": feedback.overall_effectiveness,
                "domains_improved": json.dumps(feedback.domains_improved),
                "domains_worsened": json.dumps(feedback.domains_worsened),
                "unexpected_effects": feedback.unexpected_effects,
                "resolution_time": feedback.resolution_time_hours,
                "timestamp": feedback.submitted_at.isoformat()
            }]
        )
        if not self.silent:
            print(f"Stored human feedback for episode {feedback.episode_id}")

    def retrieve_feedback(self, episode_id: str) -> Optional[dict]:
        """Retrieves feedback for a specific episode."""
        import json
        doc_id = f"fb_{episode_id}"
        results = self.feedback_collection.get(ids=[doc_id])
        
        if not results['metadatas']:
            return None
            
        meta = results['metadatas'][0]
        # Deserialize lists
        meta["domains_improved"] = json.loads(meta["domains_improved"])
        meta["domains_worsened"] = json.loads(meta["domains_worsened"])
        return meta

    def retrieve_similar_trajectories(self, task_domain: str, current_world: dict, n: int = 3) -> list[dict]:
        """Retrieve similar trajectories based on task domain and current world state."""
        import json
        if self.traj_collection.count() == 0:
            return []
            
        sorted_metrics = sorted(current_world.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        top_stressed = " ".join(f"{k}:{v}" for k, v in sorted_metrics[:3])
        query_text = f"TaskDomain: {task_domain} {top_stressed}"
        
        query_embedding = self._embed_text(query_text)
        results = self.traj_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, self.traj_collection.count())
        )
        
        output = []
        for i, meta in enumerate(results['metadatas'][0]):
            output.append({
                "task_id": meta.get("task_id", ""),
                "route_taken": meta.get("route_taken", ""),
                "reward": meta.get("reward", 0.0),
                "summary": json.loads(meta.get("summary", "{}")),
            })
        return output

    def retrieve_similar(self, conflict_title: str, current_metrics: dict, n: int = 3,
                         personality_type: str = None) -> list[dict]:
        """Retrieves the n most similar past high-reward decisions using semantic search.

        When personality_type is provided, tries personality-matched results first and
        fills remaining slots from the global pool so we always return n results.
        """
        if self.collection.count() == 0:
            return []

        sorted_metrics = sorted(current_metrics.items(), key=lambda x: x[1])
        top_stressed = " ".join(f"{k}:{v:.0f}" for k, v in sorted_metrics[:3])
        query_text = f"{conflict_title} {top_stressed}"
        query_embedding = self._embed_text(query_text)

        fetch = min(n * 4, self.collection.count())

        def _parse_results(results) -> list[dict]:
            output = []
            for i, meta in enumerate(results['metadatas'][0]):
                if meta.get("reward", 0.0) < 0.05:
                    continue
                distance = results['distances'][0][i]
                output.append({
                    "route_taken": meta.get("route_taken", ""),
                    "action_type": meta.get("action_type", ""),
                    "target_domain": meta.get("target_domain", ""),
                    "metrics_diff": meta.get("metrics_diff", ""),
                    "reward": meta.get("reward", 0.0),
                    "reasoning": meta.get("reasoning", ""),
                    "episode_id": meta.get("episode_id", ""),
                    "personality_type": meta.get("personality_type", "unknown"),
                    "similarity_score": round(1.0 / (1.0 + distance), 4),
                })
            return output

        # --- personality-matched pass ---
        matched: list[dict] = []
        if personality_type and personality_type != "unknown":
            try:
                r = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=fetch,
                    where={"personality_type": {"$eq": personality_type}},
                )
                matched = _parse_results(r)[:n]
            except Exception:
                pass  # personality filter unsupported or empty — fall through

        # --- global fill-up pass ---
        if len(matched) < n:
            r = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch,
            )
            seen_episodes = {m["episode_id"] for m in matched}
            for entry in _parse_results(r):
                if len(matched) >= n:
                    break
                if entry["episode_id"] not in seen_episodes:
                    matched.append(entry)
                    seen_episodes.add(entry["episode_id"])

        return matched[:n]

    def build_few_shot_prompt(self, conflict_title: str, current_metrics: dict,
                              personality_type: str = None) -> str:
        """Formats retrieved memories into a few-shot prompt block for the LLM."""
        memories = self.retrieve_similar(conflict_title, current_metrics, personality_type=personality_type)
        if not memories:
            return ""

        lines = ["--- PAST EXPERIENCE & HUMAN VERIFICATION ---"]
        for m in memories:
            episode_id = m.get("episode_id")
            fb_context = ""
            if episode_id:
                fb = self.retrieve_feedback(episode_id)
                if fb:
                    fb_context = f"\n  HUMAN FEEDBACK: Rated {fb['effectiveness']}/10. Notes: {fb['unexpected_effects']}"
            p_tag = f" [{m['personality_type']}]" if m.get("personality_type", "unknown") != "unknown" else ""
            short_reason = m['reasoning'][:120]
            line = (f"- Action Taken: [{m['action_type'].upper()}] on {m['target_domain'].upper()}{p_tag}\n"
                    f"  Agent's Initial Reasoning: {short_reason}{fb_context}")
            lines.append(line)

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Returns memory stats: total count, average reward, and route details."""
        if self.collection.count() == 0:
            return {"total_memories": 0, "average_reward": 0.0, "by_action_type": {}}

        all_records = self.collection.get(include=["metadatas"])
        metadatas = all_records["metadatas"]

        total = len(metadatas)
        avg_reward = sum(m.get("reward", 0.0) for m in metadatas) / total

        by_action_type = defaultdict(int)
        for m in metadatas:
            action_type = m.get("action_type")
            if action_type:
                by_action_type[action_type] += 1

        return {
            "total_memories": total,
            "average_reward": round(avg_reward, 3),
            "by_action_type": dict(by_action_type)
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
    print(f"By Action Type : {stats.get('by_action_type', stats.get('by_route_start'))}")


if __name__ == "__main__":
    main()
