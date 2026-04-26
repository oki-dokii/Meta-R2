import os
import sys
import json
import chromadb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _write_partitioned_export(export_data: dict, output_dir: str) -> list[str]:
    shards = []
    decisions = export_data["decisions"]
    ids = decisions.get("ids") or []
    midpoint = len(ids) // 2
    partitions = [
        ("preseeded_memory_p1.json", slice(0, midpoint)),
        ("preseeded_memory_p2.json", slice(midpoint, None)),
    ]

    shared = {"trajectories": export_data["trajectories"]}
    for filename, shard_slice in partitions:
        shard_path = os.path.join(output_dir, filename)
        shard_decisions = {
            "ids": decisions["ids"][shard_slice],
            "documents": decisions["documents"][shard_slice],
            "metadatas": decisions["metadatas"][shard_slice],
            "embeddings": decisions["embeddings"][shard_slice] if decisions["embeddings"] is not None else None,
        }
        with open(shard_path, "w") as f:
            json.dump({**shared, "decisions": shard_decisions}, f)
        shards.append(shard_path)

    return shards

def export_memory():
    path = "./lifestack_memory"
    output_dir = "./data"
    
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found.")
        return

    print(f"📦 Exporting wisdom from {path}...")
    client = chromadb.PersistentClient(path=path)
    
    # Export decisions
    decisions = client.get_collection(name='decisions')
    all_decisions = decisions.get(include=["documents", "metadatas", "embeddings"])
    
    # Export trajectories
    trajectories = client.get_collection(name='trajectories')
    all_trajectories = trajectories.get(include=["documents", "metadatas", "embeddings"])
    
    export_data = {
        "decisions": {
            "ids": all_decisions["ids"],
            "documents": all_decisions["documents"],
            "metadatas": all_decisions["metadatas"],
            "embeddings": [e.tolist() if hasattr(e, 'tolist') else e for e in all_decisions["embeddings"]] if all_decisions["embeddings"] is not None else None
        },
        "trajectories": {
            "ids": all_trajectories["ids"],
            "documents": all_trajectories["documents"],
            "metadatas": all_trajectories["metadatas"],
            "embeddings": [e.tolist() if hasattr(e, 'tolist') else e for e in all_trajectories["embeddings"]] if all_trajectories["embeddings"] is not None else None
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    shards = _write_partitioned_export(export_data, output_dir)

    print(
        f"✅ Successfully exported {len(all_decisions['ids'])} decisions and "
        f"{len(all_trajectories['ids'])} trajectories to {', '.join(shards)}"
    )

if __name__ == "__main__":
    export_memory()
