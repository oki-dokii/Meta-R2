import os
import sys
import json
import chromadb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def export_memory():
    path = "./lifestack_memory"
    dest = "./data/preseeded_memory.json"
    
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
    
    os.makedirs("./data", exist_ok=True)
    with open(dest, "w") as f:
        json.dump(export_data, f)
    
    print(f"✅ Successfully exported {len(all_decisions['ids'])} decisions and {len(all_trajectories['ids'])} trajectories to {dest}")

if __name__ == "__main__":
    export_memory()
