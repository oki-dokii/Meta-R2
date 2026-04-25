
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.memory import LifeStackMemory

def purge_poison():
    print("🧹 Starting Memory Purge...")
    memory = LifeStackMemory(silent=True)
    
    # 1. Find all records containing "FALLBACK" or "Rate limited"
    all_data = memory.collection.get()
    poisoned_ids = []
    
    for i, doc in enumerate(all_data['documents']):
        if "FALLBACK" in doc or "Rate limit" in doc:
            poisoned_ids.append(all_data['ids'][i])
            
    if poisoned_ids:
        print(f"🗑️ Found {len(poisoned_ids)} poisoned memories. Deleting...")
        memory.collection.delete(ids=poisoned_ids)
        # Also clean traj_collection
        traj_data = memory.traj_collection.get()
        poisoned_traj_ids = []
        for i, doc in enumerate(traj_data['documents']):
            if "FALLBACK" in doc or "Rate limit" in doc:
                poisoned_traj_ids.append(traj_data['ids'][i])
        if poisoned_traj_ids:
             memory.traj_collection.delete(ids=poisoned_traj_ids)
        print("✅ Cleanup complete.")
    else:
        print("✨ No poisoned memories found (CLEAN).")

    print(f"Final Count: {memory.collection.count()} high-quality memories.")

if __name__ == "__main__":
    purge_poison()
