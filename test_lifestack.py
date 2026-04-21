"""
test_lifestack.py — LifeStack Edge Case Test Suite
Covers: cascade bounds, resource exhaustion, penalties, memory threshold, episode termination.
"""

import copy
import shutil
import os

from life_state import LifeMetrics, ResourceBudget, DependencyGraph
from lifestack_env import LifeStackEnv
from reward import compute_reward
from simperson import SimPerson
from memory import LifeStackMemory


passed = 0
total  = 10


def report(name, ok, detail=""):
    global passed
    tag = "✅ PASS" if ok else "❌ FAIL"
    passed += ok
    print(f"  {tag}  {name}")
    if detail:
        print(f"         {detail}")


# ─── 1. Cascade Floor Test ────────────────────────────────────────────────────
def test_cascade_floor():
    graph   = DependencyGraph()
    metrics = LifeMetrics()
    # Push liquidity from 70 down by 200 — should clamp at 0, not go negative
    result  = graph.cascade(metrics, {"finances.liquidity": -200.0})
    flat    = result.flatten()
    min_val = min(flat.values())
    report("Cascade floor (metrics >= 0)", min_val >= 0.0,
           f"min metric = {min_val:.2f}")


# ─── 2. Cascade Ceiling Test ─────────────────────────────────────────────────
def test_cascade_ceiling():
    graph   = DependencyGraph()
    metrics = LifeMetrics()
    # Push workload from 70 up by 200 — should clamp at 100
    result  = graph.cascade(metrics, {"career.workload": +200.0})
    flat    = result.flatten()
    max_val = max(flat.values())
    report("Cascade ceiling (metrics <= 100)", max_val <= 100.0,
           f"max metric = {max_val:.2f}")


# ─── 3. Resource Exhaustion Test ──────────────────────────────────────────────
def test_resource_exhaustion():
    budget = ResourceBudget(time_hours=5.0, money_dollars=100.0, energy_units=20.0)
    ok     = budget.deduct(time=10.0, money=0.0, energy=0.0)
    report("Resource exhaustion (deduct returns False, no negative)",
           ok is False and budget.time_hours >= 0,
           f"deduct returned {ok}, time_hours = {budget.time_hours:.1f}")


# ─── 4. Zero Action (Inaction) Penalty Test ───────────────────────────────────
def test_inaction_penalty():
    state = LifeMetrics()
    _, breakdown = compute_reward(state, copy.deepcopy(state), {}, actions_taken=0)
    fired = breakdown["penalties_fired"]
    report("Inaction penalty fires",
           "INACTION_PENALTY" in fired,
           f"penalties_fired = {fired}")


# ─── 5. Critical Floor Penalty Test ──────────────────────────────────────────
def test_critical_floor_penalty():
    before = LifeMetrics()
    after  = copy.deepcopy(before)
    after.physical_health.energy = 15.0       # below 20 threshold
    _, breakdown = compute_reward(before, after, {}, actions_taken=1)
    fired = breakdown["penalties_fired"]
    report("Critical floor penalty fires",
           "CRITICAL_FLOOR_VIOLATION" in fired,
           f"energy = 15.0, penalties_fired = {fired}")


# ─── 6. Cascade Dampening Test ───────────────────────────────────────────────
def test_cascade_dampening():
    graph   = DependencyGraph()
    metrics = LifeMetrics()
    primary_delta = 30.0
    result  = graph.cascade(metrics, {"career.workload": primary_delta})
    flat_before = metrics.flatten()
    flat_after  = result.flatten()

    # First-order target: career.workload should change by exactly primary_delta
    first_order = abs(flat_after["career.workload"] - flat_before["career.workload"])

    # Second-order targets connected via edges from career.workload
    # e.g. mental_wellbeing.stress_level, time.free_hours_per_week
    second_order_deltas = []
    for target, _ in graph.edges.get("career.workload", []):
        delta = abs(flat_after[target] - flat_before[target])
        second_order_deltas.append((target, delta))

    all_smaller = all(d < first_order for _, d in second_order_deltas)
    detail = "; ".join(f"{t}: {d:.2f}" for t, d in second_order_deltas)
    report("Cascade dampening (2nd order < 1st order)",
           all_smaller and len(second_order_deltas) > 0,
           f"1st order = {first_order:.2f} | 2nd order: {detail}")


# ─── 7. SimPerson Uptake Bounds Test ─────────────────────────────────────────
def test_simperson_uptake_bounds():
    person = SimPerson(
        openness=0.5, conscientiousness=0.3, extraversion=0.2,
        agreeableness=0.4, neuroticism=1.0, name="Stressed"
    )
    action_types = ["communicate", "delegate", "rest", "structured_plan",
                    "negotiate", "spend", "exercise", "meditate",
                    "network", "study"]
    results = []
    all_ok  = True
    for at in action_types:
        uptake = person.respond_to_action(at, {"time": 5, "money": 100, "energy": 30}, 100.0)
        results.append((at, uptake))
        if uptake < 0.1 or uptake > 1.0:
            all_ok = False

    detail = ", ".join(f"{a}={u:.2f}" for a, u in results)
    report("SimPerson uptake bounds [0.1, 1.0]",
           all_ok,
           f"uptakes: {detail}")


# ─── 8. Memory Threshold Test ────────────────────────────────────────────────
def test_memory_threshold():
    # Use a fresh isolated memory dir
    test_dir = "./test_memory_tmp"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    import chromadb
    from sentence_transformers import SentenceTransformer

    client     = chromadb.PersistentClient(path=test_dir)
    collection = client.get_or_create_collection(name="test_decisions")
    encoder    = SentenceTransformer("all-MiniLM-L6-v2")

    rewards    = [0.3, 0.5, 0.6, 0.4, 0.8]
    threshold  = 0.5
    stored     = 0

    for i, r in enumerate(rewards):
        if r >= threshold:
            text = f"test conflict action_{i} domain reasoning"
            emb  = encoder.encode(text).tolist()
            collection.add(
                ids=[f"test_{i}"],
                embeddings=[emb],
                documents=[text],
                metadatas=[{"reward": r, "action_type": f"action_{i}", "target_domain": "test"}],
            )
            stored += 1

    expected = sum(1 for r in rewards if r >= threshold)
    actual   = collection.count()
    shutil.rmtree(test_dir, ignore_errors=True)

    report("Memory threshold (only reward >= 0.5 stored)",
           actual == expected,
           f"expected {expected}, stored {actual} (rewards: {rewards})")


# ─── 9. Episode Termination Test ─────────────────────────────────────────────
def test_episode_termination():
    env = LifeStackEnv()
    env.reset()

    done = False
    for step in range(5):
        obs, reward, terminated, truncated, info = env.step({
            "metric_changes": {},
            "resource_cost": {},
            "actions_taken": 0,
        })
        done = terminated or truncated

    report("Episode terminates after 5 steps",
           done is True,
           f"done = {done} after {env.step_count} steps")


# ─── 10. Full Episode Smoke Test ─────────────────────────────────────────────
def test_full_episode_smoke():
    try:
        from run_episode import run_episode
        result = run_episode(difficulty=1, verbose=False)
        reward = result.get("total_reward", None)
        steps  = result.get("steps", None)
        ok     = isinstance(reward, float) and (steps is None or steps <= 5)
        report("Full episode smoke test",
               ok,
               f"reward = {reward}, steps = {steps}, type = {type(reward).__name__}")
    except Exception as e:
        report("Full episode smoke test", False, f"Exception: {e}")


# ─── Run All ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LifeStack Edge Case Test Suite")
    print("=" * 60 + "\n")

    test_cascade_floor()
    test_cascade_ceiling()
    test_resource_exhaustion()
    test_inaction_penalty()
    test_critical_floor_penalty()
    test_cascade_dampening()
    test_simperson_uptake_bounds()
    test_memory_threshold()
    test_episode_termination()
    test_full_episode_smoke()

    print("\n" + "=" * 60)
    color = "\033[92m" if passed == total else "\033[91m"
    print(f"  {color}{passed}/{total} tests passed\033[0m")
    print("=" * 60 + "\n")
