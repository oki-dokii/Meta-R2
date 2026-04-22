"""
validate_simperson.py — Empirical validation of the SimPerson OCEAN model.
Verifies outputs are consistent with published stress-personality research.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from intake.simperson import SimPerson


passed = 0
total  = 5


def report(name, ok, detail=""):
    global passed
    tag = "✅ PASS" if ok else "❌ FAIL"
    passed += ok
    print(f"  {tag}  {name}")
    if detail:
        print(f"         {detail}")
    print()


# ─── Check 1: Neuroticism-stress correlation ─────────────────────────────────
def check_neuroticism_stress():
    """
    High neuroticism should degrade uptake for 'delegate' under stress.
    Starcke & Brand (2012): neurotic individuals show amplified stress
    interference with executive function — delegation requires exactly that.
    Expected: negative correlation (r < -0.5).
    """
    n_values = np.linspace(0.1, 1.0, 50)
    uptakes  = []

    for n in n_values:
        person = SimPerson(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=float(n), name="test"
        )
        u = person.respond_to_action("delegate", {"time": 5, "money": 100, "energy": 30}, 90.0)
        uptakes.append(u)

    r = np.corrcoef(n_values, uptakes)[0, 1]
    report(
        "Neuroticism-stress correlation",
        r < -0.5,
        f"r = {r:.4f} (expected < -0.5)"
    )


# ─── Check 2: Agreeableness-communication correlation ────────────────────────
def check_agreeableness_communication():
    """
    High agreeableness should boost communication uptake.
    Consistent with Costa & McCrae (1992): agreeable individuals are
    more effective at interpersonal negotiation and conflict de-escalation.
    Expected: positive correlation (r > 0.4).
    """
    a_values = np.linspace(0.1, 1.0, 50)
    uptakes  = []

    for a in a_values:
        person = SimPerson(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=float(a), neuroticism=0.5, name="test"
        )
        u = person.respond_to_action("communicate", {"time": 2, "money": 0, "energy": 10}, 50.0)
        uptakes.append(u)

    r = np.corrcoef(a_values, uptakes)[0, 1]
    report(
        "Agreeableness-communication correlation",
        r > 0.4,
        f"r = {r:.4f} (expected > 0.4)"
    )


# ─── Check 3: Stress degradation is monotonic ────────────────────────────────
def check_stress_monotonic():
    """
    For a moderately neurotic person, uptake for 'rest' should decrease
    as stress increases — higher stress impairs even recovery actions.
    Expected: strictly non-increasing uptake across stress levels.
    """
    person = SimPerson(
        openness=0.5, conscientiousness=0.5, extraversion=0.3,
        agreeableness=0.5, neuroticism=0.7, name="test"
    )
    stress_levels = [10, 30, 50, 70, 90]
    uptakes = []

    for s in stress_levels:
        u = person.respond_to_action("rest", {"time": 2, "money": 0, "energy": -20}, float(s))
        uptakes.append(u)

    monotonic = all(uptakes[i] >= uptakes[i + 1] for i in range(len(uptakes) - 1))
    detail_parts = [f"stress={s}: uptake={u:.3f}" for s, u in zip(stress_levels, uptakes)]
    report(
        "Stress degradation is monotonic",
        monotonic,
        " | ".join(detail_parts)
    )


# ─── Check 4: Personality profiles are diverse ───────────────────────────────
def check_profile_diversity():
    """
    The 5 pre-built profiles should have different dominant OCEAN traits.
    This ensures the agent encounters meaningfully different people during
    training — critical for generalisation.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simperson_profiles.json")
    with open(data_path) as f:
        profiles = json.load(f)

    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    dominants = []
    lines = []

    for p in profiles:
        scores    = {t: p[t] for t in traits}
        dominant  = max(scores, key=scores.get)
        dominants.append(dominant)
        lines.append(f"{p['name']}: dominant = {dominant} ({scores[dominant]:.2f})")

    unique_count = len(set(dominants))
    # At least 4 out of 5 should have different dominant traits
    report(
        "Personality profiles are diverse",
        unique_count >= 4,
        f"{unique_count}/5 unique dominant traits\n         " + "\n         ".join(lines)
    )


# ─── Check 5: Uptake bounds always respected ─────────────────────────────────
def check_uptake_bounds():
    """
    Across 100 random personalities × 7 action types × 3 stress levels,
    all 2100 uptake scores must be in [0.1, 1.0].
    """
    import random
    random.seed(42)

    action_types   = ["communicate", "delegate", "rest", "structured_plan",
                      "negotiate", "spend", "exercise"]
    stress_levels  = [10.0, 50.0, 90.0]
    violations     = 0
    total_checks   = 0

    for _ in range(100):
        person = SimPerson(name="rand")  # random OCEAN from defaults
        for at in action_types:
            for s in stress_levels:
                u = person.respond_to_action(at, {"time": 3, "money": 50, "energy": 20}, s)
                total_checks += 1
                if u < 0.1 or u > 1.0:
                    violations += 1

    report(
        "Uptake bounds [0.1, 1.0] always respected",
        violations == 0,
        f"{violations}/{total_checks} violations"
    )


# ─── Run All ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  SimPerson Empirical Validation Suite")
    print("  Based on: Starcke & Brand (2012), Costa & McCrae (1992)")
    print("=" * 64 + "\n")

    check_neuroticism_stress()
    check_agreeableness_communication()
    check_stress_monotonic()
    check_profile_diversity()
    check_uptake_bounds()

    print("=" * 64)
    color = "\033[92m" if passed == total else "\033[91m"
    print(f"  SimPerson Validation: {color}{passed}/{total} checks passed\033[0m")
    verdict = "YES" if passed == total else "NO"
    v_color = "\033[92m" if passed == total else "\033[91m"
    print(f"  Model is empirically consistent with published stress-personality research: {v_color}{verdict}\033[0m")
    print("=" * 64 + "\n")
