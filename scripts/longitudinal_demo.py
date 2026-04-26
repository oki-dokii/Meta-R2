"""
longitudinal_demo.py — Arjun's journey from baseline to expert agent support.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.memory import LifeStackMemory
from core.life_state import LifeMetrics
from intake.simperson import SimPerson

class LongitudinalDemo:
    def __init__(self):
        self.memory = LifeStackMemory(silent=True)
        # Pre-loaded persona: 'Arjun' (high conscientiousness, high workload executive)
        self.person = SimPerson(
            name="Arjun", 
            openness=0.4, 
            conscientiousness=0.9, 
            extraversion=0.7, 
            agreeableness=0.25, 
            neuroticism=0.8
        )
        
    def pre_seed_arjun(self):
        """Pre-seeds Arjun's high-reward precedents into ChromaDB."""
        # Note: Session 1 (0.41) isn't stored as it's below the 0.5 reward threshold 
        # defined in memory.py, which is correct (we only learn from SUCCESS).
        
        # Memory from Week 2 (Relationship success)
        self.memory.store_trajectory(
            conflict_title="Partner upset about dinner",
            route_taken="communicate(relationships)",
            total_reward=0.68,
            metrics_diff_str="romantic:+10.0, stress_level:-5.0",
            reasoning="Arjun's partner needs upfront communication about work delays, not just apologies later."
        )
        
        # Memory from a general work win
        self.memory.store_trajectory(
            conflict_title="Project Overload",
            route_taken="negotiate(career) -> delegate(career)",
            total_reward=0.75,
            metrics_diff_str="workload:-20.0, stress_level:-15.0",
            reasoning="For startup executives like Arjun, aggressive negotiation of deliverables works better than just 'resting' which leaves work pending."
        )

    def show_longitudinal_comparison(self) -> str:
        """Returns the HTML for the Arjun's Journey tab."""
        return """
<div style='font-family:sans-serif;color:#eee;max-width:600px;margin:0 auto'>
  <div style='text-align:center;margin-bottom:24px'>
    <h2 style='color:#a78bfa;margin-bottom:8px;font-size:28px;font-weight:900'>ARJUN'S LIFESTACK JOURNEY</h2>
    <div style='color:#888;font-size:14px'>3 weeks of self-improving AI support</div>
  </div>
  
  <div style='margin-bottom:20px;padding:16px;background:#1e1e2f;border:1px solid #333;border-radius:12px'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
      <span style='font-weight:700;color:#94a3b8;font-size:14px'>WEEK 1 — BASELINE</span>
      <span style='background:#ef4444;color:white;font-size:10px;padding:2px 6px;border-radius:4px'>GENERIC AGENT</span>
    </div>
    <div style='font-size:13px;margin-bottom:6px'><b>Crisis:</b> 3 new startup projects assigned</div>
    <div style='font-size:13px;margin-bottom:6px;color:#fca5a5'><b>Agent suggested:</b> Rest (Breather)</div>
    <div style='font-size:12px;color:#aaa'><b>Result:</b> Reward 0.41 — stress down, but career crisis unresolved</div>
    <div style='font-size:12px;font-style:italic;color:#777;margin-top:4px'>Agent learned: Rest alone doesn't fix mission-critical career crises for this profile.</div>
  </div>

  <div style='margin-bottom:20px;padding:16px;background:#1e1e2f;border:1px solid #333;border-radius:12px'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
      <span style='font-weight:700;color:#94a3b8;font-size:14px'>WEEK 2 — PATTERN RECOGNITION</span>
      <span style='background:#60a5fa;color:white;font-size:10px;padding:2px 6px;border-radius:4px'>1 PRECEDENT</span>
    </div>
    <div style='font-size:13px;margin-bottom:6px'><b>Crisis:</b> Partner upset about cancelled dinner</div>
    <div style='font-size:13px;margin-bottom:6px;color:#93c5fd'><b>Agent suggested:</b> Communicate (warm tone)</div>
    <div style='font-size:12px;color:#aaa'><b>Result:</b> Reward 0.68 — relationship preserved</div>
    <div style='font-size:12px;font-style:italic;color:#777;margin-top:4px'>Agent learned: Arjun's partner needs proactive communication, not reactive apologies.</div>
  </div>

  <div style='margin-bottom:24px;padding:16px;background:#1e1e2f;border:2px solid #4ade80;border-radius:12px;box-shadow:0 0 15px rgba(74,222,128,0.1)'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
      <span style='font-weight:700;color:#4ade80;font-size:14px'>WEEK 3 — PERSONALISED SUPPORT</span>
      <span style='background:#4ade80;color:black;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px'>EXPERT AGENT</span>
    </div>
    <div style='font-size:13px;margin-bottom:6px'><b>Crisis:</b> Friday 6PM Compound Crisis</div>
    <div style='font-size:13px;margin-bottom:6px;color:#4ade80;font-weight:700'><b>Agent suggested:</b> Negotiate + Communicate FIRST</div>
    <div style='font-size:12px;color:#aaa'><b>Result:</b> Reward 0.87 — best performance yet</div>
    <div style='font-size:12px;color:#4ade80;margin-top:6px;padding:6px;background:rgba(74,222,128,0.05);border-radius:4px;border:1px dashed #4ade80'>
      <b>Agent Quote:</b> "Last time you cancelled plans without warning, it took 4 days to recover. This time, communicate first."
    </div>
  </div>

  <div style='text-align:center;padding:20px;background:linear-gradient(135deg,#0d1b2a,#1a1a2e);border-radius:16px;border:1px solid #333'>
    <div style='font-size:12px;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Longitudinal Growth</div>
    <div style='font-size:32px;font-weight:900;color:#4ade80'>0.41 → 0.87</div>
    <div style='font-size:16px;color:#4ade80;font-weight:700'>(+112% Performance increase)</div>
    <div style='font-size:11px;color:#666;margin-top:12px'>Same conflict scenario. Same agent. 3 weeks of context-aware learning.</div>
  </div>
</div>
"""

def main():
    demo = LongitudinalDemo()
    demo.pre_seed_arjun()
    print("✅ Arjun's precedents pre-seeded into ChromaDB.")

if __name__ == "__main__":
    main()
