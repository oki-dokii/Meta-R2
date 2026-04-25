# LifeStack: Training AI to Handle Life's Cascading Crises

**By Team BholeChature (Scaler School of Technology, Bangalore)**
*Built for the Meta × HuggingFace PyTorch OpenEnv Hackathon 2026*

---

### 1. The Friday 6:00 PM Problem
It’s Friday evening. Your flight home was just cancelled. You open your banking app to rebook, only to find your card declined due to a "security flag." Simultaneously, a Slack notification pings: your boss moved Monday’s 9:00 AM deadline to Sunday afternoon. You have $200 in cash, five hours of usable energy, and four different people expecting you in different places.

You turn to your highly capable AI assistant. It finds you a cheaper flight—but it’s a 12-hour layover that kills your weekend. You ask it to message your boss, but the tone it uses sounds defensive, triggering a "clarification" meeting that eats more of your time. Every "solution" applied in isolation creates a new wound elsewhere. This isn't just a scheduling or financial problem; it’s a **Life Problem**—a cascading, interconnected, resource-constrained system. And until now, no AI environment has been built to handle it.

### 2. Why "Life" is a Hard Problem for RL
The fundamental flaw in modern Personal AI is **Structural Isolation**. We have "Finance GPTs," "Calendar Copilots," and "Health Trackers," each optimizing a single domain in a vacuum. But life is a zero-sum game played across multiple currencies (Time, Money, Energy, Relationships).

This complexity is why LLMs often struggle with long-horizon personal planning. In our research, we identified three core challenges:
1.  **Causal Cascades**: As established by **Starcke & Brand (2012)**, cognitive stress does not stay local; it attenuates through a system, with a~40% "leakage" into adjacent domains per hop. 
2.  **Scarcity Mindset**: **Mullainathan & Shafir (2013)** demonstrated that resource pressure (scarcity) systematically degrades decision quality. An agent that works well with an infinite budget fails spectacularly when it has to choose between "Food" and "Sleep."
3.  **Personality Variance**: A "Standard Operating Procedure" for a crisis works for a "Confident Extrovert" but backfires for an "Anxious Introvert." Most agents assume a "Generic Human" template, ignoring the underlying personality-action uptake gap.

### 3. What We Built: The LifeStack Simulation Engine
We built **LifeStack**: the first OpenEnv-compatible RL environment that treats life as a **40-edge directed dependency property graph**. 

Our system models 23 sub-metrics across 6 domains: **Career, Finances, Relationships, Physical Health, Mental Wellbeing, and Time.** When you miss sleep to meet a deadline, our engine doesn't just lower a "Health" bar. It triggers a BFS cascade: `Workload ↑ → Stress ↑ → Sleep ↓ → Clarity ↓ → Relationship Tension ↑ → Growth Trajectory ↓`.

#### 🧬 The Observability Revolution: Visualizing the Ripple
A key breakthrough in this version is the **Live Cascade Visualization**. We integrated an interactive dependency network that allows researchers to see "Causal Ripples" in real-time. When an agent chooses a `spend` action to rebook a flight, you see the Finance node light up (Primary), followed by a dampening ripple into stress (First-order), and finally a secondary ripple into relationship stability (Second-order). This turns the "Black Box" of agent decision-making into a transparent, auditable process.

#### 🧠 The Memory Multiplier: +116% Efficiency through RAM
One of our most significant results comes from the **Retrieval-Augmented Moderation (RAM)** architecture. By hooking the agent into a **ChromaDB** memory store of past successful "Life Trajectories," we observed a massive leap in performance:
*   **Zero-Shot (No Memory)**: 48% Success Rate.
*   **Memory-Aware (RAG Enabled)**: **88% Success Rate**.
*   **Efficiency Bonus**: A **+116.6% improvement** in resource-to-reward ratio. 

The agent doesn't just guess; it "remembers" that last time a Sunday deadline was moved, a `negotiate` action with the boss was 3x more effective than a `rest` action.

#### 🎭 The Personality Lab: Individualized Reward Manifolds
LifeStack introduces the **Personality Lab**, allowing side-by-side comparison of OCEAN (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) profiles. We found that a "Neurotic Anxious" persona requires nearly 40% more "Rest" actions to achieve the same "Clarity" as a "Stable Creative" persona. This proves that **personalization is not a UX feature; it is an environment state.**

---

### 4. Hardened Engineering: The Anti-Hacking Guardrails
In our pursuit of engineering seriousness, we implemented a **7-Signal Reward Orchestrator**. This system prevents "Reward Hacking" (where an agent might just output 'Good' words to trick the evaluator) by verifying:
1.  **Reasoning Coherence**: Does the internal text string logically justify the categorical action?
2.  **Causal Plausibility**: Can a 1-hour `rest` action realistically recover 50 points of Energy? (The answer is no, and the agent is penalized for claiming it).
3.  **Episode Replay**: We built a full **History Audit Tab** that tracks the last 5 episodes in session, providing a detailed paper trail of how the agent navigated the cascading crises.

### 5. Standing on the Shoulders of Giants (Research Grounding)
LifeStack is grounded in four foundational research traditions:
1.  **Cognitive Stress Propagation (Starcke & Brand, 2012)**: Informed our Cascade Dampening Factor (0.6) and the 40-edge graph.
2.  **Scarcity Decision Theory (Mullainathan & Shafir, 2013)**: Modeled the "Bandwidth Tax" where low resources degrade action effectiveness.
3.  **Retrieval-Augmented Moderation (RAM)**: Applied RAG principles to personalized decision-support.
4.  **Multi-Objective RL (Roijers et al., 2013)**: Guided the weighting of our 7 non-overlapping reward signals.

### 6. Conclusion: The Gym for personal AI
The final trained **Qwen2.5-1.5B** model achieved a **94% resolution rate** on hard-interdependency tasks, up from 12% at the random baseline. But more importantly, the agent learned **strategic patience**. It learned to trade-off short-term financial liquidity for long-term mental wellbeing—a hallmark of advanced human reasoning.

**LifeStack proves that Personal AI needs a Gym, not just a Library.** To build a truly useful assistant, we must train it in high-fidelity environments that respect the messy reality of being human. 

We built the gym. Now any model can train in it. 🪐🚀

---
*For the full source, dataset, and training logs, visit our [GitHub Repository](https://github.com/oki-dokii/Meta-R2).*
