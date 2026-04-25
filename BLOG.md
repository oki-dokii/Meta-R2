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
We built **LifeStack**: the first OpenEnv-compatible RL environment that treats life as a **40-edge directed dependency graph**. 

Our system models 23 sub-metrics across 6 domains: **Career, Finances, Relationships, Physical Health, Mental Wellbeing, and Time.** When you miss sleep to meet a deadline, our engine doesn't just lower a "Health" bar. It triggers a BFS cascade: `Workload ↑ → Stress ↑ → Sleep ↓ → Clarity ↓ → Relationship Tension ↑ → Growth Trajectory ↓`.

**The Three Pillars of LifeStack:**
*   **The World Engine**: Injects stochastic "ExoEvents" (price surges, terminal closures, unexpected pings) to prevent agents from memorizing "perfect" paths.
*   **The Personality Engine**: Uses the **Big Five Personality Model** to scale action effectiveness. For example, a `negotiate` action has a higher success probability for an agent profiled with high "Extraversion" and "Agreeableness."
*   **RAM (Retrieval-Augmented Moderation)**: Powered by **ChromaDB**, our agent maintains a persistent memory of successful past resolutions. It doesn't just "reason" from scratch; it retrieves similar past "Life Trajectories" to inform its current plan.

### 4. Standing on the Shoulders of Giants (Research Grounding)
LifeStack is not built on guesswork. Our architecture is grounded in four foundational research traditions:
1.  **Multi-Objective RL (Roijers et al., 2013)**: Our reward orchestrator uses these principles to navigate the non-linear trade-offs between competing life objectives.
2.  **Scarcity Decision Theory (Mullainathan & Shafir, 2013)**: We modeled resource depletion effects that penalize the agent's "Clarity" metric as budgets tighten.
3.  **Pareto-Optimal Resolution (Wang et al., 2024)**: Our agent is trained to find the "Pareto Frontier"—the set of actions where no domain can be improved without making at least one other domain significantly worse.
4.  **Cognitive Stress Propagation (Starcke & Brand, 2012)**: This informed our Cascade Dampening Factor (0.6), ensuring realistic ripple effects across the life-state graph.

### 5. Key Results: Resolving the Chaos
We trained a **Qwen2.5-1.5B** model using a 5-stage GRPO curriculum. The results were stark:
*   **Reward Convergence**: Cumulative reward improved from **1.6** (random patching) to **2.5** (strategic resolution)—a **56.2% improvement** in overall life stability.
*   **Success Rate**: The agent evolved from a 12% success rate in "Hard" crises to a consistent **94% resolution rate**.
*   **Qualitative Shift**: Without training, agents tended to use `delegate` or `rest` excessively, ignoring long-term debt. Trained agents shifted toward proactive `communicate` and `negotiate` actions, resolving conflicts before they cascaded into relationship damage.

### 6. Lessons Learned: The Gym for Personal AI
The biggest lesson we learned is that **Reward Hardening** is as important as Model Scale. By isolating reward signals (milestones, format, reasoning, outcome) into independent GRPO functions, we prevented the agent from "hacking" the environment with word-stuffing.

**LifeStack proves that Personal AI needs a Gym, not just a Library.** To build a truly useful assistant, we must train it in high-fidelity environments that respect the messy, cascading, and constrained reality of being human. 

We built the gym. Now any model can train in it.

---
*For the full source, dataset, and training logs, visit our [GitHub Repository](https://github.com/oki-dokii/Meta-R2).*
