# Mentor Meeting Playbook: LifeStack Engine

## 1. The 90-Second Opening Pitch
*Start with the problem, not the architecture.*

> "Most AI assistants treat your life like a single-turn Q&A. You ask 'how do I handle my boss's deadline?' and it gives you a tip. But that tip ignores the reality: you're already sleep-deprived, your partner is frustrated, and spending 4 hours on work tonight has a 7-day ripple effect on your health. 
>
> **LifeStack** is the first AI system that treats your life as a multi-domain reinforcement learning environment. We train a policy that explicitly optimizes for long-term wellbeing by understanding how a crisis in one domain cascades into others."

**Action**: Immediately open the **Situational Portal**, select **Alex (Executive)** + **Friday 6PM Conflict**, and hit **Start Simulation**. Let the cascade animation run.

---

## 2. The Technical Narrative (2 Minutes)

1.  **The Environment**: "We modeled 6 life domains and 23 metrics with a real causal dependency graph. A career crisis actually cascades into stress, which impacts sleep, which strains relationships. The animation you just saw is driven by that underlying graph."
2.  **The Training Signal**: "We trained Qwen2.5-1.5B via **GRPO** using 8 reward functions. The core innovation is our **7-day γ=0.9 discounted rollout reward**. The model is penalized if a decision looks good today but causes a system collapse by day 4. It's training for resilience, not just compliance."
3.  **The Memory Flywheel**: "Using ChromaDB, the agent retrieves its own decision history. Over time, it reasons differently based on what worked. We've measured a significant reward delta between 'Cold' and 'Warm' starts."
4.  **Real-World Grounding**: "We built a production-ready sync pattern using Google OAuth (Gmail/Calendar) with a robust fallback system for demo stability."

---

## 3. The 3 Demo Moments
*Focus on depth over breadth.*

| Moment | What to Show | Key Talking Point |
| :--- | :--- | :--- |
| **1. The Cascade** | Situational Portal -> Start Animation | "Observe how the workload spike propagates into mental wellbeing—this is the dependency graph in action, not a script." |
| **2. Policy Power** | ⚡ Trained vs. Baseline Tab | "The rule-based baseline just fixes the 'worst' metric. Our trained agent reasons across all domains to find the optimal long-term path." |
| **3. The Forecast** | Trajectory Panel (after simulation) | "This is a real environment rollout. The model was explicitly optimized to flatten these negative curves over a 168-hour horizon." |

---

## 4. Counter-Questions & Strategic Answers

**"Is this just a prompted LLM?"**
> "No. The base model is Qwen2.5-1.5B, but we've fine-tuned it using GRPO. Our reward functions are tied to the environment simulator, meaning the model's gradient is shaped by the survival of the 'SimPerson'."

**"Is the cascade graph hand-coded?"**
> "The structure is expert-defined (causal), but the policy navigating it is learned. This is the 'Prior-Agent' split: the world has rules, but the intelligence is in how you navigate them."

**"Why not just use GPT-4?"**
> "You can't train GPT-4 on your personal life-state rewards. LifeStack allows for local optimization, hyper-specific memory retrieval, and 7-day horizon planning that a general one-shot LLM can't replicate."

**"Why a 1.5B model?"**
> "It's a feature, not a limitation. It allows for on-device deployment (privacy for personal data) and fast training iterations while being fully capable of outputting the required structured reasoning."

---

## 5. System Architecture

- **Ingest**: Gmail/Calendar/Fitness signals
- **Environment**: LifeStack Env (6 domains, 23 metrics, causal graph)
- **Engine**: Conflict cascade (3-phase animation)
- **Agent**: GRPO-Trained Qwen2.5-1.5B
- **Optimization**: 8 reward functions incl. 7-day rollout
- **Memory**: ChromaDB retrieval-augmented policy

---

## 6. Closing Line
> "We've built more than a chatbot. We've built the first life-management operating system where the state is your actual life, the transitions are causally grounded, and the agent learns to prioritize your long-term health over short-term noise."
