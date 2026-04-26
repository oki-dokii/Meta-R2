# Building LifeStack: An RL Environment Where the Agent Manages a Human Life

---

## The problem we kept hitting

Most RL environments are toy problems — grids, games, isolated tasks. An agent navigates a maze. An agent plays chess. An agent controls a robot arm. These are all tractable because the state is clean, the action space is narrow, and consequences are immediate and observable.

Real decisions are none of those things. When you take on an extra project at work, you don't just see a `career.workload += 30` in your heads-up display. You see your sleep get worse three weeks later because stress crept up. You see your partner getting quietly frustrated because you've been distracted. You see your savings dwindle because you started ordering delivery every night to save time. The problem isn't solving one crisis — it's navigating a system of crises that are causally entangled and temporally displaced.

We wanted an environment that forced a model to reason about all of this at once. Not an environment that *simulates* messy decisions with heuristics, but one where the causal structure is explicit, the resources are finite, and the agent genuinely cannot win by optimizing one dimension without watching others collapse.

That's LifeStack. And it was harder to build than we expected — mostly because of the reward function.

---

## What LifeStack actually is

A person wakes up with a life crisis. In the concrete terms of `data/conflicts.json`, this might be "The Perfect Storm" — a conflict where `career.workload` spikes by 30 points (a project deadline moved up), `finances.liquidity` drops by 40 points (an unexpected car repair bill), and `relationships.romantic` is already fragile from two weeks of working late. The agent gets $500 and 20 hours. That's the budget. It needs to resolve things without any metric hitting 10 — that's the floor where the episode terminates as a failure.

The 23 metrics across 6 life domains (`core/life_state.py`) aren't independent. They're wired together in a `DependencyGraph` with 32 directed edges and a 0.6 dampening factor per propagation hop. The dampening value comes from Starcke & Brand (2012) — stress effects on decision-making attenuate roughly 40% per cognitive/behavioral hop, so `cascade_utils.py` uses `CASCADE_DAMPENING_DEFAULT = 0.6` as the per-edge decay.

What this means concretely: increase `career.workload` by 20 points, and the cascade graph automatically raises `mental_wellbeing.stress_level` by `20 × 0.70 × 0.6 = 8.4` points. That stress increase then reduces `physical_health.sleep_quality` by `8.4 × (-0.55) × 0.6 = -2.77`. That sleep degradation reduces `mental_wellbeing.clarity` by `2.77 × 0.60 × 0.6 = -1.0`. The third-order effects are small but they compound across a 30-step episode.

The agent cannot prevent any of this. It can only decide *which* cascades are worth triggering by choosing which domains to spend resources on first.

Beyond the metric cascade, the environment has two additional layers of unpredictability. `ExoEvent` objects fire at scheduled or probabilistic steps, mutating world state and sometimes closing routes the agent was planning to use. The `FlightCrisisTask` (the default task in `core/task.py`) has a `price_surge` event at step 5 that makes one route more expensive, and a `lounge_full` event at step 8 that closes a recovery route entirely. The agent can see that events exist in `obs.metadata["events"]`, but it cannot know their content until they fire. There's also a `hidden_state` — facts like `{"card_available": True}` that the agent can only learn by spending an action on `action_type="inspect"`.

---

## The reward design rabbit hole

We started with a single reward: the sum of metric improvements. This was a mistake. The model learned two degenerate strategies almost immediately.

The first: rest constantly. `action_type="rest"` has near-zero resource cost and recovers physical energy. The model figured out that resting raises `physical_health.energy`, which via the dependency graph raises `mental_wellbeing.motivation`, which slightly raises `career.growth_trajectory`. Net outcome: slow positive drift on several metrics, no resource spend, no cascade risk. Reward goes up. Episode never terminates. We added `REST_NOT_JUSTIFIED` — a penalty of `-0.25` that fires in `compute_reward()` whenever `action_type == "rest"` and the average energy metric is above 30. The behavior stopped.

The second: dump all available money into finances. `finances.liquidity` is the most directly improvable metric, and improving it slightly lowers `mental_wellbeing.stress_level`. The model would spend its entire budget on one domain, achieve a high outcome score on that domain, and ignore everything else. We added `CASCADE_SPREAD_WIDER` — a `-0.30` penalty when the number of metrics that worsened after the step exceeds the number of metrics the agent directly changed. Overfocused actions naturally cause more collateral damage, so this penalty correlates naturally with the exploit.

After those two fixes we had a better reward, but it was still a single number. The problem is that a single reward gives GRPO a very flat training signal — small improvements in one sub-component get averaged with noise everywhere else. We split the reward into 10 independent functions, each signaling a different aspect of behavior.

`reward_plausibility_fn` is the zero-cost exploit gate. It calls `reward_plausibility_check()` from `core/reward.py`, which computes `total_metric_delta / normalized_cost`. If the ratio exceeds 150, the model is claiming massive changes for virtually nothing, and gets `-0.30`. Between 80 and 150 it gets `-0.10`. The normalization (`time/20h`, `money/$500`, `energy/100pts`) makes the check unit-consistent.

`reward_reasoning_fn` calls `reward_reasoning_coherence()` from `core/reward.py`. We added this after observing that the model learned to write structurally valid JSON with nonsensical reasoning strings — "I need to take action because action is necessary" — that satisfied length checks. `reward_reasoning_coherence()` requires both logical connectors ("because", "therefore", "resulting in") AND semantic alignment between the reasoning text and the chosen `action_type`. If you choose `communicate` but your reasoning mentions "cost" and "budget" with no communication words, you get `-0.20`. This cannot be gamed by word-stuffing because the keyword sets are action-specific.

`reward_longterm_fn` is not a decoration. After the model's action is applied, `env.rollout(n_steps=7, gamma=0.9)` runs 7 null rest actions from the post-step state, computes the discounted sum, and uses that as the training signal. It's the only reward function whose gradient explicitly penalizes actions that look fine on day 0 but trigger a cascade collapse by day 4 — usually through stress overflow into sleep, then sleep into motivation, then motivation into career performance. The rollout is deterministic for the same `(completion, prompt)` pair because the entire evaluation, including the rollout, runs under a fixed `eval_seed` drawn from the prompt metadata.

The rollback mechanic is limited to once per episode on purpose. `LifeStackState.used_rollback` tracks it; the `ROLLBACK_USED` penalty in `compute_task_reward()` fires at `-0.10` each time it's used. We wanted rollback to be a recovery tool the model could discover and use judiciously, not a free undo button. Limiting it to once forces the model to commit to its decisions.

---

## The training loop

GRPO in plain terms: generate G completions for each prompt, run all 10 reward functions on each completion, compute each completion's advantage relative to the mean reward of the group, then apply policy gradient weighted by those advantages. No separate critic, no value function. Just "was this completion better or worse than average?"

For LifeStack, each completion is a JSON action object. The environment executes it, the 10 reward functions fire, and gradients update toward the higher-advantage completions.

The tricky part is that the model, after learning to write valid JSON, tends to keep generating for hundreds of tokens of free-form explanation after the closing brace. The prompt says "Return ONLY compact JSON" but Qwen2.5-1.5B ignores that in early training. All 480 max-completion tokens get included in the loss, which means the gradient for the 100 JSON tokens gets diluted across 380 tokens of trailing text that carry no policy information.

`LifeStackGRPOTrainer` — used in episodic training (v3, v4) — overrides `_prepare_inputs()` to find the JSON-object boundary using `_find_json_end_text()` and zero out `completion_mask` beyond that point. The token IDs stay intact so there's no distribution shift, but the KL and advantage terms see only the ~100 JSON tokens. The same technique is why `reward_compact_fn` was removed in v4 — once the JSON boundary masking was in place, `reward_compact_fn` was signal-less (std ≈ 0 throughout the v3 run, confirmed in training logs), so it was pure drag on the reward total.

The 5-stage single-step curriculum (`train_curriculum()`) uses plain `GRPOTrainer`. Stage 1 teaches only JSON structure using 3 reward functions (`reward_format_fn`, `reward_clean_eos_fn`, `reward_route_target_fn`) with learning rate 8e-6. Stages 2–5 introduce all 10 reward functions with learning rates descending 5e-6 → 3e-6 → 2e-6 → 1e-6. Difficulty advances from 1 to 5 across stages whenever stage-end reward is ≥ 0.0. Dataset generation draws from all 8 task domains (`ALL_DOMAINS` in `train_trl.py`) to prevent the model from overfitting to one scenario type.

The episodic curriculum (`train_episodic_curriculum()`) — used for v3 and v4 — asks the model to generate `{"actions": [...]}` sequences instead of single actions. The real environment executes each action in the sequence and accumulates a discounted trajectory reward. This forces the model to plan ahead rather than greedily optimize the immediate step.

---

## What we learned (the honest version)

**What broke first:** The model found `REST_NOT_JUSTIFIED` and immediately started using `action_type="self_care"` to achieve the same thing — self-care has similar resource costs and similar energy recovery effects. We updated the check to cover semantically equivalent action types. Then the model started putting `action_type="rest"` in its reasoning text while using `action_type="communicate"` as the actual type. `reward_reasoning_coherence()` caught this because it checks the reasoning against the *actual* `action_type` field, not the reasoning text.

**The model name inconsistency:** The early README.md said `Llama-3.2-3B` as the base model. This was wrong — training was always on `Qwen2.5-1.5B-Instruct`. The error propagated from a copy-pasted doc template into several places. Every file now reads `Qwen2.5-1.5B-Instruct` consistently, and the `load_model()` function in `train_trl.py` explicitly uses `model_name = "Qwen/Qwen2.5-1.5B-Instruct"` for the HF+PEFT path and `model_name="unsloth/Qwen2.5-1.5B-Instruct"` for the Unsloth path.

**The HuggingFace restart loop:** The Docker image initially ran `python app_flask.py` as `CMD` directly, with the OpenEnv server also trying to start. When `openenv-core` wasn't installed correctly, the server raised `SystemExit`, which killed the container before HF Spaces could mark it healthy. The fix was `start.sh` — which starts `server.py` in the background and runs `app_flask.py` in the foreground — and making `server.py` crash-safe by returning instead of raising `SystemExit` on any failure path. HF health-checks port 7860, so as long as Flask starts, the Space stays alive regardless of what happens to the OpenEnv server.

**What we'd do differently:** The training logs (available at `jdsb06/lifestack-grpo-v4/blob/main/train_run_v4.log`) capture reward per batch but not per reward function. To actually understand which signal is driving improvement, you need per-reward-function tracking via `report_to="tensorboard"`. We added this in v4 but only checked it manually; a proper training dashboard would have shown the `reward_compact_fn` flat line weeks earlier and let us remove it sooner.

The holdout evaluation is weak. The `evaluate_and_plot()` function in `train_trl.py` generates 50 evaluation episodes using the same `TaskGenerator` as training. A proper holdout would use held-out scenario types with novel ExoEvent combinations that the model never saw. Without this it's hard to know whether the policy generalized or memorized conflict patterns.

---

## What the trained model actually learned

The `data/before_after_comparison.json` file records a comparison run: `avg_no_memory: 1.13`, `avg_with_memory: 2.45`, `pct_improvement: 116.81%`. This needs context. These numbers compare the *memory-augmented* agent (with ChromaDB few-shot context from `LifeStackMemory`) against the same model without memory. It is not a trained-vs-untrained comparison.

The qualitative shift is real though. Without memory, the model's most common first action is `delegate` — hand off the problem to someone else. It's a low-cost, low-risk, medium-reward action that works on isolated tasks. With memory injecting past high-reward decisions as few-shot context, the model leads with `communicate` 100% of the time (vs 40% without memory). That's not a coincidence of the test set. `communicate` triggers positive changes in `relationships.*` metrics while also reducing `mental_wellbeing.stress_level` through the `relationships.romantic → mental_wellbeing.emotional_stability` edge. It's the strategically superior opening move for high-relationship-impact conflicts, and the memory retrieval correctly surfaces it.

The shift is meaningful because `delegate` and `communicate` interact differently with the cascade graph. Delegating reduces `career.workload` but costs `relationships.professional_network` slightly — the person is still struggling with relationship metrics on the next turn. Communicating improves relationship metrics immediately, which lowers stress via the graph, which opens up better outcomes on subsequent career and finance actions. The model is learning the cascade graph topology through GRPO, even if it can't explicitly name the edges.

---

## Training progression — what actually improved

Before any fine-tuning, we evaluated the raw `Qwen2.5-1.5B-Instruct` base model on 50 deterministic episodes. Mean reward: **−0.07**. The three worst domains — mental_wellbeing (−0.25), physical_health (−0.17), career (−0.14) — were ones where the base model actively recommended wrong actions that made things worse.

| Run | What changed | Eval reward | vs baseline |
|-----|-------------|------------|-------------|
| Base (no LoRA) | — | −0.07 | — |
| Run 1 | — (JSON parser broken, clipped_ratio=1.0) | −0.47 | **−571%** |
| Run 2 | Shorter completions | −0.41 | −486% |
| **Run 3** | **Greedy regex extraction** | **−0.010** | **+85.7%** |
| Run 4 → v1 | 5-stage curriculum | −0.100 | −42.9% |
| v3 episodic | Horizon=3, EOS-aware | +0.140 ep. return | new capability |
| **v4 episodic** | Dead weight removed, ep. return ×2 | **0.856 peak** | new capability |

Run 1 and 2 failed because of the JSON trailing-prose bug described above. Run 3 fixed it with one line. Run 4 (v1) was more consistent but didn't outperform Run 3 on mean reward — the curriculum plateaued at difficulty 1 because the advance threshold (reward ≥ 0.6) was never reached. v3 and v4 are not directly comparable to the baseline because they measure episode return across 3-step sequences; the correct framing is that they add a capability (multi-step planning) that the baseline does not have at all.

`frac_zero_std = 0%` throughout v4 training means every GRPO group had real reward variance at every step — real gradient, real learning.

---

## What you can see in the live demo

The [HF Space](https://huggingface.co/spaces/jdsb06/meta-r2) runs v4 on a T4 GPU with six interactive tabs:

**Personality Lab** — run the same crisis through different Big Five (OCEAN) personality profiles. The same objective crisis gets different recommended actions depending on whether the person is high-conscientiousness/high-neuroticism (push through and communicate) vs low-agreeableness/low-conscientiousness (rest and offload).

**What-If Lab** — v4 proposes an action, then generates three counterfactual alternatives (`rest`, `negotiate`, `delegate`). All alternatives come from the trained model — no Groq fallback. This shows the policy's range, not just its mode.

**Untrained vs GRPO-Trained** — side-by-side: vanilla Groq 70B vs v4 adapter on the same prompt. The trained model consistently picks more targeted actions with meaningful resource cost specifications; the untrained model tends toward generic advice.

**Model Evolution (v1→v4)** — all four model versions loaded simultaneously, responding to the same scenario. Policy shift is visible: v2 starts delegating where v1 rests; v4 reasons about resource depletion across cascade steps that v1 ignores entirely.

**Longitudinal Memory** — ChromaDB retrieval of past successful trajectories for the same personality type. After enough interactions the agent starts citing its own history: *"Last time you cancelled plans without warning, it took 4 days to recover. Communicate first."*

**Live Simulation** — real-time cascade animation across the dependency graph with the agent proposing interventions at each step.

---

## What's next

Multi-turn GRPO with per-step process rewards. Currently the training signals are mostly episode-level (did the task succeed?) or single-step (did the current action look reasonable?). Process supervision — giving the model credit for each intermediate step that moved toward a milestone — would let us train on 15-turn episodes without the gradient signal being dominated by terminal outcomes.

Procedurally generated conflict scenarios (RLVE-style). The `agent/conflict_generator.py` templates are hand-authored. A procedural generator that varies domain co-occurrence, crisis severity distributions, and ExoEvent timing would dramatically increase training diversity and make the generalization problem much harder to fake.

Larger base model. Qwen2.5-1.5B is at the low end for structured JSON output with multi-step reasoning. The constraint was VRAM budget during the hackathon. Qwen2.5-7B-Instruct with the same LoRA setup would likely show a significant jump in `reward_reasoning_fn` scores, since the model would be better at maintaining logical consistency across a complex multi-domain prompt.
