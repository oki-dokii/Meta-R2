# Building LifeStack: An RL Environment Where the Agent Manages a Human Life

---

## The itch we couldn't ignore

Most RL environments are toy problems. A maze. Chess. A robot arm picking up a block. We don't say this dismissively — those problems are legitimately hard — but they all share something: the state is clean, actions are narrow, and if something goes wrong you can see it immediately.

Real life doesn't work like that. When you take on an extra project at work, you don't get a HUD showing `career.workload += 30`. What you get, three weeks later, is your sleep getting worse because stress crept up. You notice your partner is quieter than usual. Your savings are lower than they should be because you've been ordering delivery every night to save time. None of these are isolated. They're causally entangled, temporally displaced, and completely invisible until they've already cascaded.

We wanted to build an environment that actually captured this. Not one that simulates messy decisions with hand-waved heuristics, but one where the causal structure is explicit, the resources are genuinely finite, and you cannot win by optimizing one thing without watching something else fall apart.

That's LifeStack. And fair warning: the reward function nearly broke us.

---

## What LifeStack actually is

A person wakes up to a life crisis. In the concrete terms of `data/conflicts.json`, this could be "The Perfect Storm" — `career.workload` spikes by 30 points (a project deadline moved up), `finances.liquidity` drops 40 points (unexpected car repair), and `relationships.romantic` is already fragile from two weeks of working late. The agent gets $500 and 20 hours. That's the budget. Resolve the crisis without any metric hitting 10 — that's the floor, and hitting it ends the episode as a failure.

The 23 metrics across 6 life domains (`core/life_state.py`) aren't independent. They're connected through a `DependencyGraph` with 32 directed edges and a 0.6 dampening factor per propagation hop. We pulled the dampening value from Starcke & Brand (2012) — stress effects on decision-making attenuate roughly 40% per cognitive or behavioral hop — so `cascade_utils.py` uses `CASCADE_DAMPENING_DEFAULT = 0.6` as the per-edge decay.

Concretely: increase `career.workload` by 20 points and the cascade graph automatically raises `mental_wellbeing.stress_level` by `20 × 0.70 × 0.6 = 8.4` points. That stress increase then reduces `physical_health.sleep_quality` by `8.4 × (-0.55) × 0.6 = -2.77`. That sleep hit reduces `mental_wellbeing.clarity` by another `2.77 × 0.60 × 0.6 = -1.0`. Third-order effects are small individually, but they compound across a 30-step episode.

The agent can't prevent any of this. It can only decide which cascades are worth triggering by choosing where to spend resources first.

Two more layers sit on top of the cascade. `ExoEvent` objects fire at scheduled or random steps, mutating world state and sometimes closing routes the agent was planning to use. The default `FlightCrisisTask` has a `price_surge` event at step 5 that makes one route more expensive, and a `lounge_full` event at step 8 that closes a recovery route entirely. The agent can see that events exist from `obs.metadata["events"]`, but not their content until they fire. There's also a `hidden_state` — things like `{"card_available": True}` — that the agent can only discover by spending an action on `action_type="inspect"`.

---

## The reward design rabbit hole

We started with the simplest possible reward: sum of metric improvements. This lasted about four hours.

The model found two exploits almost immediately.

The first was rest spam. `action_type="rest"` has near-zero resource cost and recovers physical energy. Physical energy connects to `mental_wellbeing.motivation` via the dependency graph, which connects weakly to `career.growth_trajectory`. The model figured this out: rest constantly, watch several metrics slowly drift positive, never spend anything, never risk a cascade. Reward goes up. Episode never terminates. No real task being solved. We added `REST_NOT_JUSTIFIED` — a `-0.25` penalty in `compute_reward()` whenever `action_type == "rest"` and average energy is already above 30. That stopped it.

The second was finance dumping. `finances.liquidity` is the most directly improvable metric, and improving it slightly lowers stress. The model would spend its entire budget on finances, get a high score in that domain, and ignore everything else. We added `CASCADE_SPREAD_WIDER` — a `-0.30` penalty when the number of metrics that worsened exceeds the number the agent directly changed. Overfocused actions cause more collateral damage, so this penalty correlates naturally with the exploit.

After fixing those two, we had a better reward but still just a single number. A single reward gives GRPO a flat training signal — small improvements in one sub-component get averaged with noise across everything else. We split the reward into 10 independent functions.

`reward_plausibility_fn` is the zero-cost exploit gate. It calls `reward_plausibility_check()`, which computes `total_metric_delta / normalized_cost`. If the ratio exceeds 150, the model is claiming massive improvements for almost nothing, and gets `-0.30`. Between 80 and 150, it gets `-0.10`. The normalization (`time/20h`, `money/$500`, `energy/100pts`) makes this unit-consistent.

`reward_reasoning_fn` came from a specific observation: the model learned to write structurally valid JSON with reasoning strings like *"I need to take action because action is necessary"* — long enough to pass length checks, semantically empty. `reward_reasoning_coherence()` requires both logical connectors ("because", "therefore", "resulting in") AND semantic alignment between the reasoning text and the chosen `action_type`. If you pick `communicate` but your reasoning only mentions budget and cost, you get `-0.20`. This can't be gamed by word-stuffing because the keyword sets are action-specific.

`reward_longterm_fn` was the one that needed the most thought. After the model's action is applied, `env.rollout(n_steps=7, gamma=0.9)` runs 7 null rest actions from the post-step state and uses the discounted sum as the training signal. It's the only reward function whose gradient explicitly penalizes actions that look fine on day 0 but trigger a cascade collapse by day 4 — usually through stress overflow into sleep, then sleep into motivation, then motivation into career performance. The rollout is deterministic for the same `(completion, prompt)` pair because everything runs under a fixed `eval_seed` drawn from the prompt metadata.

The rollback mechanic is limited to once per episode on purpose. `LifeStackState.used_rollback` tracks it; the `ROLLBACK_USED` penalty fires at `-0.10` each time. We wanted rollback to be something the model could discover and use carefully, not a free undo button.

---

## The training loop

GRPO in plain terms: generate G completions for each prompt, run all 10 reward functions on each, compute each completion's advantage relative to the group mean, then apply policy gradient weighted by those advantages. No critic, no value function. Just "was this completion better or worse than average?"

For LifeStack, each completion is a JSON action object. The environment executes it, the 10 reward functions fire, and gradients update toward the higher-advantage completions.

The tricky part was: after the model learned to write valid JSON, it kept generating hundreds of tokens of free-form explanation after the closing brace. The prompt says "Return ONLY compact JSON" but `Qwen2.5-1.5B` ignores that in early training. All 480 max-completion tokens get included in the loss, which means the gradient for the ~100 JSON tokens gets diluted across 380 tokens of trailing text that carry no policy information.

`LifeStackGRPOTrainer` — used in episodic training (v3, v4) — overrides `_prepare_inputs()` to find the JSON-object boundary using `_find_json_end_text()` and zero out `completion_mask` beyond that point. Token IDs stay intact so there's no distribution shift, but the KL and advantage terms see only the JSON tokens. This is also why `reward_compact_fn` was removed in v4 — once JSON boundary masking was in place, `reward_compact_fn` had std ≈ 0 throughout v3 training (confirmed in the logs), so it was pure drag on the reward total.

The 5-stage single-step curriculum (`train_curriculum()`) uses plain `GRPOTrainer`. Stage 1 teaches only JSON structure with 3 reward functions (`reward_format_fn`, `reward_clean_eos_fn`, `reward_route_target_fn`) at learning rate 8e-6. Stages 2–5 introduce all 10 reward functions with learning rates descending 5e-6 → 3e-6 → 2e-6 → 1e-6. Difficulty advances from 1 to 5 across stages when stage-end reward is ≥ 0.0. Dataset generation draws from all 8 task domains in `ALL_DOMAINS` to prevent the model from overfitting to one scenario type.

The episodic curriculum (`train_episodic_curriculum()`) — used for v3 and v4 — asks the model to generate `{"actions": [...]}` sequences instead of single actions. The real environment executes each action in the sequence and accumulates a discounted trajectory reward. This forces the model to plan ahead rather than greedily optimize each individual step.

---

## What we actually learned (no coating of any kind)

**What broke first:** The model found `REST_NOT_JUSTIFIED` and immediately pivoted to `action_type="self_care"` — same resource cost, similar recovery effects. We updated the check to cover semantically equivalent action types. Then it started putting `action_type="rest"` in the reasoning text while using `action_type="communicate"` as the actual type field. `reward_reasoning_coherence()` caught this because it checks reasoning against the *actual* `action_type` field, not the reasoning text.

**The model name inconsistency:** Early docs said `Llama-3.2-3B` as the base model. This was wrong — training was always on `Qwen2.5-1.5B-Instruct`. The error came from a copy-pasted doc template and propagated into several places. Everything now consistently reads `Qwen2.5-1.5B-Instruct`, and `load_model()` in `train_trl.py` explicitly uses `model_name = "Qwen/Qwen2.5-1.5B-Instruct"` for the HF+PEFT path and `"unsloth/Qwen2.5-1.5B-Instruct"` for the Unsloth path.

**The HF restart loop:** The Docker image initially ran `python app_flask.py` as `CMD` directly, with the OpenEnv server also trying to start. When `openenv-core` wasn't installed correctly, the server raised `SystemExit`, which killed the container before HF Spaces could mark it healthy. The fix was `start.sh` — which starts `server.py` in the background and runs `app_flask.py` in the foreground — and making `server.py` crash-safe by returning instead of raising on any failure path. HF health-checks port 7860, so as long as Flask starts, the Space stays alive regardless of what happens to the OpenEnv server.

**What we'd do differently:** Training logs capture reward per batch but not per reward function. To understand which signal is actually driving improvement, you need per-reward-function tracking via `report_to="tensorboard"`. We added this in v4 but only checked it manually — a proper training dashboard would have shown the flat `reward_compact_fn` line weeks earlier and let us remove it sooner.

The holdout evaluation is also weak. `evaluate_and_plot()` in `train_trl.py` generates 50 evaluation episodes using the same `TaskGenerator` as training. A real holdout would use held-out scenario types with novel ExoEvent combinations the model never saw during training. Without that it's genuinely hard to say whether the policy generalized or just memorized conflict patterns.

---

## What the model actually learned

`data/before_after_comparison.json` records: `avg_no_memory: 1.13`, `avg_with_memory: 2.45`, `pct_improvement: 116.81%`. This needs context — it compares the memory-augmented agent (with ChromaDB few-shot context from `LifeStackMemory`) against the same model without memory. It is not a trained-vs-untrained comparison.

The qualitative shift is real though. Without memory, the model's most common first action is `delegate` — hand the problem off to someone else. Low-cost, low-risk, medium-reward. Works on isolated tasks. With memory injecting past high-reward decisions as few-shot context, the model leads with `communicate` 100% of the time vs 40% without memory. That's not a test set coincidence.

`communicate` triggers positive changes in `relationships.*` metrics while also reducing `mental_wellbeing.stress_level` through the `relationships.romantic → mental_wellbeing.emotional_stability` edge. It's the strategically superior opening move for high-relationship-impact conflicts, and the memory retrieval correctly surfaces it.

The cascade graph explains why this matters. Delegating reduces `career.workload` but costs `relationships.professional_network` slightly — the agent is still stuck with relationship problems on the next turn. Communicating improves relationship metrics immediately, which lowers stress via the graph, which opens better outcomes on career and finance actions later. The model is learning the cascade graph topology through GRPO, even without being able to name the edges explicitly.

---

## Training progression — what actually changed

Before any fine-tuning: raw `Qwen2.5-1.5B-Instruct` on 50 deterministic episodes. Mean reward: **−0.07**. The three worst domains — mental_wellbeing (−0.25), physical_health (−0.17), career (−0.14) — were ones where the base model actively recommended actions that made things worse.

| Run | What changed | Eval reward | vs baseline |
|-----|-------------|------------|-------------|
| Base (no LoRA) | — | −0.07 | — |
| Run 1 | — (JSON parser broken, clipped_ratio=1.0) | −0.47 | **−571%** |
| Run 2 | Shorter completions | −0.41 | −486% |
| **Run 3** | **Greedy regex extraction** | **−0.010** | **+85.7%** |
| Run 4 → v1 | 5-stage curriculum | −0.100 | −42.9% |
| v3 episodic | Horizon=3, EOS-aware | +0.140 ep. return | new capability |
| **v4 episodic** | Dead weight removed, ep. return ×2 | **0.856 peak** | new capability |

Runs 1 and 2 failed because of the JSON trailing-prose problem described above. Run 3 fixed it with one line of regex. Run 4 (v1) was more consistent but didn't outperform Run 3 on mean reward — the curriculum stalled at difficulty 1 because the advance threshold (reward ≥ 0.6) was never reached. v3 and v4 aren't directly comparable to the single-step baseline because they measure episode return across 3-step sequences; the correct framing is they add a capability — multi-step planning — that the baseline doesn't have at all.

`frac_zero_std = 0%` throughout v4 training means every GRPO group had real reward variance at every step — real gradient, real learning.

---

## What you can do in the live demo

The [HF Space](https://huggingface.co/spaces/jdsb06/meta-r2) runs v4 on a T4 GPU with six interactive tabs:

**Personality Lab** — run the same crisis through different Big Five (OCEAN) personality profiles. High-conscientiousness/high-neuroticism gets "push through and communicate." Low-agreeableness/low-conscientiousness gets "rest and offload." Same objective crisis, different recommended actions depending on who's facing it.

**What-If Lab** — v4 proposes an action, then generates three counterfactual alternatives (`rest`, `negotiate`, `delegate`). All from the trained model, no Groq fallback. This shows the policy's range, not just its mode.

**Untrained vs GRPO-Trained** — side-by-side: vanilla Groq 70B versus v4 adapter on the same prompt. The trained model picks more targeted actions with meaningful resource cost specifications. The untrained model tends toward generic advice.

**Model Evolution (v1→v4)** — all four model versions loaded simultaneously, responding to the same scenario. v2 starts delegating where v1 rests. v4 reasons about resource depletion across cascade steps that v1 ignores entirely.

**Longitudinal Memory** — ChromaDB retrieval of past successful trajectories for the same personality type. After enough interactions the agent starts citing its own history: *"Last time you cancelled plans without warning, it took 4 days to recover. Communicate first."*

**Live Simulation** — real-time cascade animation across the dependency graph with the agent proposing interventions at each step.

---

## What's next

Multi-turn GRPO with per-step process rewards. Right now the training signals are mostly episode-level (did the task succeed?) or single-step (did the current action look reasonable?). Process supervision — giving the model credit for each intermediate step that moved toward a milestone — would let us train on 15-turn episodes without the gradient signal being dominated by terminal outcomes.

Procedurally generated conflict scenarios. The `agent/conflict_generator.py` templates are hand-authored. A procedural generator that varied domain co-occurrence, crisis severity distributions, and ExoEvent timing would dramatically increase training diversity and make the generalization problem much harder to fake.

Larger base model. `Qwen2.5-1.5B` is at the low end for structured JSON output with multi-step reasoning — the VRAM budget during the hackathon was the constraint. `Qwen2.5-7B-Instruct` with the same LoRA setup would likely show a significant jump in `reward_reasoning_fn` scores, since that model would be better at maintaining logical consistency across a complex multi-domain prompt.
