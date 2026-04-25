"""
train_trl.py — LifeStack GRPO Training via HuggingFace TRL + Unsloth

Trains a small LLM (Qwen2.5-1.5B-Instruct) to resolve daily-life conflicts
across 8 domains using Group Relative Policy Optimization (GRPO).

Supported domains:
    career, finances, relationships, physical_health,
    mental_wellbeing, time, flight_crisis, code_merge_crisis

Usage (Colab / GPU):
    !pip install unsloth trl datasets transformers accelerate
    !python train_trl.py                       # full curriculum (5 stages)
    !python train_trl.py --dry-run             # 1-step smoke test (CPU OK)
"""

import json
import os
import copy
import random
import numpy as np
import types
import sys
import importlib.machinery

# ── EARLY PATCHES ─────────────────────────────────────────
# Unsloth MUST be imported before transformers/trl to apply its patches.
# It ALSO globally monkey-patches trl.GRPOTrainer with UnslothGRPOTrainer,
# which then expects an Unsloth-patched model and calls model.for_training()
# at runtime. So if we want the plain HF+PEFT path to work, we must skip this
# import entirely — otherwise we end up with a plain HF model running through
# Unsloth's patched trainer → AttributeError on for_training().
if os.environ.get("LIFESTACK_NO_UNSLOTH", "").lower() in ("1", "true", "yes"):
    print("[early-init] LIFESTACK_NO_UNSLOTH set → skipping `import unsloth` "
          "to keep trl.GRPOTrainer unpatched.")
else:
    try:
        import unsloth
    except Exception as e:
        # Colab environments can fail inside unsloth import with non-ImportError
        # exceptions (for example NameError from incompatible dependency combos).
        print(f"[warning] Unsloth import failed, continuing with HF fallback: {e}")

def _install_trl_optional_dependency_shims() -> None:
    """
    TRL GRPO imports callbacks that can hard-import optional packages like
    `mergekit` and `llm_blender` even when GRPO doesn't use those paths.
    Install lightweight shims so training remains runnable on Colab/Kaggle.
    """
    # Always install shims before importing TRL.
    # This avoids failures from incompatible optional dependency versions.
    mergekit_mod = types.ModuleType("mergekit")
    mergekit_mod.__path__ = []  # mark as package
    mergekit_config_mod = types.ModuleType("mergekit.config")
    mergekit_merge_mod = types.ModuleType("mergekit.merge")

    class MergeConfiguration:  # noqa: D401
        """Compatibility placeholder for TRL optional mergekit import."""

        @classmethod
        def model_validate(cls, data):
            return data

    class MergeOptions:  # noqa: D401
        """Compatibility placeholder for TRL optional mergekit import."""

        def __init__(self, *args, **kwargs):
            pass

    def run_merge(*args, **kwargs):
        return None

    mergekit_config_mod.MergeConfiguration = MergeConfiguration
    mergekit_merge_mod.MergeOptions = MergeOptions
    mergekit_merge_mod.run_merge = run_merge
    mergekit_mod.config = mergekit_config_mod
    mergekit_mod.merge = mergekit_merge_mod
    mergekit_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit", loader=None)
    mergekit_config_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit.config", loader=None)
    mergekit_merge_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit.merge", loader=None)
    sys.modules["mergekit"] = mergekit_mod
    sys.modules["mergekit.config"] = mergekit_config_mod
    sys.modules["mergekit.merge"] = mergekit_merge_mod

    llm_blender_mod = types.ModuleType("llm_blender")

    class Blender:  # noqa: D401
        """Compatibility placeholder for TRL optional llm_blender import."""

        def __init__(self, *args, **kwargs):
            pass

        def rank(self, *args, **kwargs):
            return [0]

        def score(self, *args, **kwargs):
            return [0.0]

    llm_blender_mod.Blender = Blender
    llm_blender_mod.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)
    sys.modules["llm_blender"] = llm_blender_mod
    # vLLM is optional for GRPO; provide import-safe shim for environments
    # where import checks pass but real import fails due incomplete installs.
    vllm_mod = types.ModuleType("vllm")

    class SamplingParams:  # noqa: D401
        """Compatibility placeholder for TRL optional vllm import."""

        def __init__(self, *args, **kwargs):
            pass

    class LLM:  # noqa: D401
        """Compatibility placeholder for TRL optional vllm import."""

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return []

    vllm_mod.SamplingParams = SamplingParams
    vllm_mod.LLM = LLM
    vllm_mod.__version__ = "0.11.1"
    vllm_mod.__path__ = []  # mark as package for TRL optional vLLM imports
    vllm_mod.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)
    sys.modules["vllm"] = vllm_mod

    vllm_dist_mod = types.ModuleType("vllm.distributed")
    vllm_dist_mod.__path__ = []
    vllm_device_mod = types.ModuleType("vllm.distributed.device_communicators")
    vllm_device_mod.__path__ = []
    vllm_pynccl_mod = types.ModuleType("vllm.distributed.device_communicators.pynccl")

    class PyNcclCommunicator:  # noqa: D401
        """Compatibility placeholder for TRL optional vLLM client imports."""

        def __init__(self, *args, **kwargs):
            pass

    vllm_pynccl_mod.PyNcclCommunicator = PyNcclCommunicator
    vllm_dist_mod.__spec__ = importlib.machinery.ModuleSpec("vllm.distributed", loader=None)
    vllm_device_mod.__spec__ = importlib.machinery.ModuleSpec("vllm.distributed.device_communicators", loader=None)
    vllm_pynccl_mod.__spec__ = importlib.machinery.ModuleSpec("vllm.distributed.device_communicators.pynccl", loader=None)
    sys.modules["vllm.distributed"] = vllm_dist_mod
    sys.modules["vllm.distributed.device_communicators"] = vllm_device_mod
    sys.modules["vllm.distributed.device_communicators.pynccl"] = vllm_pynccl_mod

    vllm_tf_mod = types.ModuleType("vllm.transformers_utils")
    vllm_tf_mod.__path__ = []
    vllm_tok_mod = types.ModuleType("vllm.transformers_utils.tokenizer")

    def cached_tokenizer(tokenizer, *args, **kwargs):
        return tokenizer

    vllm_tok_mod.cached_tokenizer = cached_tokenizer
    vllm_tf_mod.__spec__ = importlib.machinery.ModuleSpec("vllm.transformers_utils", loader=None)
    vllm_tok_mod.__spec__ = importlib.machinery.ModuleSpec("vllm.transformers_utils.tokenizer", loader=None)
    sys.modules["vllm.transformers_utils"] = vllm_tf_mod
    sys.modules["vllm.transformers_utils.tokenizer"] = vllm_tok_mod

    # Some HF Spaces images include a partial vLLM install whose package
    # metadata reports version "N/A" which crashes `packaging.Version("N/A")`.
    # Raise PackageNotFoundError instead — TRL catches that and treats vLLM
    # as absent, so is_vllm_available() returns False and VLLMGeneration is
    # never imported regardless of how many vLLM submodules TRL would need.
    try:
        import importlib.metadata as _importlib_metadata
        from importlib.metadata import PackageNotFoundError as _PkgNF

        _original_version = _importlib_metadata.version

        def _patched_version(package_name):
            if str(package_name).lower() == "vllm":
                raise _PkgNF("vllm")
            return _original_version(package_name)

        _importlib_metadata.version = _patched_version
    except Exception:
        pass
    print("[warning] using local shims for mergekit/llm_blender compatibility.")


_install_trl_optional_dependency_shims()

# Belt-and-suspenders: some TRL builds cache is_vllm_available at module-load
# time before our metadata patch takes effect. Force False directly.
try:
    import trl.import_utils as _trl_iu
    _trl_iu.is_vllm_available = lambda: False
    for _attr in ("_vllm_version",):
        if hasattr(_trl_iu, _attr):
            setattr(_trl_iu, _attr, None)
    del _trl_iu, _attr
except Exception:
    pass

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Fix for TRL 0.15.1 + Transformers 4.56.2 incompatibility with _get_train_sampler
import inspect
_original_get_train_sampler = GRPOTrainer._get_train_sampler
def _patched_get_train_sampler(self, *args, **kwargs):
    sig = inspect.signature(_original_get_train_sampler)
    if len(sig.parameters) == 1:
        return _original_get_train_sampler(self)
    return _original_get_train_sampler(self, *args, **kwargs)
GRPOTrainer._get_train_sampler = _patched_get_train_sampler

# ──────────────────────────────────────────────────────────────────────────────
# TRL version-safe GRPOConfig constructor
# ──────────────────────────────────────────────────────────────────────────────

def _make_grpo_config(**kwargs) -> GRPOConfig:
    """
    Build a GRPOConfig while silently dropping kwargs not supported by the
    installed TRL version.

    Background: GRPOConfig fields differ across TRL minor versions.
    For example, `max_prompt_length` was added mid-1.x; older TRL 1.2.0
    wheels on Python 3.10 (HF Spaces) do not have it, causing TypeError.
    This wrapper makes the training script compatible with all 1.x builds.
    """
    import dataclasses
    supported = {f.name for f in dataclasses.fields(GRPOConfig)}
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[trl-compat] GRPOConfig: skipping unsupported fields for this TRL build: {dropped}")
    return GRPOConfig(**filtered)


def _dtype_flags(model=None) -> tuple[bool, bool]:
    """
    Return (bf16, fp16) for GRPOConfig based on the ACTUAL loaded model dtype.

    Why the float16 case returns (False, False):
      - fp16=True  → Accelerate creates a GradScaler. GradScaler.unscale_() requires
                     float32 gradients, but Unsloth's LoRA kernels produce float16
                     gradients → "Attempting to unscale FP16 gradients" crash.
      - bf16=True  → Unsloth's patched trainer raises TypeError when the model's
                     compute dtype is float16 but bf16=True is requested.
      - (False, False) → No AMP / no GradScaler. Model forward pass runs in float16
                     (LoRA matrices are float16), gradients are float16, the small
                     LR (≈3e-6) prevents overflow. Unsloth's smart gradient
                     offloading still applies. Safe on all Ampere+ GPUs.
    """
    if model is not None:
        try:
            for p in model.parameters():
                if p.dtype == torch.bfloat16:
                    return True, False
                if p.dtype == torch.float16:
                    # fp16=True  → GradScaler → "Attempting to unscale FP16 gradients"
                    # bf16=True  → Unsloth TypeError (float16 model vs bf16 config)
                    # (False, False) → no AMP; Unsloth handles precision internally
                    return False, False
                # 4-bit params have a quant_state; check their compute dtype
                qs = getattr(p, "quant_state", None)
                if qs is not None:
                    cdtype = getattr(qs, "dtype", None)
                    if cdtype == torch.bfloat16:
                        return True, False
                    if cdtype == torch.float16:
                        return False, False
                break
        except Exception:
            pass
    # Fallback: Ampere+ → bfloat16; older → no AMP (fp16 GradScaler breaks Unsloth)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        return (cap >= 8, False)
    return False, False


# ──────────────────────────────────────────────────────────────────────────────
# JSON boundary helper — used by reward_compact_fn AND LifeStackGRPOTrainer
# ──────────────────────────────────────────────────────────────────────────────

def _find_json_end_text(text: str) -> int | None:
    """
    Return the character index just AFTER the closing `}` of the first complete
    top-level JSON object in `text`, or None if no complete object is found.

    Uses json.JSONDecoder.raw_decode to avoid reimplementing the full grammar.
    Strips markdown fences before searching.
    """
    import re as _re
    inner = text
    if "```json" in inner:
        inner = inner.split("```json")[-1].split("```")[0]
    elif "```" in inner:
        inner = inner.split("```")[-1].split("```")[0]
    dec = json.JSONDecoder()
    for m in _re.finditer(r"\{", inner):
        try:
            _, end_idx = dec.raw_decode(inner[m.start():].lstrip())
            return m.start() + end_idx
        except json.JSONDecodeError:
            continue
    return None


class LifeStackGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer that masks completion tokens after the first complete JSON object.

    Problem: the model writes valid JSON then continues for hundreds of tokens of
    free-form explanation, always hitting max_completion_length (480 tokens).  The
    GRPO gradient is diluted across all 480 tokens when only the first ~100 carry
    any meaningful policy information.

    Fix: after the parent's generation + reward pass, we re-decode each completion,
    locate the JSON-object boundary, and zero-out completion_mask beyond that point.
    The token IDs stay intact (no distribution shift), only the mask changes, so:
      - ref_per_token_logps already computed — masked tokens contribute 0 to the KL term
      - per_token_loss already computed   — masked tokens contribute 0 to the advantage term
      - effective completion length drops to ~100 tokens → 3–5× sharper gradient signal

    This does NOT change generation behaviour (the model still emits 480 tokens).
    The model gradually learns to stop because shorter completions receive higher
    reward (reward_compact_fn) while their gradient is full-strength; longer ones
    receive lower reward AND have most of their gradient zeroed out.
    """

    def _prepare_inputs(self, inputs):
        result = super()._prepare_inputs(inputs)

        completion_ids = result["completion_ids"]           # (B, L)
        completion_mask = result["completion_mask"].clone() # (B, L)
        eos_id = self.processing_class.eos_token_id

        # Rows that already contain a real EOS need no intervention.
        already_done = (completion_ids == eos_id).any(dim=1)

        for i in range(completion_ids.size(0)):
            if already_done[i]:
                continue
            text = self.processing_class.decode(
                completion_ids[i], skip_special_tokens=True
            )
            end_char = _find_json_end_text(text)
            if end_char is None or end_char >= len(text) - 3:
                continue  # JSON fills (nearly) the whole completion — nothing to trim

            # Estimate the token boundary.  Re-encoding the prefix gives a good
            # approximation; off-by-1 is acceptable because we only touch the mask.
            prefix_ids = self.processing_class.encode(
                text[:end_char], add_special_tokens=False
            )
            cutoff = min(len(prefix_ids) + 1, completion_mask.size(1))
            completion_mask[i, cutoff:] = 0

        out = dict(result)
        out["completion_mask"] = completion_mask
        return out

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LifeStack imports
from core.life_state import LifeMetrics, ResourceBudget, DependencyGraph
from core.reward import compute_reward
from agent.conflict_generator import generate_conflict, TEMPLATES, TaskGenerator
from intake.simperson import SimPerson
from core.task import Task, FlightCrisisTask


def _tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_trl_model_compat(model):
    """Patch small TRL/Transformers compatibility fields onto wrapped models."""
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model


# ──────────────────────────────────────────────
# 1. MODEL SETUP (Unsloth for 4-bit efficiency)
# ──────────────────────────────────────────────

def load_model():
    """Load model for GRPO training.

    Two paths, controlled by the LIFESTACK_NO_UNSLOTH env var:

      LIFESTACK_NO_UNSLOTH=1  → plain HF + PEFT (RECOMMENDED on A100 80GB)
      anything else (default) → try Unsloth, fall back to HF+PEFT on failure

    Why the env-var opt-out exists:
      Unsloth's pre-quantized checkpoint
      `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit` has
      bnb_4bit_compute_dtype=float16 baked in. The 4-bit dequantization step
      always emits float16 activations during forward, regardless of the
      `dtype=` parameter. That collides with the LoRA matrices' dtype:
        - LoRA float32 → "self and mat2 must have the same dtype, Half vs Float"
        - LoRA float16 → Unsloth auto-enables fp16 mode → GradScaler crash
        - LoRA bf16   → still float16 activations → same Half/BFloat16 mismatch
      No dtype combination satisfies all three constraints simultaneously on
      this torch / transformers / TRL / Unsloth stack. The HF+PEFT fallback
      is plain bf16 end-to-end with no kernel surprises and runs comfortably
      on a single A100 80GB (1.5B params × 2 bytes ≈ 3 GB).

    dtype choice: bfloat16 on Ampere+ (A100/H100), float16 otherwise.
    bf16 has float32-equivalent dynamic range so no GradScaler is needed.
    """
    _load_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )

    # Explicit opt-out: skip Unsloth entirely. Stable, predictable, no kernel hell.
    if os.environ.get("LIFESTACK_NO_UNSLOTH", "").lower() in ("1", "true", "yes"):
        print("[load_model] LIFESTACK_NO_UNSLOTH set → using plain HF+PEFT path.")
        return _load_model_hf_peft(_load_dtype)

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct",
            max_seq_length=1024,
            dtype=_load_dtype,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        for _, param in model.named_parameters():
            if param.requires_grad and param.dtype == torch.float32:
                param.data = param.data.to(_load_dtype)
        return _ensure_trl_model_compat(model), tokenizer
    except Exception as e:
        print(f"[warning] Unsloth model load failed, using HF+PEFT fallback: {e}")
        return _load_model_hf_peft(_load_dtype)


def _load_model_hf_peft(_load_dtype: "torch.dtype"):
    """Plain HF + PEFT LoRA loader. Stable, predictable, no kernel surprises."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_load_dtype, device_map="auto"
    )
    # Enable gradient checkpointing for memory efficiency on the full bf16 model.
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception:
        pass
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model = _ensure_trl_model_compat(model)
    model.print_trainable_parameters()
    return model, tokenizer


def load_model_for_dry_run():
    """
    Tiny CPU-friendly model used only for --dry-run pipeline validation.
    Keeps dry-run fast and avoids downloading multi-GB checkpoints locally.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
    )
    model = _ensure_trl_model_compat(model)
    model.eval()
    print(f"  Using tiny dry-run model: {model_name}")
    return model, tokenizer


# ──────────────────────────────────────────────
# 2. DATASET: Generate conflict prompts
# ──────────────────────────────────────────────

def build_prompt_for_task(task, person, metrics, budget, seed=42, step=0, event_descriptions=None):
    """Build a compact prompt from task state while preserving reward metadata."""
    flat = metrics.flatten()

    # Keep only 5 high-signal metrics to fit prompt+completion in a tight token budget.
    metric_priority = [
        "career.workload",
        "finances.liquidity",
        "relationships.romantic",
        "physical_health.energy",
        "mental_wellbeing.stress_level",
        "time.free_hours_per_week",
        "time.commute_burden",
    ]
    key_metrics = [k for k in metric_priority if k in flat][:5]
    if len(key_metrics) < 5:
        for k in flat:
            if k not in key_metrics:
                key_metrics.append(k)
            if len(key_metrics) == 5:
                break
    metrics_str = "\n".join(f"- {k}: {flat[k]:.1f}" for k in key_metrics)

    event_context = ""
    if event_descriptions:
        recent = event_descriptions[-2:]
        compact_events = [e[:140] for e in recent]
        event_context = "\nRecent events:\n" + "\n".join(f"- {e}" for e in compact_events)

    listed_route_ids = [r.id for r in task.viable_routes[:2]]

    # Keep SYSTEM_METADATA for reward reconstruction.
    metadata = {
        "domain": task.domain,
        "disruption": task.mutable_world,
        "difficulty": task.difficulty,
        "seed": seed,
        "step": step,
        "route_ids": listed_route_ids,
        "budget": {
            "time": budget.time_hours,
            "money": budget.money_dollars,
            "energy": budget.energy_units
        }
    }
    metadata_str = json.dumps(metadata, separators=(",", ":"))

    # Cap routes to 2 to keep the context short but actionable.
    routes_str = "\n".join(
        f"- {r.id}: {r.name} (needs {', '.join(r.required_action_types[:2])})"
        for r in task.viable_routes[:2]
    )
    if not routes_str:
        routes_str = "- none"

    return (
        "You are LifeStack. Return ONLY compact JSON.\n"
        f"<SYSTEM_METADATA>\n{metadata_str}\n</SYSTEM_METADATA>\n"
        f"Task: {task.goal}\n"
        f"Story: {task.domain_metadata.get('story', '')[:160]}\n"
        f"Key metrics:\n{metrics_str}\n"
        f"Budget: time={budget.time_hours:.1f}, money={budget.money_dollars:.1f}, energy={budget.energy_units:.1f}\n"
        f"Routes (max 2):\n{routes_str}\n"
        "Required keys: action_type, target_domain, metric_changes, resource_cost, reasoning.\n"
        "To complete a listed route, use action_type=\"execute\" and target_domain=the exact route id.\n"
        "Prefer completing one route over vague domain-only metric edits when a route is available.\n"
        "For a general domain intervention, target_domain must be one life domain.\n"
        "Keep reasoning under 25 words. No markdown.\n"
        f'{{"action_type": "negotiate|communicate|delegate|spend|reschedule|rest|deprioritize|execute", '
        f'"target_domain": "career|finances|relationships|physical_health|mental_wellbeing|time OR <route_id>", '
        f'"metric_changes": {{"domain.submetric": delta}}, '
        f'"resource_cost": {{"time": 0, "money": 0, "energy": 0}}, '
        f'"reasoning": "brief explanation"}}'
        f"{event_context}"
    )



# All 8 TaskGenerator domains — covers the full daily-life action space.
# transport_crisis randomly dispatches to: flight, train, car, rideshare, transit-strike
ALL_DOMAINS = [
    "career",
    "finances",
    "relationships",
    "physical_health",
    "mental_wellbeing",
    "time",
    "transport_crisis",   # ← was flight_crisis; now covers all 5 transport modes
    "code_merge_crisis",
]

def generate_dataset(n_prompts: int = 200, difficulty: int = None) -> Dataset:
    """
    Generate n conflict prompts as a HuggingFace Dataset.

    Samples evenly across ALL 8 daily-life domains (career, finances,
    relationships, physical_health, mental_wellbeing, time,
    transport_crisis [flight/train/car/rideshare/transit-strike], code_merge_crisis)
    so GRPO learns a general life-management policy.

    Args:
        n_prompts: Total number of prompts to generate.
        difficulty: If given, fix all prompts to this difficulty (1-5).
                    If None, cycles evenly through levels 1-5.
    """
    person_pool = [
        SimPerson(name="Alex",  openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2, extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam",   openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.90),
        SimPerson(name="Jordan",openness=0.7, conscientiousness=0.5, extraversion=0.6, agreeableness=0.50, neuroticism=0.40),
        SimPerson(name="Maya",  openness=0.8, conscientiousness=0.7, extraversion=0.3, agreeableness=0.80, neuroticism=0.60),
    ]

    generator = TaskGenerator()
    prompts = []
    for i in range(n_prompts):
        person = random.choice(person_pool)
        # Round-robin across all 8 domains — guarantees balanced coverage
        domain = ALL_DOMAINS[i % len(ALL_DOMAINS)]
        # Cycle difficulty 1-5 unless fixed
        curr_diff = difficulty if difficulty else (i % 5) + 1

        # Save the outer random state so that task seeding is deterministic
        # but does NOT corrupt the outer RNG chain between loop iterations.
        outer_state = random.getstate()
        task_seed = random.randint(0, 999999)
        random.seed(task_seed)
        task = generator.generate(domain=domain, difficulty=curr_diff)

        # Overlay a matching legacy conflict disruption for richer metric seeding
        conflict = generate_conflict(curr_diff)
        task.mutable_world.update(conflict.primary_disruption)
        task.visible_world.update(conflict.primary_disruption)

        metrics = LifeMetrics()
        graph = DependencyGraph()
        metrics = graph.cascade(metrics, task.mutable_world)

        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )

        # Randomly pick a starting step (0, 2, or 4) to activate replan signal
        start_step = random.choice([0, 2, 4])
        # Restore outer state now — env fast-forward below must not bleed into
        # subsequent iterations' seed selection.
        random.setstate(outer_state)
        # Advance outer state past the seed we consumed so next iteration differs.
        _ = random.random()

        event_log = []
        if start_step > 0:
            from core.lifestack_env import LifeStackEnv, LifeStackAction
            env = LifeStackEnv()
            env.reset(task=task, conflict=task.mutable_world)
            for s in range(start_step):
                # Take null actions to let events fire naturally
                obs = env.step(LifeStackAction(action_type="rest", target="time", actions_taken=0))
                for event_id in obs.metadata.get("info", []):
                    if event_id.startswith("EVENT_FIRED:"):
                        event_log.append(event_id[len("EVENT_FIRED:"):].strip())
            metrics = env.state.current_metrics
            budget = env.state.budget

        prompt = build_prompt_for_task(task, person, metrics, budget, seed=task_seed, step=start_step, event_descriptions=event_log)
        prompts.append({"prompt": prompt, "difficulty": curr_diff, "domain": domain})

    return Dataset.from_list(prompts)


def build_episode_prompt_for_task(
    task,
    person,
    metrics,
    budget,
    seed=42,
    horizon: int = 3,
    event_descriptions=None,
    history=None,
) -> str:
    """Build a compact prompt asking for a short action sequence."""
    base = build_prompt_for_task(
        task,
        person,
        metrics,
        budget,
        seed=seed,
        step=0,
        event_descriptions=event_descriptions,
    )
    history = history or []
    if history:
        history_lines = []
        for item in history[-3:]:
            history_lines.append(
                f"- step {item['step']}: {item['action_type']} -> {item['target']} "
                f"(reward {item['reward']:.3f})"
            )
        history_text = "\nEpisode history:\n" + "\n".join(history_lines)
    else:
        history_text = ""

    return (
        base
        + history_text
        + "\nEpisode objective: return a compact JSON object with an actions list.\n"
        + f"Plan up to {horizon} sequential actions. Stop early if a route can be completed.\n"
        + 'Schema: {"actions":[{"action_type":"execute","target_domain":"<route_id or domain>",'
        + '"metric_changes":{"domain.submetric":0},"resource_cost":{"time":0,"money":0,"energy":0},'
        + '"reasoning":"brief"}]}\n'
    )


def generate_episodic_dataset(
    n_episodes: int = 40,
    difficulty: int = None,
    horizon: int = 3,
) -> Dataset:
    """Generate prompts whose reward is computed over an action sequence."""
    person_pool = [
        SimPerson(name="Alex", openness=0.4, conscientiousness=0.9, extraversion=0.7, agreeableness=0.25, neuroticism=0.8),
        SimPerson(name="Chloe", openness=0.9, conscientiousness=0.2, extraversion=0.5, agreeableness=0.70, neuroticism=0.15),
        SimPerson(name="Sam", openness=0.5, conscientiousness=0.6, extraversion=0.1, agreeableness=0.65, neuroticism=0.90),
        SimPerson(name="Jordan", openness=0.7, conscientiousness=0.5, extraversion=0.6, agreeableness=0.50, neuroticism=0.40),
        SimPerson(name="Maya", openness=0.8, conscientiousness=0.7, extraversion=0.3, agreeableness=0.80, neuroticism=0.60),
    ]
    generator = TaskGenerator()
    graph = DependencyGraph()
    prompts = []

    for i in range(n_episodes):
        person = random.choice(person_pool)
        domain = ALL_DOMAINS[i % len(ALL_DOMAINS)]
        curr_diff = difficulty if difficulty else min(5, 1 + (i % 5))

        outer_state = random.getstate()
        task_seed = random.randint(0, 999999)
        random.seed(task_seed)
        task = generator.generate(domain=domain, difficulty=curr_diff)
        conflict = generate_conflict(curr_diff)
        task.mutable_world.update(conflict.primary_disruption)
        task.visible_world.update(conflict.primary_disruption)
        random.setstate(outer_state)
        _ = random.random()

        metrics = graph.cascade(LifeMetrics(), task.mutable_world)
        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        prompt = build_episode_prompt_for_task(
            task,
            person,
            metrics,
            budget,
            seed=task_seed,
            horizon=horizon,
        )
        prompts.append({
            "prompt": prompt,
            "difficulty": curr_diff,
            "domain": domain,
            "episode_horizon": horizon,
        })

    return Dataset.from_list(prompts)


# ──────────────────────────────────────────────
# 3. REWARD FUNCTION for GRPO
# ──────────────────────────────────────────────

_GLOBAL_REWARD_CALL_COUNT = 0
LOG_INTERVAL = 20
LOG_DIR = "training_logs"
SAMPLE_LOG_PATH = os.path.join(LOG_DIR, "generations.jsonl")

# Per-batch evaluation cache: keyed by (completion, prompt) tuple.
# Cleared at the start of each reward-function batch call so stale results
# from a previous GRPO generation cycle never bleed through.
_EVAL_CACHE: dict[tuple[str, str], dict] = {}
_EPISODE_EVAL_CACHE: dict[tuple[str, str, int], dict] = {}

# Guard: print reward_human_feedback_fn unavailability warning only once per run.
_HFB_WARN_SHOWN: bool = False


def _cached_lifestack_evaluation(completion: str, prompt: str) -> dict:
    """Return get_lifestack_evaluation result, caching by (completion, prompt) pair.

    Five reward functions (task_success, milestone, replan, human_feedback, longterm)
    all call get_lifestack_evaluation for the same inputs inside a single GRPO step.
    Without caching this spins up 5 independent LifeStackEnv instances per completion.
    The cache cuts env construction to once per completion per batch.
    """
    key = (completion, prompt)
    if key not in _EVAL_CACHE:
        _EVAL_CACHE[key] = get_lifestack_evaluation(completion, prompt)
    return _EVAL_CACHE[key]


def _clear_eval_cache() -> None:
    _EVAL_CACHE.clear()


def _clear_episode_eval_cache() -> None:
    _EPISODE_EVAL_CACHE.clear()


def _load_first_json_object(completion: str) -> dict:
    """Parse the first complete JSON object, ignoring trailing model text."""
    text = completion.strip()
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[-1].split("```")[0]

    decoder = json.JSONDecoder()
    import re as _re
    for _m in _re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[_m.start():].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def _route_ids_from_prompt(prompt: str) -> set[str]:
    """Extract route ids that were explicitly shown to the model."""
    import re as _re

    m = _re.search(r'<SYSTEM_METADATA>\n(.*?)\n</SYSTEM_METADATA>', prompt, _re.DOTALL)
    if m:
        try:
            meta = json.loads(m.group(1).strip())
            return {str(r) for r in meta.get("route_ids", []) if r}
        except Exception:
            pass

    route_ids: set[str] = set()
    in_routes = False
    for line in prompt.splitlines():
        if line.startswith("Routes"):
            in_routes = True
            continue
        if in_routes and line.startswith("- "):
            route_ids.add(line[2:].split(":", 1)[0].strip())
            continue
        if in_routes and line:
            break
    return route_ids


def _actions_from_completion(completion: str) -> list[dict]:
    """Accept either a single action object or an episode {"actions": [...]} payload."""
    data = _load_first_json_object(completion)
    if isinstance(data.get("actions"), list):
        return [a for a in data["actions"] if isinstance(a, dict)]
    return [data]


def _to_lifestack_action(data: dict, completion: str, actions_taken: int = 1):
    """Convert model JSON into the env action, supporting optional route_id."""
    from core.lifestack_env import LifeStackAction

    return LifeStackAction(
        action_type=data.get("action_type"),
        target=data.get("route_id") or data.get("target_domain"),
        metric_changes=data.get("metric_changes", {}),
        resource_cost=data.get("resource_cost", {}),
        reasoning=data.get("reasoning", ""),
        completion=completion,
        actions_taken=actions_taken,
    )


def _metadata_from_prompt(prompt: str) -> dict:
    import re

    m = re.search(r'<SYSTEM_METADATA>\n(.*?)\n</SYSTEM_METADATA>', prompt, re.DOTALL)
    if not m:
        raise ValueError("SYSTEM_METADATA missing from prompt")
    return json.loads(m.group(1).strip())


def _task_from_metadata(meta: dict):
    gen = TaskGenerator()
    domain = meta.get("domain", "flight_crisis")
    task = gen.generate(domain=domain, difficulty=meta.get("difficulty", 3))
    task.mutable_world.update(meta.get("disruption", {}))
    task.visible_world.update(meta.get("disruption", {}))
    return task


def get_lifestack_evaluation(completion: str, prompt: str) -> dict:
    """Run the environment and return the full reward breakdown. Computed fresh per call to prevent hacking."""
    from core.lifestack_env import LifeStackEnv, LifeStackAction
    import re
    
    try:
        # 1. Parse JSON
        data = _load_first_json_object(completion)

        # 2. Extract Task Metadata
        m = re.search(r'<SYSTEM_METADATA>\n(.*?)\n</SYSTEM_METADATA>', prompt, re.DOTALL)
        if not m:
            return {"reward": -0.5, "breakdown": {}}
        
        meta = json.loads(m.group(1).strip())
        try:
            # Use TaskGenerator so routes/milestones/success_conditions are populated.
            from agent.conflict_generator import TaskGenerator
            gen = TaskGenerator()
            domain = meta.get("domain", "flight_crisis")
            # Keep seed active through the ENTIRE env evaluation — task gen, reset,
            # fast-forward, and the action step.  Without this, stochastic events
            # (event.step == -1, random.random() < probability) fire differently each
            # call, so reward_task_success_fn / reward_milestone_fn / reward_replan_fn
            # see inconsistent env states for the same completion.
            eval_seed = meta.get("seed", 42)
            random.seed(eval_seed)
            task = gen.generate(domain=domain, difficulty=meta.get("difficulty", 3))
            # Overlay the actual disruption that was presented in the prompt
            task.mutable_world.update(meta.get("disruption", {}))
            task.visible_world.update(meta.get("disruption", {}))
        except Exception as e:
            print(f"[reward] Task construction failed: {e}")
            random.seed()
            return {"reward": -0.5, "breakdown": {"error": str(e)}}

        # Validate required fields are present and non-None.
        _required = ("id", "goal", "constraints", "mutable_world", "visible_world")
        if any(getattr(task, f, None) is None for f in _required):
            print("[reward] Task missing required fields after construction.")
            random.seed()
            return {"reward": -0.5, "breakdown": {"error": "missing_fields"}}

        # 3. Step Env — still under eval_seed so events are deterministic per (completion, prompt)
        env = LifeStackEnv()
        env.reset(task=task, conflict=meta.get("disruption", {}))

        # Fast-forward to the state the model saw
        curr_step = meta.get("step", 0)
        for _ in range(curr_step):
            env.step(LifeStackAction(action_type="rest", target="time", actions_taken=0))

        initial_metrics = dict(env.state.current_metrics.flatten())
        action = _to_lifestack_action(data, completion, actions_taken=1)
        obs = env.step(action)

        # 7-day discounted rollout — real long-term signal, not decoration.
        # Runs BEFORE random.seed() so the null steps share the same eval_seed,
        # keeping the trajectory deterministic for the same (completion, prompt).
        rollout_data = env.rollout(n_steps=7, gamma=0.9)
        random.seed()  # restore global RNG — eval_seed must not bleed into trainer

        # Inject longterm component into the breakdown so reward_longterm_fn
        # can extract it without a second env construction.
        breakdown = obs.metadata.get("breakdown", {})
        components = breakdown.get("components", {})
        components["longterm"] = rollout_data["discounted_reward"]
        breakdown["components"] = components

        result = {
            "reward": float(obs.reward),
            "breakdown": breakdown,
            "action": action,
            "obs_metrics": dict(obs.metrics),
            "initial_metrics": initial_metrics,
            "longterm_reward": rollout_data["discounted_reward"],
            "trajectory": rollout_data["trajectory"],
        }

        # 4. Global Logging
        global _GLOBAL_REWARD_CALL_COUNT
        _GLOBAL_REWARD_CALL_COUNT += 1
        if _GLOBAL_REWARD_CALL_COUNT % LOG_INTERVAL == 0:
            if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
            log_entry = {
                "step": _GLOBAL_REWARD_CALL_COUNT,
                "prompt": prompt[:500] + "...",
                "completion": completion,
                "action": data,
                "reward": result["reward"],
                "longterm_reward": result["longterm_reward"],
                "breakdown": result["breakdown"],
                "components": components,
            }
            with open(SAMPLE_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            if components:
                comp_str = " | ".join(f"{k}={v:.3f}" for k, v in components.items())
                print(f"[step {_GLOBAL_REWARD_CALL_COUNT}] r0={result['reward']:.3f} | r_lt={result['longterm_reward']:.3f} | {comp_str}")

        return result
        
    except Exception:
        random.seed()  # always restore RNG on any failure path
        return {"reward": -0.5, "breakdown": {}, "action": None, "initial_metrics": meta.get("disruption", {}) if 'meta' in locals() else {}}

def reward_clean_eos_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Reward the model for stopping immediately after the closing JSON brace.

    clipped_ratio=1.0 happens because the model generates valid JSON then keeps
    producing free-form text, so every token after '}' contributes zero reward but
    still costs KL budget — every gradient update gets clipped.

    Signal:
      +0.20  completion ends within 8 chars of the first complete JSON object
      +0.10  completion ends within 32 chars (minor trailing whitespace OK)
       0.00  some trailing text but still parseable
      -0.10  >64 chars of trailing garbage after the JSON closes
    """
    import re as _re
    rewards = []
    for c in completions:
        try:
            text = c.strip()
            # Strip markdown fences
            if "```json" in text:
                inner = text.split("```json")[-1].split("```")[0]
            elif "```" in text:
                inner = text.split("```")[-1].split("```")[0]
            else:
                inner = text

            decoder = json.JSONDecoder()
            for _m in _re.finditer(r"\{", inner):
                try:
                    _, end_idx = decoder.raw_decode(inner[_m.start():].strip())
                    # end_idx is relative to the slice; trailing chars = len(inner) - (_m.start() + end_idx)
                    trailing = len(inner) - (_m.start() + end_idx)
                    if trailing <= 8:
                        rewards.append(0.20)
                    elif trailing <= 32:
                        rewards.append(0.10)
                    elif trailing <= 64:
                        rewards.append(0.00)
                    else:
                        rewards.append(-0.10)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                rewards.append(-0.10)   # no valid JSON found
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_compact_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Strong signal against the fill-the-context habit.

    reward_clean_eos_fn has a narrow [-0.10, +0.20] range that is routinely
    swamped by format and return rewards.  This function uses a wider [-0.5, +0.4]
    range to make compactness a first-class training objective.

    Scoring is based on trailing characters after the first complete JSON object
    (same boundary as _find_json_end_text uses for gradient masking in
    LifeStackGRPOTrainer, so reward and gradient are always aligned).

      trailing ≤ 10   → +0.40  (stops right after the closing brace)
      trailing ≤ 60   → +0.20  (whitespace / newline only)
      trailing ≤ 200  → -0.20  (a sentence of explanation)
      trailing  > 200 → -0.50  (multi-line wall of text)
      no valid JSON   → -0.50
    """
    rewards = []
    for c in completions:
        try:
            text = c.strip()
            end_char = _find_json_end_text(text)
            if end_char is None:
                rewards.append(-0.50)
                continue
            trailing = len(text) - end_char
            if trailing <= 10:
                rewards.append(0.40)
            elif trailing <= 60:
                rewards.append(0.20)
            elif trailing <= 200:
                rewards.append(-0.20)
            else:
                rewards.append(-0.50)
        except Exception:
            rewards.append(-0.50)
    return rewards


def reward_format_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    from core.reward import reward_format_compliance
    return [
        reward_format_compliance(c, valid_route_ids=_route_ids_from_prompt(p))
        for c, p in zip(completions, prompts)
    ]


def reward_route_target_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Reward using listed route ids so milestone/completion signals can fire."""
    rewards = []
    for c, p in zip(completions, prompts):
        try:
            data = _load_first_json_object(c)
            route_ids = {r.lower() for r in _route_ids_from_prompt(p)}
            target = str(data.get("route_id") or data.get("target_domain", "")).lower()
            action_type = str(data.get("action_type", "")).lower()
            if target in route_ids and action_type == "execute":
                rewards.append(0.30)
            elif target in route_ids:
                rewards.append(0.15)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_plausibility_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Penalize zero-cost metric changes (Independent Logic Check)."""
    from core.reward import reward_plausibility_check
    import json
    results = []
    for c in completions:
        try:
            data = _load_first_json_object(c)
            mc = data.get("metric_changes", {})
            rc = data.get("resource_cost", {})
            results.append(reward_plausibility_check(mc, rc))
        except Exception:
            results.append(0.0)
    return results

def reward_task_success_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Core outcome reward isolated to completion (Environment Simulation)."""
    _clear_eval_cache()   # new batch — clear stale entries
    results = []
    for c, p in zip(completions, prompts):
        eval_res = _cached_lifestack_evaluation(c, p)
        if not eval_res.get("breakdown"):
            results.append(eval_res.get("reward", -0.5))
        else:
            results.append(eval_res.get("breakdown", {}).get("components", {}).get("completion", 0.0))
    return results

def reward_milestone_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Monitor progress through logical bottlenecks (Environment Simulation)."""
    return [_cached_lifestack_evaluation(c, p).get("breakdown", {}).get("components", {}).get("milestone", 0.0) for c, p in zip(completions, prompts)]

def reward_reasoning_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Evaluate planning coherence (Independent Semantic/Logic Check)."""
    from core.reward import reward_reasoning_coherence
    import json
    results = []
    for c in completions:
        try:
            data = _load_first_json_object(c)
            
            reasoning = data.get("reasoning", "")
            a_type = data.get("action_type", "")
            # reward_reasoning_coherence returns [-0.30, 0.30] — no scaling needed
            results.append(reward_reasoning_coherence(reasoning, action_type=a_type))
        except Exception:
            results.append(-0.1)
    return results

def reward_human_feedback_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Rewards actions that align with past human outcome feedback (ChromaDB memory).

    Requires chromadb + a pre-populated LifeStackMemory database.
    Falls back silently to neutral 0.0 when:
      - chromadb is not installed (e.g. fresh Kaggle / Colab session)
      - the memory DB is empty or unreachable
    Returns 0.0 (abstain) rather than penalising the model.
    """
    global _HFB_WARN_SHOWN
    # ── Guard: skip gracefully if chromadb / memory unavailable ──────────
    try:
        from core.feedback import OutcomeFeedback, compute_human_feedback_reward
        from agent.memory import LifeStackMemory
        memo = LifeStackMemory(silent=True)
    except (ImportError, Exception) as e:
        if not _HFB_WARN_SHOWN:
            print(f"[warning] reward_human_feedback_fn unavailable ({e}), returning neutral rewards.")
            _HFB_WARN_SHOWN = True
        # chromadb not installed or DB init failed — abstain instead of
        # globally depressing the curriculum reward.
        return [0.0] * len(completions)

    rewards = []
    for c, p in zip(completions, prompts):
        try:
            eval_res = _cached_lifestack_evaluation(c, p)
            action = eval_res.get("action")
            if not action:
                rewards.append(0.0)
                continue

            # Use task prompt to query feedback instead of model-generated reasoning
            # to avoid reward-hacking ChromaDB. Must use query_embeddings to match
            # the custom _embed_text() space used when storing feedback.
            # Bug 8: Use embeddings instead of raw text for query
            q_emb = memo._embed_text(p)
            similar_fb_list = memo.feedback_collection.query(
                query_embeddings=[q_emb],
                n_results=1
            ).get('metadatas', [[]])[0]

            if not similar_fb_list:
                rewards.append(0.0)
                continue

            fb_meta = similar_fb_list[0]
            fb = OutcomeFeedback(
                episode_id=fb_meta["episode_id"],
                overall_effectiveness=fb_meta["effectiveness"],
                domains_improved=json.loads(fb_meta["domains_improved"]),
                domains_worsened=json.loads(fb_meta["domains_worsened"])
            )

            from core.lifestack_env import LifeStackObservation
            obs = LifeStackObservation(metrics=eval_res.get("obs_metrics", {}))
            init_metrics = eval_res.get("initial_metrics", {})
            fb_reward = compute_human_feedback_reward(init_metrics, obs, fb)
            rewards.append(fb_reward)

        except Exception:
            rewards.append(0.0)

    return rewards

def reward_replan_fn(completions, prompts, **kwargs) -> list[float]:
    """Exposes the internal replan bonus as a standalone GRPO signal."""
    rewards = []
    for c, p in zip(completions, prompts):
        eval_data = _cached_lifestack_evaluation(c, p)
        rewards.append(eval_data.get("breakdown", {}).get("components", {}).get("replan", 0.0))
    return rewards

def reward_longterm_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    7-day γ=0.9 discounted rollout reward.

    After the model's action is applied, the env runs 7 null/rest steps to
    model "what happens to your life if nothing extraordinary occurs after
    this decision."  The discounted sum is the training signal.

    This is the only reward function whose gradient explicitly penalises
    actions that look good on day 0 but trigger a cascade collapse by day 4.
    It is NOT a decoration — the rollout runs inside the real LifeStack env.
    """
    return [
        _cached_lifestack_evaluation(c, p).get("longterm_reward", 0.0)
        for c, p in zip(completions, prompts)
    ]


def get_episode_evaluation(completion: str, prompt: str, horizon: int = 3, gamma: float = 0.9) -> dict:
    """Evaluate a generated action sequence by stepping the real environment."""
    from core.lifestack_env import LifeStackEnv

    try:
        actions = _actions_from_completion(completion)
        if not actions:
            return {"reward": -0.5, "steps": [], "success": False, "failure": True}

        meta = _metadata_from_prompt(prompt)
        eval_seed = meta.get("seed", 42)
        random.seed(eval_seed)
        task = _task_from_metadata(meta)
        env = LifeStackEnv()
        env.reset(task=task, conflict=meta.get("disruption", {}))

        discounted = 0.0
        steps = []
        final_obs = None
        for idx, action_data in enumerate(actions[:horizon]):
            env_action = _to_lifestack_action(action_data, completion, actions_taken=1)
            obs = env.step(env_action)
            final_obs = obs
            step_reward = float(obs.reward or 0.0)
            discounted += (gamma ** idx) * step_reward
            steps.append({
                "step": idx + 1,
                "action_type": env_action.action_type,
                "target": env_action.target,
                "reward": step_reward,
                "success": bool(obs.metadata.get("success")),
                "failure": bool(obs.metadata.get("failure")),
                "events": obs.metadata.get("events", []),
            })
            if obs.done:
                break

        random.seed()
        if not steps:
            return {"reward": -0.5, "steps": [], "success": False, "failure": True}

        success = bool(final_obs and final_obs.metadata.get("success"))
        failure = bool(final_obs and final_obs.metadata.get("failure"))
        terminal_bonus = 0.25 if success else (-0.15 if failure else 0.0)
        route_completion = 1.0 if success else 0.0
        normalized = discounted / max(1, min(horizon, len(actions)))
        reward = max(-1.0, min(1.0, normalized + terminal_bonus))

        return {
            "reward": reward,
            "discounted_reward": discounted,
            "terminal_bonus": terminal_bonus,
            "route_completion": route_completion,
            "steps": steps,
            "success": success,
            "failure": failure,
        }
    except Exception as e:
        random.seed()
        return {"reward": -0.5, "steps": [], "success": False, "failure": True, "error": str(e)}


def _cached_episode_evaluation(completion: str, prompt: str, horizon: int) -> dict:
    key = (completion, prompt, horizon)
    if key not in _EPISODE_EVAL_CACHE:
        _EPISODE_EVAL_CACHE[key] = get_episode_evaluation(completion, prompt, horizon=horizon)
    return _EPISODE_EVAL_CACHE[key]


def reward_episode_format_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Format score for either single-action JSON or {"actions": [...]} JSON."""
    from core.reward import reward_format_compliance

    rewards = []
    for completion, prompt in zip(completions, prompts):
        try:
            actions = _actions_from_completion(completion)
            if not actions:
                rewards.append(-0.5)
                continue
            route_ids = _route_ids_from_prompt(prompt)
            per_action = [
                reward_format_compliance(json.dumps(action), valid_route_ids=route_ids)
                for action in actions
            ]
            rewards.append(float(np.mean(per_action)))
        except Exception:
            rewards.append(-0.5)
    return rewards


def reward_episode_plausibility_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Average plausibility over the proposed action sequence."""
    from core.reward import reward_plausibility_check

    rewards = []
    for completion in completions:
        try:
            actions = _actions_from_completion(completion)
            if not actions:
                rewards.append(-0.1)
                continue
            scores = [
                reward_plausibility_check(a.get("metric_changes", {}), a.get("resource_cost", {}))
                for a in actions
            ]
            rewards.append(float(np.mean(scores)))
        except Exception:
            rewards.append(-0.1)
    return rewards


def reward_episode_return_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Trajectory-level reward: discounted env return plus terminal success shaping."""
    _clear_episode_eval_cache()
    raw_horizons = kwargs.get("episode_horizon", 3)
    if not isinstance(raw_horizons, list):
        raw_horizons = [raw_horizons] * len(completions)
    rewards = []
    for c, p, h in zip(completions, prompts, raw_horizons):
        horizon = int(h or 3)
        rewards.append(_cached_episode_evaluation(c, p, horizon).get("reward", -0.5))
    return rewards

# ──────────────────────────────────────────────
# 4. CHECKPOINT HELPERS
# ──────────────────────────────────────────────

def find_latest_checkpoint(stage_dir: str):
    """
    Scan a stage output directory for the most recent Trainer checkpoint.
    Returns the checkpoint path, or None if none exist.
    """
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(stage_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1])
    )
    return checkpoints[-1] if checkpoints else None


_CURRICULUM_STATE_FILE = "curriculum_state.json"

def save_stage_state(output_dir: str, stage: int, curr_diff: int):
    """Persist curriculum progress so we can resume after a session cut."""
    path = os.path.join(output_dir, _CURRICULUM_STATE_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed_stage": stage, "next_difficulty": curr_diff}, f)
    print(f"  [ckpt] Curriculum state saved → stage={stage}, next_diff={curr_diff}")


def load_stage_state(output_dir: str) -> tuple[int, int]:
    """
    Returns (start_stage, curr_diff) from a previous run.
    Falls back to (1, 1) if no state file exists.
    """
    path = os.path.join(output_dir, _CURRICULUM_STATE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        start_stage = state["completed_stage"] + 1
        curr_diff   = state["next_difficulty"]
        print(f"  [ckpt] Resuming from stage {start_stage}, difficulty {curr_diff}")
        return start_stage, curr_diff
    return 1, 1


# ──────────────────────────────────────────────
# 5. TRAINING LOOP  (checkpoint-aware)
# ──────────────────────────────────────────────

def train_curriculum(
    n_stages=5,
    n_prompts_per_stage=100,
    output_dir="./lifestack_model",
    resume=False,
    start_stage=None,
    max_prompt_length: int | None = 2048,
    max_completion_length: int = 224,
):
    """
    Curriculum training with automatic checkpoint saving and resume.

    Each stage saves a checkpoint every 25 steps and persists curriculum
    state to curriculum_state.json.  If the session is killed mid-stage,
    re-run with --resume and the trainer will pick up from the last
    saved checkpoint automatically.

    Args:
        resume:      If True, read curriculum_state.json to find the last
                     completed stage and continue from there.
        start_stage: Override the starting stage (1-indexed). Useful for
                     manual restart (e.g. --start-stage 3).
    """
    print("=" * 60)
    print("🚀 LIFESTACK SUCCESS-BASED CURRICULUM TRAINING")
    print("=" * 60)

    model, tokenizer = load_model()

    # ── Determine where to start ────────────────────────────────────────
    if resume:
        first_stage, curr_diff = load_stage_state(output_dir)
    elif start_stage:
        first_stage = start_stage
        curr_diff   = 1          # difficulty resets; user can edit state file for fine control
    else:
        first_stage, curr_diff = 1, 1

    trainer = None
    for stage in range(first_stage, n_stages + 1):
        print(f"\n[STAGE {stage}/{n_stages}] Difficulty={curr_diff}")

        stage_dir = f"{output_dir}/stage_{stage}"

        # ── Check for a mid-stage checkpoint from a previous session ─────
        resume_ckpt = find_latest_checkpoint(stage_dir) if resume else None
        if resume_ckpt:
            print(f"  [ckpt] Resuming mid-stage from: {resume_ckpt}")
        else:
            # Generate fresh data only for a clean start of the stage
            dataset = generate_dataset(n_prompts_per_stage, difficulty=curr_diff)

        # ── Per-stage LR decay: start aggressive, tighten on later stages ───
        # Stage 1: 8e-6 (format warm-up needs faster convergence)
        # Stage 2: 5e-6, Stage 3: 3e-6, Stage 4: 2e-6, Stage 5+: 1e-6
        stage_lr_schedule = {1: 8e-6, 2: 5e-6, 3: 3e-6, 4: 2e-6}
        stage_lr = stage_lr_schedule.get(stage, 1e-6)

        # ── GRPOConfig with checkpoint cadence ───────────────────────────
        config = _make_grpo_config(
            output_dir=stage_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=stage_lr,
            warmup_ratio=0.05,           # 5% warmup steps — smoother convergence
            max_prompt_length=max_prompt_length,
            # Completion budget for GRPO generations (separate from model max context).
            max_completion_length=max_completion_length,
            temperature=0.7,
            # TRL rule: num_generations must divide per_device_train_batch_size.
            num_generations=4,
            # Read actual model dtype so bf16/fp16 flags never mismatch Unsloth's
            # patched trainer (raises TypeError if both disagree).
            bf16=_dtype_flags(model)[0],
            fp16=_dtype_flags(model)[1],
            # ── Checkpoint settings ──────────────────────────────────────
            # 100 prompts / (batch=4 × accum=4) ≈ 6 optimizer steps per stage.
            # save_steps=25 would never fire; use 5 to save at step 5 of ~6.
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            # ── Logging ─────────────────────────────────────────────────
            logging_steps=5,
            # tensorboard only if installed; fall back to none to avoid ImportError on Colab/Kaggle
            report_to="tensorboard" if _tensorboard_available() else "none",
        )
        config.unsloth_num_chunks = -1

        if stage == 1:
            # Warm-up: learn valid JSON structure first, then optimize decisions.
            stage_reward_funcs = [reward_format_fn, reward_clean_eos_fn, reward_route_target_fn]
            print("  Warm-up reward mode: format + EOS-cleanliness")
            # Encourage real EOS: format + route are larger-magnitude; keep EOS nontrivial.
            config.reward_weights = [1.0, 1.5, 1.0]
        else:
            stage_reward_funcs = [
                reward_format_fn,
                reward_clean_eos_fn,
                reward_route_target_fn,
                reward_plausibility_fn,
                reward_task_success_fn,
                reward_milestone_fn,
                reward_replan_fn,
                reward_reasoning_fn,
                reward_human_feedback_fn,
                reward_longterm_fn,
            ]
            config.reward_weights = [1.0, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.5]

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,  # TRL 1.x: renamed from tokenizer=
            args=config,
            train_dataset=dataset if not resume_ckpt else generate_dataset(n_prompts_per_stage, difficulty=curr_diff),
            reward_funcs=stage_reward_funcs,
        )

        # Pass the checkpoint path — Trainer will reload weights + optimizer state
        trainer.train(resume_from_checkpoint=resume_ckpt)

        # ── Save completed stage model ───────────────────────────────────
        trainer.save_model(stage_dir)
        tokenizer.save_pretrained(stage_dir)
        print(f"  ✅ Stage {stage} model saved → {stage_dir}")

        # ── Curriculum progression logic ─────────────────────────────────
        # TRL 1.x logs mean reward as "reward"; some builds use "train/reward" — check both
        last_log = trainer.state.log_history[-1] if trainer.state.log_history else {}
        avg_reward = last_log.get("reward", last_log.get("train/reward", 0.0))
        # Advance on non-negative reward. The earlier 0.2/0.6 gates kept every
        # completed run stuck at difficulty 1, which hides whether the policy can
        # generalize once basic syntax and route targeting are working.
        advance_threshold = 0.0
        if avg_reward >= advance_threshold and curr_diff < 5:
            print(f"  ✅ Reward {avg_reward:.3f} ≥ {advance_threshold} — advancing to difficulty {curr_diff + 1}")
            curr_diff += 1
        else:
            print(f"  ⚠️  Reward {avg_reward:.3f} ≤ {advance_threshold} — holding at difficulty {curr_diff}")

        # ── Persist curriculum state AFTER each stage ────────────────────
        # This is what lets us resume correctly on next session
        save_stage_state(output_dir, stage, curr_diff)

    # ── Final model save ─────────────────────────────────────────────────
    # Guard: if all stages were already complete (resume from finished run),
    # the loop body never executed and trainer/tokenizer are not bound.
    if trainer is None and first_stage > n_stages:
        print(f"\n✅ All {n_stages} stages already complete. Model is at {output_dir}")
        print("   Run --full-episode to evaluate, or --push-to-hub to upload.")
        return None
    if trainer is None:
        raise RuntimeError("No trainer was created; check stage/resume configuration.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n🏁 Training complete. Final model → {output_dir}")
    return trainer


def train_episodic_curriculum(
    n_stages=2,
    n_episodes_per_stage=40,
    output_dir="./lifestack_model_v2",
    episode_horizon: int = 3,
    model=None,
    tokenizer=None,
    resume=False,
    start_stage=None,
    max_prompt_length: int | None = 2048,
    max_completion_length: int | None = None,
    num_train_epochs: int = 1,
):
    """
    GRPO fine-tuning where each completion is rewarded by an environment episode.

    Standard GRPO still samples one completion per prompt, so the trainable object
    is a compact action sequence (`{"actions": [...]}`). The same environment
    then executes that sequence step by step and returns a trajectory reward.
    """
    print("=" * 60)
    print("🎬 LIFESTACK EPISODIC GRPO TRAINING")
    print("=" * 60)

    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    if resume:
        first_stage, curr_diff = load_stage_state(output_dir)
    elif start_stage:
        first_stage, curr_diff = start_stage, 1
    else:
        first_stage, curr_diff = 1, 1

    trainer = None
    for stage in range(first_stage, n_stages + 1):
        print(f"\n[EPISODE STAGE {stage}/{n_stages}] Difficulty={curr_diff} | horizon={episode_horizon}")
        stage_dir = f"{output_dir}/episode_stage_{stage}"
        resume_ckpt = find_latest_checkpoint(stage_dir) if resume else None
        if resume_ckpt:
            print(f"  [ckpt] Resuming mid-stage from: {resume_ckpt}")

        dataset = generate_episodic_dataset(
            n_episodes=n_episodes_per_stage,
            difficulty=curr_diff,
            horizon=episode_horizon,
        )

        stage_lr_schedule = {1: 3e-6, 2: 2e-6}
        stage_lr = stage_lr_schedule.get(stage, 1e-6)
        if max_completion_length is None:
            # ~256 tokens of JSON headroom per proposed action, capped for VRAM.
            max_completion = min(1024, max(512, 256 * episode_horizon))
        else:
            max_completion = int(max_completion_length)

        config = _make_grpo_config(
            output_dir=stage_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=stage_lr,
            warmup_ratio=0.05,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion,
            # 0.9 restores entropy — entropy=0.36 (previous run) means the model
            # was nearly deterministic and never explored EOS naturally.
            temperature=0.9,
            num_generations=4,
            bf16=_dtype_flags(model)[0],
            fp16=_dtype_flags(model)[1],
            save_strategy="steps",
            save_steps=5,
            save_total_limit=3,
            logging_steps=5,
            report_to="tensorboard" if _tensorboard_available() else "none",
        )
        config.unsloth_num_chunks = -1
        # Weights for: format | EOS-clean | plausibility | episode-return | compact
        # reward_compact_fn has [-0.5, +0.4] range vs reward_clean_eos_fn [-0.1, +0.2].
        # Weight compact at 2.5x and EOS-clean at 2.0x so the combined compactness
        # signal (~[-1.2, +1.2]) is competitive with format+return (~[-1, +1]).
        config.reward_weights = [1.0, 2.0, 0.75, 1.0, 2.5]

        trainer = LifeStackGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=config,
            train_dataset=dataset,
            reward_funcs=[
                reward_episode_format_fn,
                reward_clean_eos_fn,
                reward_episode_plausibility_fn,
                reward_episode_return_fn,
                reward_compact_fn,
            ],
        )
        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(stage_dir)
        tokenizer.save_pretrained(stage_dir)
        print(f"  ✅ Episode stage {stage} model saved → {stage_dir}")

        last_log = trainer.state.log_history[-1] if trainer.state.log_history else {}
        avg_reward = last_log.get("reward", last_log.get("train/reward", 0.0))
        advance_threshold = 0.0
        if avg_reward >= advance_threshold and curr_diff < 5:
            print(f"  ✅ Episode reward {avg_reward:.3f} ≥ {advance_threshold} — advancing to difficulty {curr_diff + 1}")
            curr_diff += 1
        else:
            print(f"  ⚠️  Episode reward {avg_reward:.3f} ≤ {advance_threshold} — holding at difficulty {curr_diff}")
        save_stage_state(output_dir, stage, curr_diff)

    if trainer is None and first_stage > n_stages:
        print(f"\n✅ All {n_stages} episodic stages already complete. Model is at {output_dir}")
        return None
    if trainer is None:
        raise RuntimeError("No episodic trainer was created; check stage/resume configuration.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n🏁 Episodic training complete. Final model → {output_dir}")
    return trainer


# ──────────────────────────────────────────────
# 5. EVALUATION + REWARD CURVE
# ──────────────────────────────────────────────

def evaluate_and_plot(model_dir="./lifestack_model"):
    """Load the trained model, run 50 evaluation episodes, plot the curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 50)
    print("  EVALUATION")
    print("=" * 50)

    # Use Unsloth's loader to avoid peft version conflicts on Kaggle/Colab
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("  Loaded via Unsloth FastLanguageModel")
    except Exception as unsloth_err:
        print(f"  Unsloth load failed ({unsloth_err}), falling back to AutoModelForCausalLM")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    graph = DependencyGraph()
    rewards = []

    generator = TaskGenerator()
    for ep in range(50):
        difficulty = min(5, 1 + ep // 10)
        # Cycle through all 8 domains during evaluation
        domain = ALL_DOMAINS[ep % len(ALL_DOMAINS)]
        ep_seed = ep * 137  # deterministic per episode so reward_task_success_fn reconstructs the same task
        random.seed(ep_seed)
        task = generator.generate(domain=domain, difficulty=difficulty)
        random.seed()

        metrics = LifeMetrics()
        # Initial disruption from legacy templates
        conflict = generate_conflict(difficulty)
        metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})

        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        person = SimPerson(name="Eval")

        prompt = build_prompt_for_task(task, person, metrics, budget, seed=ep_seed, step=0)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = reward_task_success_fn([completion], [prompt])[0]
        rewards.append(r)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/50 | Reward: {r:.3f} | Avg: {np.mean(rewards):.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 51), rewards, color="steelblue", alpha=0.6, label="Episode Reward")

    # Rolling average
    window = 5
    rolling = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
    ax.plot(range(1, 51), rolling, color="crimson", linewidth=2, linestyle="--", label="5-ep Rolling Avg")

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("LifeStack GRPO — Evaluation Reward Curve (Qwen2.5-1.5B)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Evaluation Episode (post-training)", fontsize=11)
    ax.set_ylabel("Completion Reward [-1, +1]", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Annotate mean
    mean_r = float(np.mean(rewards))
    ax.axhline(y=mean_r, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(48, mean_r + 0.02, f"mean={mean_r:.2f}", ha="right", fontsize=9, color="steelblue")
    fig.tight_layout()
    fig.savefig("grpo_reward_curve.png", dpi=150)
    plt.close(fig)
    print("📊 Saved grpo_reward_curve.png")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 6. POST-TRAINING VALIDATION
# ──────────────────────────────────────────────

MIN_MODEL_SIZE_BYTES = 5 * 1024 * 1024   # 5 MB — LoRA adapter ~39 MB, placeholder ~few KB

def validate_saved_model(output_dir: str = "./lifestack_model"):
    """
    Validates that a real model was saved (not a placeholder).
    Raises RuntimeError if pytorch_model.bin or model.safetensors is missing / too small.
    """
    import glob
    weight_files = (
        glob.glob(os.path.join(output_dir, "*.bin")) +
        glob.glob(os.path.join(output_dir, "*.safetensors")) +
        glob.glob(os.path.join(output_dir, "**", "*.safetensors"), recursive=True) +
        glob.glob(os.path.join(output_dir, "**", "*.bin"), recursive=True)
    )
    # Deduplicate
    weight_files = list(set(weight_files))

    if not weight_files:
        raise RuntimeError(
            f"[VALIDATION FAIL] No weight files found in {output_dir}.\n"
            "Real training never completed — run train_trl.py on a GPU instance."
        )

    total_bytes = sum(os.path.getsize(f) for f in weight_files)
    if total_bytes < MIN_MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"[VALIDATION FAIL] Total weight size = {total_bytes} bytes ({total_bytes/1e6:.2f} MB).\n"
            f"Expected > {MIN_MODEL_SIZE_BYTES/1e6:.0f} MB for a real model.\n"
            f"Found files: {weight_files}\n"
            "This looks like a placeholder. Run full training on a GPU."
        )

    print(f"[VALIDATION PASS] Model saved correctly.")
    print(f"  Weight files : {len(weight_files)}")
    print(f"  Total size   : {total_bytes / 1e6:.1f} MB")
    return total_bytes


# ──────────────────────────────────────────────
# 7. DRY-RUN MODE (validates pipeline without GPU)
# ──────────────────────────────────────────────

def dry_run(
    output_dir: str = "./lifestack_model_dryrun",
    episode_train: bool = False,
    episode_horizon: int = 2,
    max_prompt_length: int | None = 2048,
    max_completion_length: int = 224,
    episodic_max_completion: int | None = None,
):
    """
    Runs a single GRPO training step on a minimal dataset (4 prompts).
    Verifies the entire pipeline: dataset → prompt → reward → trainer.train() → save.
    Does NOT require a GPU.  Saved weights will be small (< 50 MB) — that is expected.

    Use this to confirm:
    - All imports resolve
    - Reward functions are callable
    - Trainer.train() completes without error
    - model.save_pretrained() writes real weight files
    """
    print("=" * 60)
    mode = "EPISODIC" if episode_train else "SINGLE-STEP"
    print(f"🧪 LIFESTACK DRY-RUN ({mode}, 1 step, CPU, tiny dataset)")
    print("=" * 60)

    model, tokenizer = load_model_for_dry_run()

    if episode_train:
        dataset = generate_episodic_dataset(n_episodes=4, difficulty=1, horizon=episode_horizon)
    else:
        dataset = generate_dataset(n_prompts=4, difficulty=1)
    print(f"  Dataset size : {len(dataset)} prompts")

    if episode_train:
        if episodic_max_completion is None:
            comp = min(1024, max(512, 256 * episode_horizon))
        else:
            comp = int(episodic_max_completion)
    else:
        comp = int(max_completion_length)

    config = _make_grpo_config(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        max_prompt_length=max_prompt_length,
        max_completion_length=comp,
        temperature=0.7,
        num_generations=4,
        max_steps=1,          # ONE step — just proves the pipeline works
        bf16=False,
        fp16=False,
        report_to="none",     # No tensorboard for dry-run
        logging_steps=1,
    )
    config.unsloth_num_chunks = -1

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 1.x: renamed from tokenizer=
        args=config,
        train_dataset=dataset,
        reward_funcs=(
            [reward_episode_format_fn, reward_episode_return_fn]
            if episode_train else
            [reward_format_fn, reward_route_target_fn]
        ),
    )

    print("  Running 1 training step...")
    trainer.train()
    print("  ✅ trainer.train() completed.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✅ model.save_pretrained() → {output_dir}")

    # Check something real was saved
    import glob
    weight_files = (
        glob.glob(os.path.join(output_dir, "*.bin")) +
        glob.glob(os.path.join(output_dir, "*.safetensors")) +
        glob.glob(os.path.join(output_dir, "**", "*.safetensors"), recursive=True)
    )
    weight_files = list(set(weight_files))
    total_bytes = sum(os.path.getsize(f) for f in weight_files)

    print(f"\n  Weight files saved : {len(weight_files)}")
    for f in weight_files:
        print(f"    {f}  ({os.path.getsize(f)/1e6:.2f} MB)")
    print(f"  Total weight size  : {total_bytes/1e6:.2f} MB")

    if total_bytes == 0:
        raise RuntimeError("[DRY-RUN FAIL] No bytes written. save_pretrained() did not produce weights.")
    if total_bytes <= 100:  # 17 bytes = placeholder
        raise RuntimeError(
            f"[DRY-RUN FAIL] Only {total_bytes} bytes written — this is a placeholder, not real weights."
        )

    print("\n  ✅ DRY-RUN PASSED — full training pipeline is wired correctly.")
    next_call = "train_episodic_curriculum()" if episode_train else "train_curriculum()"
    print(f"  → Run {next_call} on a GPU for a production model (> 50 MB).")
    return trainer


# ──────────────────────────────────────────────
# 8. MULTI-STEP FULL EPISODE RUNNER
# ──────────────────────────────────────────────

def _model_completion(model, tokenizer, prompt: str, max_new_tokens: int = 160, temperature: float = 0.3) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def rollout_model_episode(
    model,
    tokenizer,
    task,
    person,
    seed: int,
    horizon: int = 5,
    temperature: float = 0.3,
) -> dict:
    """Closed-loop model rollout: generate one action, step env, repeat."""
    from core.lifestack_env import LifeStackEnv, LifeStackAction

    random.seed(seed)
    env = LifeStackEnv()
    env.reset(task=task, conflict=task.mutable_world)
    random.seed()

    history = []
    total_reward = 0.0
    for step in range(horizon):
        prompt = build_prompt_for_task(
            task,
            person,
            env.state.current_metrics,
            env.state.budget,
            seed=seed,
            step=step,
            event_descriptions=env.state.fired_event_ids,
        )
        if history:
            prompt += "\nPast actions:\n" + "\n".join(
                f"- step {h['step']}: {h['action_type']} -> {h['target']} (reward {h['reward']:.3f})"
                for h in history[-3:]
            )
            prompt += "\nReturn the next single JSON action only.\n"

        completion = _model_completion(model, tokenizer, prompt, max_new_tokens=128, temperature=temperature)
        try:
            data = _actions_from_completion(completion)[0]
            env_action = _to_lifestack_action(data, completion, actions_taken=1)
        except Exception:
            env_action = LifeStackAction(
                action_type="rest",
                target="time",
                metric_changes={},
                resource_cost={},
                actions_taken=0,
            )

        obs = env.step(env_action)
        step_reward = float(obs.reward or 0.0)
        total_reward += step_reward
        history.append({
            "step": step + 1,
            "action_type": env_action.action_type,
            "target": env_action.target,
            "reward": step_reward,
            "completion": completion,
            "success": bool(obs.metadata.get("success")),
            "failure": bool(obs.metadata.get("failure")),
            "events": obs.metadata.get("info", []),
        })
        if obs.done:
            break

    return {
        "total_reward": total_reward,
        "steps": history,
        "success": bool(history and history[-1].get("success")),
        "failure": bool(history and history[-1].get("failure")),
    }


def run_full_episode(
    model_dir: str = "./lifestack_model",
    n_episodes: int = 10,
    push_to_hub: bool = False,
    hub_repo_id: str = "lifestack-grpo",
):
    """
    Run multi-step episodes with the trained model (post-training evaluation).

    Each episode plays up to 5 sequential env steps so the model handles
    long-horizon decision chains, not just single actions.

    Args:
        model_dir:    Saved GRPO model directory.
        n_episodes:   Number of full episodes to roll out.
        push_to_hub:  If True, push model + tokenizer to HuggingFace Hub.
        hub_repo_id:  Hub repo id (e.g. "username/lifestack-grpo").
    """
    from core.lifestack_env import LifeStackEnv, LifeStackAction

    print("\n" + "=" * 60)
    print("🎮 MULTI-STEP FULL EPISODE RUNNER")
    print("=" * 60)

    # Load model — Unsloth first, HF+PEFT fallback
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir, max_seq_length=1024, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("  Loaded via Unsloth")
    except Exception as e:
        print(f"  Unsloth failed ({e}), using AutoModelForCausalLM + PeftModel")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    generator = TaskGenerator()
    graph = DependencyGraph()
    episode_rewards = []

    for ep in range(n_episodes):
        domain = ALL_DOMAINS[ep % len(ALL_DOMAINS)]
        ep_seed = ep * 31 + 7
        random.seed(ep_seed)
        task = generator.generate(domain=domain, difficulty=min(5, 1 + ep // 2))
        conflict = generate_conflict(min(5, 1 + ep // 2))
        random.seed()

        metrics = LifeMetrics()
        metrics = graph.cascade(metrics, {**task.mutable_world, **conflict.primary_disruption})
        budget_dict = task.constraints.get("budget", {})
        budget = ResourceBudget(
            time_hours=budget_dict.get("time", 20.0),
            money_dollars=budget_dict.get("money", 500.0),
            energy_units=budget_dict.get("energy", 100.0),
        )
        person = SimPerson(name="EvalAgent", openness=0.6, conscientiousness=0.7,
                           extraversion=0.5, agreeableness=0.6, neuroticism=0.4)

        env = LifeStackEnv()
        env.reset(task=task, conflict=task.mutable_world)

        ep_total = 0.0
        horizon = min(getattr(task, "horizon", 5), 5)

        for step in range(horizon):
            prompt = build_prompt_for_task(task, person, env.state.current_metrics,
                                           env.state.budget, seed=ep_seed, step=step)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=128, temperature=0.3, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            try:
                d = _load_first_json_object(completion)
                env_action = LifeStackAction(
                    action_type=d.get("action_type", "rest"),
                    target=d.get("route_id") or d.get("target_domain", "time"),
                    metric_changes=d.get("metric_changes", {}),
                    resource_cost=d.get("resource_cost", {}),
                    reasoning=d.get("reasoning", ""),
                    actions_taken=1,
                )
            except Exception:
                env_action = LifeStackAction(action_type="rest", target="time",
                                              metric_changes={}, resource_cost={}, actions_taken=0)
            obs = env.step(env_action)
            ep_total += obs.reward
            if obs.done:
                break

        episode_rewards.append(ep_total)
        print(f"  Ep {ep+1:2d}/{n_episodes} | {domain:20s} | reward={ep_total:.3f}")

    mean_r = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"\n  Mean episode reward : {mean_r:.3f}")
    print(f"  Best episode reward : {max(episode_rewards):.3f}")

    if push_to_hub:
        try:
            print(f"\n  Pushing to HuggingFace Hub: {hub_repo_id} ...")
            model.push_to_hub(hub_repo_id)
            tokenizer.push_to_hub(hub_repo_id)
            print(f"  ✅ Pushed → https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"  ❌ push_to_hub failed: {e}")
            print("  Tip: `huggingface-cli login` or set HF_TOKEN env var first.")

    return episode_rewards


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LifeStack GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (CPU, no GPU needed)
  python train_trl.py --dry-run

  # Fresh full run
  python train_trl.py --stages 5 --prompts-per-stage 200

  # Resume after Colab / Kaggle session cut
  python train_trl.py --resume

  # Manually restart from stage 3
  python train_trl.py --start-stage 3

  # Run multi-step episodes with the trained model
  python train_trl.py --full-episode --output-dir ./lifestack_model

  # Train v2 with episode-level rewards after a short single-step warm-up
  python train_trl.py --episode-train --stages 2 --episodes-per-stage 40 --episode-horizon 3

  # Validate the episodic path locally
  python train_trl.py --episode-train --dry-run

  # Train then push to HuggingFace Hub
  python train_trl.py --stages 5 --push-to-hub --hub-repo-id username/lifestack-grpo
        """
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 training step on 4 prompts to validate the full pipeline (no GPU required)."
    )
    parser.add_argument(
        "--stages", type=int, default=5,
        help="Number of curriculum stages (default: 5)."
    )
    parser.add_argument(
        "--prompts-per-stage", type=int, default=100,
        help="Prompts per curriculum stage (default: 100)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./lifestack_model",
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last saved checkpoint + curriculum_state.json."
    )
    parser.add_argument(
        "--start-stage", type=int, default=None,
        help="Force-start from a specific stage number (1-indexed). Ignores curriculum_state.json."
    )
    parser.add_argument(
        "--full-episode", action="store_true",
        help="Run multi-step episodes with the trained model (post-training evaluation)."
    )
    parser.add_argument(
        "--episode-train", action="store_true",
        help="Train with episode-level action-sequence rewards instead of single-step rewards."
    )
    parser.add_argument(
        "--episode-horizon", type=int, default=3,
        help="Maximum number of actions in an episode-training completion (default: 3)."
    )
    parser.add_argument(
        "--episodes-per-stage", type=int, default=40,
        help="Episode prompts per episodic stage (default: 40)."
    )
    parser.add_argument(
        "--episode-warmup-stages", type=int, default=1,
        help="Single-step warm-up stages before episodic fine-tuning (default: 1)."
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push trained model to HuggingFace Hub after training or --full-episode."
    )
    parser.add_argument(
        "--hub-repo-id", type=str, default="lifestack-grpo",
        help="HuggingFace Hub repository ID for --push-to-hub (default: lifestack-grpo)."
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=2048,
        help="Token budget for the left-truncated prompt in GRPO (default: 2048).",
    )
    parser.add_argument(
        "--max-completion-length", type=int, default=224,
        help="Token budget for each GRPO completion in single-step curriculum training (default: 224).",
    )
    parser.add_argument(
        "--episodic-max-completion", type=int, default=0,
        help="Override episodic max completion length; 0 = auto (default: 0).",
    )
    parser.add_argument(
        "--num-train-epochs", type=int, default=1,
        help=(
            "Training epochs per stage.  "
            "Previous run used 1 epoch (~10 optimizer steps).  "
            "Set to 3–5 for a longer run without changing data size."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(
            output_dir="./lifestack_model_dryrun",
            episode_train=args.episode_train,
            episode_horizon=args.episode_horizon,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            episodic_max_completion=None
            if args.episodic_max_completion == 0
            else int(args.episodic_max_completion),
        )
    elif args.full_episode:
        run_full_episode(
            model_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
        )
    elif args.episode_train:
        warmup_trainer = None
        if args.episode_warmup_stages > 0 and not args.resume:
            warmup_trainer = train_curriculum(
                n_stages=args.episode_warmup_stages,
                n_prompts_per_stage=args.prompts_per_stage,
                output_dir=args.output_dir,
                resume=False,
                start_stage=args.start_stage,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
            )
        trainer = train_episodic_curriculum(
            n_stages=args.stages,
            n_episodes_per_stage=args.episodes_per_stage,
            output_dir=args.output_dir,
            episode_horizon=args.episode_horizon,
            model=warmup_trainer.model if warmup_trainer else None,
            tokenizer=warmup_trainer.processing_class if warmup_trainer else None,
            resume=args.resume,
            start_stage=args.start_stage if not warmup_trainer else None,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=None
            if args.episodic_max_completion == 0
            else int(args.episodic_max_completion),
            num_train_epochs=args.num_train_epochs,
        )
        validate_saved_model(args.output_dir)
        if args.push_to_hub and trainer:
            try:
                print(f"\nPushing episodic model to HuggingFace Hub: {args.hub_repo_id} ...")
                trainer.model.push_to_hub(args.hub_repo_id)
                trainer.processing_class.push_to_hub(args.hub_repo_id)
                print(f"✅ Pushed → https://huggingface.co/{args.hub_repo_id}")
            except Exception as e:
                print(f"❌ push_to_hub failed: {e}")
    else:
        trainer = train_curriculum(
            n_stages=args.stages,
            n_prompts_per_stage=args.prompts_per_stage,
            output_dir=args.output_dir,
            resume=args.resume,
            start_stage=args.start_stage,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
        )
        validate_saved_model(args.output_dir)
        evaluate_and_plot(args.output_dir)
        if args.push_to_hub:
            try:
                print(f"\nPushing to HuggingFace Hub: {args.hub_repo_id} ...")
                trainer.model.push_to_hub(args.hub_repo_id)
                trainer.processing_class.push_to_hub(args.hub_repo_id)
                print(f"✅ Pushed → https://huggingface.co/{args.hub_repo_id}")
            except Exception as e:
                print(f"❌ push_to_hub failed: {e}")
