"""
plot_training.py  –  Parse a train_run_vN.log and emit publication-quality plots.

Usage (on the HF Jupyter server or Colab):
    python scripts/plot_training.py --log train_run_v4.log --out plots/

Outputs (in --out dir):
    reward_curve.png          – rolling-mean total reward + raw scatter
    reward_components.png     – per reward-function breakdown
    loss_curve.png            – HF Trainer loss
    training_summary.png      – 4-panel combined figure (perfect for slides / README)

The parser handles three log formats emitted by train_trl.py:
  1. [step N] r0=X.XXX | r_lt=Y.YYY | comp1=A | comp2=B ...   (custom reward log)
  2. {'loss': '...', 'reward': '...', 'rewards/fn/mean': '...', 'epoch': '...', ...}
     (HF/TRL Trainer dicts – values may be quoted strings OR bare floats)
  3. Raw JSON-lines in training_logs/generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless – safe on Jupyter too
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── colour palette (dark, print-friendly) ─────────────────────────────────────
C = {
    "reward":     "#4C8BF5",   # Google-blue
    "longterm":   "#34A853",   # Google-green
    "task":       "#FBBC05",   # Google-yellow
    "milestone":  "#EA4335",   # Google-red
    "replan":     "#9B59B6",   # purple
    "loss":       "#E67E22",   # orange
    "bg":         "#0F0F0F",
    "grid":       "#2A2A2A",
    "text":       "#E0E0E0",
    "band":       "#4C8BF5",
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["bg"],
    "axes.edgecolor":   C["grid"],
    "axes.labelcolor":  C["text"],
    "xtick.color":      C["text"],
    "ytick.color":      C["text"],
    "text.color":       C["text"],
    "grid.color":       C["grid"],
    "legend.facecolor": "#1A1A1A",
    "legend.edgecolor": C["grid"],
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

_STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+"
    r"r0=([+-]?\d+\.\d+).*?"
    r"r_lt=([+-]?\d+\.\d+)"
    r"((?:\s*\|\s*\w+=[-+]?\d+\.\d+)*)"
)
_COMP_RE  = re.compile(r"(\w+)=([-+]?\d+\.\d+)")
_DICT_RE  = re.compile(r"\{[^{}]+\}")


def _safe_float(v) -> float | None:
    """Convert a value that may be a bare float or a quoted string like '-8.941e-09'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_dict_line(raw: str) -> dict[str, Any] | None:
    """Parse a Python-dict-style log line where values may be quoted strings or bare floats."""
    # Strategy 1: replace single-quotes → double-quotes for JSON
    try:
        js = raw.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        return json.loads(js)
    except Exception:
        pass
    # Strategy 2: eval (safe fallback – no builtins)
    try:
        return eval(raw, {"__builtins__": {}})  # noqa: S307
    except Exception:
        return None


# Known per-reward-function keys emitted by TRL GRPO trainer
_REWARD_FN_KEYS = [
    "rewards/reward_format_fn/mean",
    "rewards/reward_clean_eos_fn/mean",
    "rewards/reward_route_target_fn/mean",
    "rewards/reward_task_success_fn/mean",
    "rewards/reward_milestone_fn/mean",
    "rewards/reward_replan_fn/mean",
    "rewards/reward_longterm_fn/mean",
    "rewards/reward_compact_fn/mean",
    "rewards/reward_human_feedback_fn/mean",
]

# Human-friendly short labels for the above keys
_REWARD_FN_LABELS = {
    "rewards/reward_format_fn/mean":         "Format",
    "rewards/reward_clean_eos_fn/mean":       "Clean EOS",
    "rewards/reward_route_target_fn/mean":    "Route/Target",
    "rewards/reward_task_success_fn/mean":    "Task success",
    "rewards/reward_milestone_fn/mean":       "Milestone",
    "rewards/reward_replan_fn/mean":          "Replan",
    "rewards/reward_longterm_fn/mean":        "Long-term",
    "rewards/reward_compact_fn/mean":         "Compact",
    "rewards/reward_human_feedback_fn/mean":  "Human feedback",
}


def parse_log(log_path: str | Path) -> dict[str, list]:
    """Return parsed series from a train_run_vN.log file."""
    log_path = Path(log_path)
    if not log_path.exists():
        sys.exit(f"[plot_training] Log file not found: {log_path}")

    series: dict[str, list] = {
        # Format-1 fields (custom [step N] r0=... lines)
        "step": [], "reward": [], "longterm": [],
        "task_success": [], "milestone": [], "replan": [],
        # Format-2 fields (HF Trainer dict lines)
        "train_step": [], "loss": [], "lr": [], "epoch": [],
        "trainer_reward": [], "reward_std": [],
    }
    # Per-reward-function series (populated from Format-2 dicts)
    fn_series: dict[str, list] = {k: [] for k in _REWARD_FN_KEYS}

    text = log_path.read_text(errors="replace")

    # ── Format 1: [step N] r0=... r_lt=... ────────────────────────────────────
    for m in _STEP_RE.finditer(text):
        step       = int(m.group(1))
        r0         = float(m.group(2))
        r_lt       = float(m.group(3))
        comp_block = m.group(4) or ""
        comps      = dict(_COMP_RE.findall(comp_block))

        series["step"].append(step)
        series["reward"].append(r0)
        series["longterm"].append(r_lt)
        series["task_success"].append(float(comps.get("task_success", comps.get("completion", 0.0))))
        series["milestone"].append(float(comps.get("milestone", 0.0)))
        series["replan"].append(float(comps.get("replan", 0.0)))

    # ── Format 2: {'loss': '...', 'reward': '...', 'rewards/fn/mean': '...', ...}
    # Values are quoted strings in the v4 log (e.g. 'reward': '0.5788')
    global_step_counter = 0
    for m in _DICT_RE.finditer(text):
        d = _parse_dict_line(m.group(0))
        if not isinstance(d, dict):
            continue
        # Skip the final train_runtime summary dict (no per-step metrics)
        if "train_runtime" in d:
            continue
        if "loss" not in d and "reward" not in d:
            continue

        global_step_counter += 1
        explicit_step = d.get("step") or d.get("global_step")
        step_val = int(float(explicit_step)) if explicit_step is not None else global_step_counter
        series["train_step"].append(step_val)

        loss_v = _safe_float(d.get("loss"))
        if loss_v is not None:
            series["loss"].append(loss_v)

        lr_v = _safe_float(d.get("learning_rate"))
        if lr_v is not None:
            series["lr"].append(lr_v)

        ep_v = _safe_float(d.get("epoch"))
        if ep_v is not None:
            series["epoch"].append(ep_v)

        rw_v = _safe_float(d.get("reward"))
        if rw_v is not None:
            series["trainer_reward"].append(rw_v)

        rs_v = _safe_float(d.get("reward_std"))
        if rs_v is not None:
            series["reward_std"].append(rs_v)

        # Per-function reward means
        for key in _REWARD_FN_KEYS:
            fv = _safe_float(d.get(key))
            if fv is not None:
                fn_series[key].append(fv)

    # ── Format 3: JSONL generations log ───────────────────────────────────────
    jsonl_path = log_path.parent / "training_logs" / "generations.jsonl"
    if jsonl_path.exists() and not series["step"]:
        print(f"[plot_training] Loading supplemental JSONL: {jsonl_path}")
        for line in jsonl_path.read_text().splitlines():
            try:
                d = json.loads(line)
                series["step"].append(int(d.get("step", 0)))
                series["reward"].append(float(d.get("reward", 0)))
                series["longterm"].append(float(d.get("longterm_reward", 0)))
                comps = d.get("breakdown", {}).get("components", {})
                series["task_success"].append(float(comps.get("completion", comps.get("task_success", 0))))
                series["milestone"].append(float(comps.get("milestone", 0)))
                series["replan"].append(float(comps.get("replan", 0)))
            except Exception:
                pass

    # Attach fn_series so callers can access it
    series["_fn_series"] = fn_series  # type: ignore[assignment]

    total_reward  = len(series["step"]) or len(series["trainer_reward"])
    total_trainer = len(series["loss"])
    print(f"[plot_training] Parsed {len(series['step'])} custom-format steps, "
          f"{len(series['trainer_reward'])} trainer reward entries, "
          f"{total_trainer} loss entries.")
    active_fns = [k for k, v in fn_series.items() if v]
    if active_fns:
        print(f"[plot_training] Per-function reward series found: "
              f"{[_REWARD_FN_LABELS[k] for k in active_fns]}")

    if total_reward == 0 and total_trainer == 0:
        sys.exit("[plot_training] Nothing parsed. Check the log path and format.")

    return series


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_mean(arr: list[float], w: int = 20) -> np.ndarray:
    if len(arr) < 2:
        return np.array(arr, dtype=float)
    a = np.array(arr, dtype=float)
    kernel = np.ones(w) / w
    padded = np.pad(a, (w - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ═══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate_trend(ax, x, y_smooth):
    """Add a start/end annotation showing net improvement."""
    if len(y_smooth) < 4:
        return
    start_val = float(np.mean(y_smooth[:max(1, len(y_smooth)//10)]))
    end_val   = float(np.mean(y_smooth[-max(1, len(y_smooth)//10):]))
    delta     = end_val - start_val
    sign      = "+" if delta >= 0 else ""
    ax.annotate(
        f"Δ = {sign}{delta:.3f}",
        xy=(x[-1], y_smooth[-1]),
        xytext=(-60, 12),
        textcoords="offset points",
        fontsize=9,
        color=C["text"],
        arrowprops=dict(arrowstyle="->", color=C["text"], lw=0.8),
    )


def plot_reward_curve(series: dict, out_dir: Path) -> Path:
    # Prefer custom-format steps; fall back to trainer_reward from HF Trainer dicts
    if series["reward"]:
        steps   = np.array(series["step"])
        rewards = np.array(series["reward"])
        xlabel  = "Reward Call Step"
        ylabel  = "Immediate Reward  r₀"
        std_arr = None
    elif series["trainer_reward"]:
        n = len(series["trainer_reward"])
        steps   = np.array(series["train_step"] if series["train_step"] else range(1, n + 1))
        rewards = np.array(series["trainer_reward"])
        xlabel  = "Training Step"
        ylabel  = "Mean Reward (per GRPO step)"
        std_arr = np.array(series["reward_std"]) if len(series["reward_std"]) == n else None
    else:
        print("[plot_training] No reward data – skipping reward_curve.png")
        return None

    w = max(2, len(steps) // 10)
    smooth = rolling_mean(rewards.tolist(), w)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(steps, rewards, s=30, alpha=0.6, color=C["reward"], label="Per-step reward", zorder=2)
    ax.plot(steps, smooth, lw=2.5, color=C["reward"], label=f"Rolling mean (w={w})", zorder=3)

    # shade ±1 std band (from logged reward_std if available, else computed)
    if std_arr is not None:
        ax.fill_between(steps, rewards - std_arr, rewards + std_arr,
                        alpha=0.2, color=C["band"], label="±1 σ (logged)")
    elif len(rewards) >= w:
        sq_sm = rolling_mean((rewards ** 2).tolist(), w)
        std_sm = np.sqrt(np.maximum(0, sq_sm - smooth ** 2))
        ax.fill_between(steps, smooth - std_sm, smooth + std_sm,
                        alpha=0.18, color=C["band"], label="±1 σ band")

    _annotate_trend(ax, steps, smooth)
    ax.axhline(0, color=C["grid"], lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("LifeStack GRPO – Reward Curve (v4 run)", pad=12)
    ax.legend(framealpha=0.7, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    out = out_dir / "reward_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training] Saved: {out}")
    return out


def plot_reward_components(series: dict, out_dir: Path) -> Path:
    fn_series = series.get("_fn_series", {})

    # Build component dict: prefer per-fn rewards from HF Trainer; fall back to custom format
    active_fns = {k: v for k, v in fn_series.items() if v}
    use_custom = (not active_fns) and series["step"]

    if not active_fns and not use_custom:
        print("[plot_training] No component reward data – skipping reward_components.png")
        return None

    if use_custom:
        steps_raw = series["step"]
        components = {
            "Long-term":   (series["longterm"],    C["longterm"]),
            "Task success":(series["task_success"],C["task"]),
            "Milestone":   (series["milestone"],   C["milestone"]),
            "Replan":      (series["replan"],       C["replan"]),
        }
    else:
        n = max(len(v) for v in active_fns.values())
        steps_raw = (series["train_step"] if len(series["train_step"]) == n
                     else list(range(1, n + 1)))
        palette = [C["reward"], C["longterm"], C["task"], C["milestone"],
                   C["replan"], "#00BCD4", "#FF5722", "#9C27B0", "#607D8B"]
        components = {
            _REWARD_FN_LABELS[k]: (v, palette[i % len(palette)])
            for i, (k, v) in enumerate(active_fns.items())
        }

    steps = np.array(steps_raw)
    ncols = 2
    nrows = (len(components) + 1) // ncols
    w = max(2, len(steps) // 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    fig.suptitle("LifeStack GRPO – Per-Function Reward Components (v4 run)",
                 y=1.01, fontsize=13)

    for ax, (label, (vals, color)) in zip(axes_flat, components.items()):
        vals_arr = np.array(vals[:len(steps)], dtype=float)
        if len(vals_arr) == 0 or (vals_arr == 0).all():
            ax.text(0.5, 0.5, "No signal", ha="center", va="center",
                    transform=ax.transAxes, color=C["text"], fontsize=11)
            ax.set_title(label, fontsize=10)
            continue
        xs = steps[:len(vals_arr)]
        smooth = rolling_mean(vals_arr.tolist(), w)
        ax.scatter(xs, vals_arr, s=25, alpha=0.5, color=color)
        ax.plot(xs, smooth, lw=2.0, color=color)
        ax.axhline(0, color=C["grid"], lw=0.6, ls="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Reward")
        ax.grid(True, lw=0.3)
        _annotate_trend(ax, xs, smooth)

    # Hide any unused subplot panels
    for ax in list(axes_flat)[len(components):]:
        ax.set_visible(False)

    for ax in (axes[-1] if nrows > 1 else [axes]):
        if hasattr(ax, '__iter__'):
            for a in ax:
                a.set_xlabel("Training Step")
        else:
            ax.set_xlabel("Training Step")

    fig.tight_layout()
    out = out_dir / "reward_components.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training] Saved: {out}")
    return out


def plot_loss_curve(series: dict, out_dir: Path) -> Path:
    loss = series["loss"]
    steps = series["train_step"]
    if len(loss) == 0:
        print("[plot_training] No loss data in log – skipping loss_curve.png")
        return None

    x = np.array(steps if steps else range(len(loss)), dtype=float)
    y = np.array(loss, dtype=float)
    w = max(5, len(y) // 20)
    smooth = rolling_mean(y.tolist(), w)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y, s=8, alpha=0.3, color=C["loss"], label="Raw loss")
    ax.plot(x, smooth, lw=2.0, color=C["loss"], label=f"Rolling mean (w={w})")
    _annotate_trend(ax, x, smooth)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("LifeStack GRPO – Training Loss (v4 run)", pad=12)
    ax.legend(framealpha=0.7, fontsize=9)
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    out = out_dir / "loss_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training] Saved: {out}")
    return out


def plot_summary_4panel(series: dict, out_dir: Path) -> Path:
    """4-panel combined figure: reward, reward_std, components, loss."""
    fn_series  = series.get("_fn_series", {})

    # Determine reward source
    has_custom  = len(series["step"]) > 0
    has_trainer = len(series["trainer_reward"]) > 0
    has_loss    = len(series["loss"]) > 0
    has_fn      = any(v for v in fn_series.values())

    if not has_custom and not has_trainer and not has_loss:
        return None

    # Build reward arrays
    if has_custom:
        steps_r  = np.array(series["step"])
        rewards  = np.array(series["reward"], dtype=float)
        longterm = np.array(series["longterm"], dtype=float)
        rlabel   = "r₀"
    elif has_trainer:
        n = len(series["trainer_reward"])
        steps_r  = np.array(series["train_step"] if series["train_step"] else range(1, n + 1))
        rewards  = np.array(series["trainer_reward"], dtype=float)
        longterm = (np.array(series["reward_std"], dtype=float)
                    if len(series["reward_std"]) == n else None)
        rlabel   = "Mean reward"
    else:
        steps_r = rewards = longterm = None
        rlabel = ""

    n_loss = len(series["loss"])
    steps_t = np.array(series["train_step"][:n_loss] if series["train_step"]
                       else range(1, n_loss + 1), dtype=float)

    w_r = max(2, len(steps_r) // 10) if steps_r is not None else 2
    w_l = max(2, n_loss // 10) if n_loss else 2

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("LifeStack GRPO Training Evidence  –  Run v4", fontsize=14, y=1.00)
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.36)
    ax_r  = fig.add_subplot(gs[0, 0])
    ax_lt = fig.add_subplot(gs[0, 1])
    ax_c  = fig.add_subplot(gs[1, 0])
    ax_l  = fig.add_subplot(gs[1, 1])

    # ── Panel 1: total reward ──────────────────────────────────────────────────
    if rewards is not None:
        sm = rolling_mean(rewards.tolist(), w_r)
        ax_r.scatter(steps_r, rewards, s=40, alpha=0.6, color=C["reward"])
        ax_r.plot(steps_r, sm, lw=2.5, color=C["reward"])
        ax_r.axhline(0, color=C["grid"], lw=0.7, ls="--")
        ax_r.set_title(f"Total Reward  ({rlabel})", fontsize=10)
        ax_r.set_xlabel("Step"); ax_r.set_ylabel(rlabel)
        _annotate_trend(ax_r, steps_r, sm)
    else:
        ax_r.text(0.5, 0.5, "No reward data", ha="center", va="center",
                  transform=ax_r.transAxes, color=C["text"])

    # ── Panel 2: reward std OR longterm ───────────────────────────────────────
    if longterm is not None and len(longterm) == len(steps_r):
        sm_lt = rolling_mean(longterm.tolist(), w_r)
        ax_lt.scatter(steps_r, longterm, s=40, alpha=0.6, color=C["longterm"])
        ax_lt.plot(steps_r, sm_lt, lw=2.5, color=C["longterm"])
        ax_lt.axhline(0, color=C["grid"], lw=0.7, ls="--")
        p2_title = "Long-term Reward" if has_custom else "Reward Std Dev"
        ax_lt.set_title(p2_title, fontsize=10)
        ax_lt.set_xlabel("Step"); ax_lt.set_ylabel(p2_title)
        _annotate_trend(ax_lt, steps_r, sm_lt)
    else:
        ax_lt.text(0.5, 0.5, "No secondary reward data",
                   ha="center", va="center", transform=ax_lt.transAxes, color=C["text"])

    # ── Panel 3: per-function rewards (line chart) ─────────────────────────────
    palette = [C["reward"], C["longterm"], C["task"], C["milestone"],
               C["replan"], "#00BCD4", "#FF5722", "#607D8B"]
    active_fns = {k: v for k, v in fn_series.items() if v}
    if active_fns:
        n_fn = max(len(v) for v in active_fns.values())
        xs = np.array(series["train_step"][:n_fn] if series["train_step"]
                      else range(1, n_fn + 1))
        for i, (key, vals) in enumerate(active_fns.items()):
            col = palette[i % len(palette)]
            va  = np.array(vals[:len(xs)], dtype=float)
            sm  = rolling_mean(va.tolist(), max(2, len(va) // 10))
            ax_c.plot(xs[:len(va)], sm, lw=2.0, color=col,
                      label=_REWARD_FN_LABELS[key])
        ax_c.axhline(0, color=C["grid"], lw=0.6, ls="--")
        ax_c.set_title("Per-Function Rewards", fontsize=10)
        ax_c.set_xlabel("Step"); ax_c.set_ylabel("Reward")
        ax_c.legend(fontsize=7, framealpha=0.6)
    elif has_custom:
        comp_data = [
            ("Task success", np.array(series["task_success"], dtype=float), C["task"]),
            ("Milestone",    np.array(series["milestone"],    dtype=float), C["milestone"]),
            ("Replan",       np.array(series["replan"],       dtype=float), C["replan"]),
        ]
        bottoms = np.zeros(len(steps_r))
        bar_w = max(1.0, (steps_r[-1] - steps_r[0]) / max(1, len(steps_r)) * 0.9)
        for label, vals, color in comp_data:
            clipped = np.clip(vals, 0, None)
            ax_c.bar(steps_r, clipped, bottom=bottoms, color=color,
                     alpha=0.7, width=bar_w, label=label)
            bottoms += clipped
        ax_c.set_title("Component Rewards (stacked)", fontsize=10)
        ax_c.set_xlabel("Step"); ax_c.set_ylabel("Stacked reward")
        ax_c.legend(fontsize=8, framealpha=0.6)
    else:
        ax_c.text(0.5, 0.5, "No component data", ha="center", va="center",
                  transform=ax_c.transAxes, color=C["text"])

    # ── Panel 4: loss ─────────────────────────────────────────────────────────
    if has_loss:
        ly  = np.array(series["loss"], dtype=float)
        sm_l = rolling_mean(ly.tolist(), w_l)
        ax_l.scatter(steps_t, ly, s=40, alpha=0.5, color=C["loss"])
        ax_l.plot(steps_t, sm_l, lw=2.5, color=C["loss"])
        ax_l.set_title("Training Loss", fontsize=10)
        ax_l.set_xlabel("Step"); ax_l.set_ylabel("Loss")
        _annotate_trend(ax_l, steps_t, sm_l)
    else:
        ax_l.text(0.5, 0.5, "No loss entries in log",
                  ha="center", va="center", transform=ax_l.transAxes,
                  color=C["text"], fontsize=9)

    for ax in [ax_r, ax_lt, ax_c, ax_l]:
        ax.grid(True, lw=0.3)

    fig.tight_layout()
    out = out_dir / "training_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training] Saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Generate training plots from a train_run_vN.log")
    ap.add_argument("--log",  default="train_run_v4.log",
                    help="Path to the training log file (default: train_run_v4.log)")
    ap.add_argument("--out",  default="plots",
                    help="Output directory for PNG files (default: ./plots/)")
    ap.add_argument("--window", type=int, default=0,
                    help="Rolling-mean window size (0 = auto)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = parse_log(args.log)

    plot_reward_curve(series, out_dir)
    plot_reward_components(series, out_dir)
    plot_loss_curve(series, out_dir)
    plot_summary_4panel(series, out_dir)

    print(f"\n[plot_training] All plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
