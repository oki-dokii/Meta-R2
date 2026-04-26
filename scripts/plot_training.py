"""
plot_training.py  –  Parse a train_run_vN.log and emit publication-quality plots.

Usage (on the HF Jupyter server or Colab):
    python scripts/plot_training.py --log train_run_v4.log --out plots/

Outputs (in --out dir):
    reward_curve.png          – rolling-mean r0 + raw scatter
    reward_components.png     – task_success / milestone / replan / longterm
    loss_curve.png            – HF Trainer loss (if present)
    training_summary.png      – 4-panel combined figure (perfect for slides / README)

The parser handles three log formats emitted by train_trl.py:
  1. [step N] r0=X.XXX | r_lt=Y.YYY | comp1=A | comp2=B ...
  2. {'loss': X, 'reward': Y, 'learning_rate': Z, 'epoch': W, 'step': N, ...}
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


def _parse_dict_line(raw: str) -> dict[str, Any] | None:
    """Try to parse a line that looks like a Python dict."""
    try:
        # Replace single-quotes with double-quotes conservatively
        js = raw.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        return json.loads(js)
    except Exception:
        pass
    try:
        return eval(raw, {"__builtins__": {}})  # noqa: S307 – sandboxed eval
    except Exception:
        return None


def parse_log(log_path: str | Path) -> dict[str, list]:
    """Return parsed series from a train_run_vN.log file."""
    log_path = Path(log_path)
    if not log_path.exists():
        sys.exit(f"[plot_training] Log file not found: {log_path}")

    series: dict[str, list] = {
        "step": [], "reward": [], "longterm": [],
        "task_success": [], "milestone": [], "replan": [],
        # from HF Trainer
        "train_step": [], "loss": [], "lr": [], "epoch": [],
        "trainer_reward": [], "reward_std": [],
    }

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

    # ── Format 2: {'loss': ..., 'step': ..., 'reward': ...} ───────────────────
    for m in _DICT_RE.finditer(text):
        d = _parse_dict_line(m.group(0))
        if not isinstance(d, dict):
            continue
        if "loss" in d or "reward" in d:
            s = d.get("step") or d.get("global_step")
            if s is not None:
                series["train_step"].append(int(s))
            if "loss" in d:
                series["loss"].append(float(d["loss"]))
            if "learning_rate" in d:
                series["lr"].append(float(d["learning_rate"]))
            if "epoch" in d:
                series["epoch"].append(float(d["epoch"]))
            if "reward" in d:
                series["trainer_reward"].append(float(d["reward"]))
            if "reward_std" in d:
                series["reward_std"].append(float(d["reward_std"]))

    # ── Format 3: JSONL generations log ───────────────────────────────────────
    jsonl_path = log_path.parent / "training_logs" / "generations.jsonl"
    if jsonl_path.exists() and not series["step"]:
        print(f"[plot_training] No step-format lines found in log; loading {jsonl_path}")
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

    total_reward = len(series["step"])
    total_trainer = len(series["loss"])
    print(f"[plot_training] Parsed {total_reward} reward steps, {total_trainer} trainer loss entries.")
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
    steps   = np.array(series["step"])
    rewards = np.array(series["reward"])
    if len(steps) == 0:
        print("[plot_training] No reward data – skipping reward_curve.png")
        return None

    w = max(10, len(steps) // 30)
    smooth = rolling_mean(rewards.tolist(), w)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(steps, rewards, s=6, alpha=0.25, color=C["reward"], label="Raw r₀", zorder=2)
    ax.plot(steps, smooth, lw=2.0, color=C["reward"], label=f"Rolling mean (w={w})", zorder=3)

    # shade ±1 std band
    arr = rewards
    if len(arr) >= w:
        std_smooth = rolling_mean((arr ** 2).tolist(), w)
        std_smooth = np.sqrt(np.maximum(0, std_smooth - smooth ** 2))
        ax.fill_between(steps, smooth - std_smooth, smooth + std_smooth,
                        alpha=0.18, color=C["band"], label="±1 σ band")

    _annotate_trend(ax, steps, smooth)
    ax.axhline(0, color=C["grid"], lw=0.8, ls="--")
    ax.set_xlabel("Reward Call Step")
    ax.set_ylabel("Immediate Reward  r₀")
    ax.set_title("LifeStack GRPO – Reward Curve (v4 run)", pad=12)
    ax.legend(framealpha=0.7, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, lw=0.4)
    fig.tight_layout()
    out = out_dir / "reward_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training] Saved: {out}")
    return out


def plot_reward_components(series: dict, out_dir: Path) -> Path:
    steps = np.array(series["step"])
    if len(steps) == 0:
        return None

    components = {
        "Longterm":    (series["longterm"],    C["longterm"]),
        "Task success":(series["task_success"],C["task"]),
        "Milestone":   (series["milestone"],   C["milestone"]),
        "Replan":      (series["replan"],       C["replan"]),
    }
    w = max(10, len(steps) // 30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle("LifeStack GRPO – Reward Components (v4 run)", y=1.01, fontsize=13)

    for ax, (label, (vals, color)) in zip(axes.flat, components.items()):
        vals_arr = np.array(vals, dtype=float)
        if vals_arr.sum() == 0:
            ax.text(0.5, 0.5, "No signal", ha="center", va="center",
                    transform=ax.transAxes, color=C["text"], fontsize=11)
            ax.set_title(label, fontsize=10)
            continue
        smooth = rolling_mean(vals_arr.tolist(), w)
        ax.scatter(steps, vals_arr, s=5, alpha=0.2, color=color)
        ax.plot(steps, smooth, lw=2.0, color=color)
        ax.axhline(0, color=C["grid"], lw=0.6, ls="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Reward component")
        ax.grid(True, lw=0.3)
        _annotate_trend(ax, steps, smooth)

    for ax in axes[1]:
        ax.set_xlabel("Reward Call Step")

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
    """4-panel combined figure: reward, longterm, components stacked, loss."""
    steps_r = np.array(series["step"])
    steps_t = np.array(series["train_step"] or range(len(series["loss"])), dtype=float)
    has_reward = len(steps_r) > 0
    has_loss   = len(series["loss"]) > 0

    if not has_reward and not has_loss:
        return None

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("LifeStack GRPO Training Evidence  –  Run v4", fontsize=14, y=1.00)

    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
    ax_r  = fig.add_subplot(gs[0, 0])   # top-left:  reward
    ax_lt = fig.add_subplot(gs[0, 1])   # top-right: longterm
    ax_c  = fig.add_subplot(gs[1, 0])   # bot-left:  components stacked
    ax_l  = fig.add_subplot(gs[1, 1])   # bot-right: loss

    w_r = max(10, len(steps_r) // 30) if has_reward else 10
    w_l = max(5,  len(series["loss"]) // 20) if has_loss else 5

    # ── Panel 1: immediate reward ──────────────────────────────────────────────
    if has_reward:
        rw = np.array(series["reward"], dtype=float)
        sm = rolling_mean(rw.tolist(), w_r)
        ax_r.scatter(steps_r, rw, s=5, alpha=0.2, color=C["reward"])
        ax_r.plot(steps_r, sm, lw=2.0, color=C["reward"])
        ax_r.axhline(0, color=C["grid"], lw=0.7, ls="--")
        ax_r.set_title("Immediate Reward  r₀", fontsize=10)
        ax_r.set_xlabel("Step"); ax_r.set_ylabel("r₀")
        _annotate_trend(ax_r, steps_r, sm)
    else:
        ax_r.text(0.5, 0.5, "No data", ha="center", va="center",
                  transform=ax_r.transAxes, color=C["text"])

    # ── Panel 2: longterm reward ───────────────────────────────────────────────
    if has_reward:
        lt = np.array(series["longterm"], dtype=float)
        sm_lt = rolling_mean(lt.tolist(), w_r)
        ax_lt.scatter(steps_r, lt, s=5, alpha=0.2, color=C["longterm"])
        ax_lt.plot(steps_r, sm_lt, lw=2.0, color=C["longterm"])
        ax_lt.axhline(0, color=C["grid"], lw=0.7, ls="--")
        ax_lt.set_title("Long-term Reward  r_lt", fontsize=10)
        ax_lt.set_xlabel("Step"); ax_lt.set_ylabel("r_lt")
        _annotate_trend(ax_lt, steps_r, sm_lt)

    # ── Panel 3: component area chart ─────────────────────────────────────────
    if has_reward:
        comp_data = {
            "Task success": (np.array(series["task_success"], dtype=float), C["task"]),
            "Milestone":    (np.array(series["milestone"],    dtype=float), C["milestone"]),
            "Replan":       (np.array(series["replan"],       dtype=float), C["replan"]),
        }
        bottoms = np.zeros(len(steps_r))
        for label, (vals, color) in comp_data.items():
            clipped = np.clip(vals, 0, None)
            ax_c.bar(steps_r, clipped, bottom=bottoms,
                     color=color, alpha=0.7, width=max(1, (steps_r[-1]-steps_r[0])/len(steps_r)*0.9),
                     label=label)
            bottoms += clipped
        ax_c.set_title("Component Rewards (stacked)", fontsize=10)
        ax_c.set_xlabel("Step"); ax_c.set_ylabel("Stacked reward")
        ax_c.legend(fontsize=8, framealpha=0.6)

    # ── Panel 4: loss ─────────────────────────────────────────────────────────
    if has_loss:
        ly = np.array(series["loss"], dtype=float)
        sm_l = rolling_mean(ly.tolist(), w_l)
        ax_l.scatter(steps_t, ly, s=8, alpha=0.3, color=C["loss"])
        ax_l.plot(steps_t, sm_l, lw=2.0, color=C["loss"])
        ax_l.set_title("Training Loss", fontsize=10)
        ax_l.set_xlabel("Step"); ax_l.set_ylabel("Loss")
        _annotate_trend(ax_l, steps_t, sm_l)
    else:
        ax_l.text(0.5, 0.5, "No loss entries in log\n(TRL GRPO may not emit scalar loss)",
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
