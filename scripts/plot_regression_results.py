"""Plot regression optuna results from saved CSV files.

Usage:
    py -m scripts.plot_regression_results --task 2 --window 10 --experiment raw --target MOT
    py -m scripts.plot_regression_results --task 2 --window 20 --experiment rawzscore --target COG_V1
    py -m scripts.plot_regression_results --task 2 --window 10 --experiment zscores --target all --val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path("outputs/regression_optuna")
ALL_TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]
BINS = 30
ALPHA = 0.65


# ── helpers ───────────────────────────────────────────────────────

def _load_csv(task: int, window: str, experiment: str, rfe: str, target: str, csv_name: str) -> pd.DataFrame | None:
    path = BASE_DIR / f"task{task}" / f"W{window}_{experiment}_{rfe}" / target / csv_name
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return None
    return pd.read_csv(path)


def _show_all() -> None:
    """Show all collected figures at once, then wait for user."""
    if plt.get_fignums():
        plt.show(block=False)
        plt.pause(0.3)


def _clean_rho(col: pd.Series) -> pd.Series:
    return col.dropna()[np.isfinite(col.dropna())]


# ── plot: side-by-side val vs test ────────────────────────────────

def plot_val_test_comparison(val_df: pd.DataFrame, test_df: pd.DataFrame, tag: str) -> None:
    metrics = [
        ("R²", "r2", "steelblue", "crimson"),
        ("MAE", "mae", "seagreen", "darkred"),
        ("ρ", "rho", "mediumpurple", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (label, col, c_val, c_test) in zip(axes, metrics):
        v = val_df[col].dropna().values
        t = test_df[col].dropna().values
        if col == "rho":
            v = v[np.isfinite(v)]
            t = t[np.isfinite(t)]
        if len(v) == 0 or len(t) == 0:
            ax.set_title(f"{label} — no data")
            continue

        lo = min(v.min(), t.min())
        hi = max(v.max(), t.max())
        bins = np.linspace(lo, hi, BINS + 1)

        ax.hist(v, bins=bins, color=c_val, alpha=ALPHA, label=f"Val mean={v.mean():.4f}", edgecolor="white")
        ax.hist(t, bins=bins, color=c_test, alpha=ALPHA, label=f"Test mean={t.mean():.4f}", edgecolor="white")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1) if label == "R²" else None
        ax.set_xlabel(label)
        ax.set_ylabel("Splits")
        ax.set_title(f"{label} — val vs test")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(tag, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


# ── plot: scatter val vs test (per split) ─────────────────────────

def plot_val_test_scatter(val_df: pd.DataFrame, test_df: pd.DataFrame, tag: str) -> None:
    r2_v = val_df["r2"].dropna().values
    r2_t = test_df["r2"].dropna().values
    min_len = min(len(r2_v), len(r2_t))
    if min_len < 10:
        print("  [SKIP] Scatter — too few splits")
        return
    r2_v, r2_t = r2_v[:min_len], r2_t[:min_len]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(r2_v, r2_t, s=8, alpha=0.4, color="steelblue")

    lim = min(r2_v.min(), r2_t.min()), max(r2_v.max(), r2_t.max())
    ax.plot(lim, lim, "r--", linewidth=1, alpha=0.6, label="Identity")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    ax.set_xlabel("R² validation")
    ax.set_ylabel("R² test")
    ax.set_title(f"R² val vs test — {tag}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()


# ── plot: test distributions (used when --val is OFF) ─────────────

def _hist_single(ax, values, color, xlabel, title, vline_zero=False):
    if len(values) == 0:
        ax.set_title(f"{title} — no data")
        return
    ax.hist(values, bins=BINS, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(values.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean={values.mean():.4f}")
    if vline_zero:
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Splits")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def plot_test_distributions(test_df: pd.DataFrame, tag: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    r2 = test_df["r2"].dropna().values
    _hist_single(axes[0], r2, "steelblue", "R²", "R² distribution", vline_zero=True)

    mae = test_df["mae"].dropna().values
    _hist_single(axes[1], mae, "seagreen", "MAE", "MAE distribution")

    rho = _clean_rho(test_df["rho"])
    _hist_single(axes[2], rho, "mediumpurple", "Spearman ρ", "ρ distribution")

    fig.suptitle(tag, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


# ── plot: observed vs predicted (fast, no KDE) ────────────────────

def plot_obs_vs_pred(pred_df: pd.DataFrame, tag: str) -> None:
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    if len(y_true) == 0:
        return

    r_s, _ = spearmanr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.2, color="steelblue", rasterized=True)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, alpha=0.6, label="Identity")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted — {tag}")
    ax.text(0.05, 0.95, f"Spearman ρ = {r_s:.4f}", transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()


# ── per-target runner ─────────────────────────────────────────────

def run_target(task: int, window: str, experiment: str, rfe: str, target: str, show_val: bool) -> None:
    print(f"\n{'='*56}")
    print(f"  T{task} | W{window} | {experiment} | {target}")
    print(f"{'='*56}")

    test_df = _load_csv(task, window, experiment, rfe, target, "final_test_iterations.csv")
    if test_df is None or len(test_df) == 0:
        print("  [SKIP] No test data")
        return

    tag = f"T{task}W{window}_{experiment}_{target}"
    print(f"  Test splits: {len(test_df)}")

    if show_val:
        val_df = _load_csv(task, window, experiment, rfe, target, "final_validation_iterations.csv")
        if val_df is not None and len(val_df) > 0:
            print(f"  Val splits:   {len(val_df)}")
            plot_val_test_comparison(val_df, test_df, tag)
            plot_val_test_scatter(val_df, test_df, tag)
    else:
        plot_test_distributions(test_df, tag)

    pred_df = _load_csv(task, window, experiment, rfe, target, "final_predictions.csv")
    if pred_df is not None and len(pred_df) > 0:
        plot_obs_vs_pred(pred_df, tag)

    _show_all()
    input("  Press Enter -> next target")


# ── entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot regression optuna results from CSV files.")
    parser.add_argument("--task", type=int, default=2, choices=[2, 6, 7])
    parser.add_argument("--window", required=True, help="Window number, e.g. 10, 20")
    parser.add_argument("--experiment", default="raw", choices=["raw", "zscores", "rawzscore"])
    parser.add_argument("--rfe", default="fixed", choices=["fixed", "global", "split-wise"])
    parser.add_argument("--target", default="all", help="Target name or 'all'")
    parser.add_argument("--val", action="store_true", help="Show val vs test comparison instead of test-only")
    args = parser.parse_args()

    targets = ALL_TARGETS if args.target == "all" else [args.target]

    for target in targets:
        run_target(args.task, args.window, args.experiment, args.rfe, target, args.val)

    print("\nDone.")


if __name__ == "__main__":
    main()
