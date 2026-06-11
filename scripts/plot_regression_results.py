"""Plot regression optuna results from saved CSV files.

Usage:
    py -m scripts.plot_regression_results --task 2 --window 10 --experiment raw --target MOT
    py -m scripts.plot_regression_results --task 2 --window 20 --experiment rawzscore --target all
    py -m scripts.plot_regression_results --task 2 --window 10 --experiment zscores --target COG_V1 --val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

BASE_DIR = Path("outputs/regression_optuna")
ALL_TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]


def _load_csv(task: int, window: str, experiment: str, rfe: str, target: str, csv_name: str) -> pd.DataFrame | None:
    path = BASE_DIR / f"task{task}" / f"W{window}_{experiment}_{rfe}" / target / csv_name
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        return None
    return pd.read_csv(path)


def _make_tag(task: int, window: str, experiment: str, target: str, set_name: str) -> str:
    return f"T{task}W{window}_{experiment}_{target}_{set_name}"


# ── plot functions ────────────────────────────────────────────────

def plot_r2(df: pd.DataFrame, tag: str) -> None:
    r2 = df["r2"].dropna().values
    if len(r2) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r2, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(r2.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean = {r2.mean():.4f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("R²")
    ax.set_ylabel("Iterations")
    ax.set_title(f"R² distribution — {tag}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.show()


def plot_mae(df: pd.DataFrame, tag: str) -> None:
    mae = df["mae"].dropna().values
    if len(mae) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mae, bins=40, color="seagreen", edgecolor="white", alpha=0.85)
    ax.axvline(mae.mean(), color="darkred", linestyle="--", linewidth=2, label=f"Mean = {mae.mean():.4f}")
    ax.set_xlabel("MAE")
    ax.set_ylabel("Iterations")
    ax.set_title(f"MAE distribution — {tag}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.show()


def plot_rho(df: pd.DataFrame, tag: str) -> None:
    rho = df["rho"].dropna().values
    rho = rho[np.isfinite(rho)]
    if len(rho) == 0:
        print(f"  [SKIP] ρ plot — no valid ρ values for {tag}")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rho, bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(rho.mean(), color="darkred", linestyle="--", linewidth=2, label=f"Mean = {rho.mean():.4f}")
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Iterations")
    ax.set_title(f"Spearman ρ distribution — {tag}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.show()


def plot_obs_vs_pred(pred_df: pd.DataFrame, tag: str) -> None:
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    if len(y_true) == 0:
        return

    r_s, _ = spearmanr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(y_true[idx], y_pred[idx], c=z[idx], s=15, cmap="viridis", alpha=0.6)
    except Exception:
        ax.scatter(y_true, y_pred, s=15, alpha=0.5, color="steelblue")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Identity")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted — {tag}")
    ax.text(0.05, 0.95, f"Spearman ρ = {r_s:.4f}", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.show()


# ── main ──────────────────────────────────────────────────────────

def run_target(task: int, window: str, experiment: str, rfe: str, target: str, show_val: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  Target: {target}")
    print(f"  Config: task{task} / W{window} / {experiment} / {rfe}")
    print(f"{'='*60}")

    # Test
    test_df = _load_csv(task, window, experiment, rfe, target, "final_test_iterations.csv")
    if test_df is not None and len(test_df) > 0:
        tag = _make_tag(task, window, experiment, target, "test")
        print(f"  Test splits: {len(test_df)}")
        plot_r2(test_df, tag)
        plot_mae(test_df, tag)
        plot_rho(test_df, tag)
    else:
        print(f"  [SKIP] No test data for {target}")

    # Validation (optional)
    if show_val:
        val_df = _load_csv(task, window, experiment, rfe, target, "final_validation_iterations.csv")
        if val_df is not None and len(val_df) > 0:
            tag = _make_tag(task, window, experiment, target, "val")
            print(f"  Validation splits: {len(val_df)}")
            plot_r2(val_df, tag)
            plot_mae(val_df, tag)
            plot_rho(val_df, tag)

    # Observed vs Predicted (from predictions CSV)
    pred_df = _load_csv(task, window, experiment, rfe, target, "final_predictions.csv")
    if pred_df is not None and len(pred_df) > 0:
        tag = _make_tag(task, window, experiment, target, "pred")
        print(f"  Predictions: {len(pred_df)} rows")
        plot_obs_vs_pred(pred_df, tag)
    else:
        print(f"  [SKIP] No predictions for {target}")

    input("  Press Enter to continue to next target...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot regression optuna results from CSV files.")
    parser.add_argument("--task", type=int, default=2, choices=[2, 6, 7])
    parser.add_argument("--window", required=True, help="Window number, e.g. 10, 20")
    parser.add_argument("--experiment", default="raw", choices=["raw", "zscores", "rawzscore"])
    parser.add_argument("--rfe", default="fixed", choices=["fixed", "global", "split-wise"])
    parser.add_argument("--target", default="all", help="Target name or 'all'")
    parser.add_argument("--val", action="store_true", help="Also show validation distribution plots")
    args = parser.parse_args()

    targets = ALL_TARGETS if args.target == "all" else [args.target]

    for target in targets:
        run_target(args.task, args.window, args.experiment, args.rfe, target, args.val)

    print("\nDone.")


if __name__ == "__main__":
    main()
