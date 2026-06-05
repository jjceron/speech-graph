"""Plotting utilities for Monte Carlo linear regression results.

Called from linear_regression_mc.py to produce:
  - R² distribution across MC iterations
  - RMSE distribution across MC iterations
  - Observed vs predicted scatter plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_r2_distribution(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    tag: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results_df["r2"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(results_df["r2"].mean(), color="crimson", linestyle="--", linewidth=2,
               label=f"Mean = {results_df['r2'].mean():.4f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("R²")
    ax.set_ylabel("Iterations")
    ax.set_title(f"R² distribution — {tag}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fpath = Path(output_dir) / f"r2_dist_{tag}.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath.name}")


def plot_rmse_distribution(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    tag: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results_df["rmse"], bins=40, color="seagreen", edgecolor="white", alpha=0.85)
    ax.axvline(results_df["rmse"].mean(), color="darkred", linestyle="--", linewidth=2,
               label=f"Mean = {results_df['rmse'].mean():.4f}")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Iterations")
    ax.set_title(f"RMSE distribution — {tag}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fpath = Path(output_dir) / f"rmse_dist_{tag}.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath.name}")


def plot_rho_distribution(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    tag: str,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results_df["rho"], bins=40, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(results_df["rho"].mean(), color="darkred", linestyle="--", linewidth=2,
               label=f"Mean = {results_df['rho'].mean():.4f}")
    ax.set_xlabel("Spearman rho (predicted vs observed)")
    ax.set_ylabel("Iterations")
    ax.set_title(f"Spearman rho distribution — {tag}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fpath = Path(output_dir) / f"rho_dist_{tag}.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath.name}")


def plot_observed_vs_predicted(
    y_true: list[float],
    y_pred: list[float],
    output_dir: str | Path,
    tag: str,
    dpi: int = 150,
) -> None:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    from scipy.stats import spearmanr, pearsonr
    r_pearson, _ = pearsonr(y_true, y_pred)
    r_spearman, _ = spearmanr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))

    try:
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(y_true[idx], y_pred[idx], c=z[idx], s=15, cmap="viridis", alpha=0.6)
    except Exception:
        ax.scatter(y_true, y_pred, s=15, alpha=0.5, color="steelblue")

    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Identity line")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted — {tag}", fontsize=13)
    ax.text(0.05, 0.95, f"Pearson r = {r_pearson:.4f}\nSpearman rho = {r_spearman:.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fpath = Path(output_dir) / f"obs_vs_pred_{tag}.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath.name}")


def plot_all_regression_figures(
    result: dict,
    output_dir: str | Path,
    tag: str,
) -> None:
    plot_r2_distribution(result["all_results"], output_dir, tag)
    plot_rmse_distribution(result["all_results"], output_dir, tag)
    plot_rho_distribution(result["all_results"], output_dir, tag)
    plot_observed_vs_predicted(result["y_true_all"], result["y_pred_all"], output_dir, tag)
