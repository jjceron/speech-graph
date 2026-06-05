"""Plotting utilities for correlation analysis heatmaps.

Called from correlation_analysis.py to produce heatmaps of simple and
partial Spearman correlations (raw and z-score features) with annotated values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pivot_correlation_matrix(
    results: pd.DataFrame,
    ftype: str,
) -> pd.DataFrame:
    sub = results[results["type"] == ftype].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["_task_window"] = sub["task"].astype(str) + "_" + sub["window_tag"]

    sub = sub.pivot_table(
        index="feature",
        columns="_task_window",
        values=["r_MOT", "r_partial_MOT", "r_COG", "r_partial_COG"],
        aggfunc="first",
    )
    sub.columns = [f"{col[0]}_{col[1]}" for col in sub.columns]
    sub = sub.reindex(sorted(sub.columns), axis=1)
    return sub


def _build_heatmap_data(
    results: pd.DataFrame,
    ftype: str,
    corr_type: str,
) -> pd.DataFrame:
    sub = results[results["type"] == ftype].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["_task_window"] = sub["task"].astype(str) + "_" + sub["window_tag"]

    col_map = {
        "simple": ("r_MOT", "r_COG"),
        "partial": ("r_partial_MOT", "r_partial_COG"),
    }
    r_mot_col, r_cog_col = col_map[corr_type]

    pivot = sub.pivot_table(
        index="feature",
        columns="_task_window",
        values=[r_mot_col, r_cog_col],
        aggfunc="first",
    )
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    task_windows = sorted(sub["_task_window"].unique())
    mot_cols = [f"{r_mot_col}_{tw}" for tw in task_windows if f"{r_mot_col}_{tw}" in pivot.columns]
    cog_cols = [f"{r_cog_col}_{tw}" for tw in task_windows if f"{r_cog_col}_{tw}" in pivot.columns]

    out = pd.DataFrame(index=pivot.index)
    for tw in task_windows:
        mot_c = f"{r_mot_col}_{tw}"
        cog_c = f"{r_cog_col}_{tw}"
        if mot_c in pivot.columns:
            out[f"MOT_{tw}"] = pivot[mot_c]
        if cog_c in pivot.columns:
            out[f"COG_{tw}"] = pivot[cog_c]

    return out


def plot_correlation_heatmaps(
    results: pd.DataFrame,
    output_dir: str | Path,
    adj_var: str = "School year",
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 150,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for ftype in ("raw", "z"):
        label = "Raw metrics" if ftype == "raw" else "Z-score metrics"

        for corr_type, corr_label in [("simple", "Simple"), ("partial", "Partial")]:
            matrix = _build_heatmap_data(results, ftype, corr_type)
            if matrix.empty:
                continue

            vmax = max(matrix.abs().max().max(), 1e-8)

            title = f"Simple Spearman correlations — {label}" if corr_type == "simple" else f"Partial Spearman correlations (controlling {adj_var}) — {label}"

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                matrix, annot=True, fmt=".3f", cmap=cmap,
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Spearman rho"},
            )
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Target / Task-Window")
            ax.set_ylabel("Feature")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            fname = f"heatmap_{corr_type}_{ftype}.png"
            fpath = output_dir / fname
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fpath.name}")

    print("  Correlation heatmaps done.")


def plot_combined_heatmaps(
    results: pd.DataFrame,
    output_dir: str | Path,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 150,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for ftype in ("raw", "z"):
        label = "Raw metrics" if ftype == "raw" else "Z-score metrics"
        simple_m = _build_heatmap_data(results, ftype, "simple")
        partial_m = _build_heatmap_data(results, ftype, "partial")

        if simple_m.empty and partial_m.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        for ax, matrix, corr_label in zip(
            axes, [simple_m, partial_m],
            ["Simple Spearman rho", "Partial Spearman rho (controlling School year)"]
        ):
            if matrix.empty:
                ax.set_visible(False)
                continue
            vmax = max(matrix.abs().max().max(), 1e-8)
            sns.heatmap(
                matrix, annot=True, fmt=".3f", cmap=cmap,
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "rho"},
            )
            ax.set_title(corr_label, fontsize=11)
            ax.set_xlabel("Target / Task-Window")
            ax.set_ylabel("Feature")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig.suptitle(f"Correlation analysis — {label}", fontsize=15, y=1.02)
        plt.tight_layout()

        fname = f"heatmap_combined_{ftype}.png"
        fpath = output_dir / fname
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fpath.name}")

    print("  Combined heatmaps done.")
