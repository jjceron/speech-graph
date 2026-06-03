"""Visualization helpers for 05_run composite SpeechGraph indices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_index_effects(results: pd.DataFrame, out_dir: str | Path, primary_model: str) -> Path:
    out_dir = _ensure_dir(out_dir)
    df = results[(results["model"] == primary_model) & (results["family"] == "primary")].copy()
    df = df.sort_values("adj_beta_high")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.7 * len(df) + 2)))
    beta = df["adj_beta_high"].to_numpy(float)
    lo = df["bootstrap_ci_low"].to_numpy(float)
    hi = df["bootstrap_ci_high"].to_numpy(float)
    xerr = np.vstack([beta - lo, hi - beta])
    ax.errorbar(beta, y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    labels = []
    for _, r in df.iterrows():
        q = r.get("perm_q_primary_family", np.nan)
        p = r.get("perm_p_one_sided_greater", np.nan)
        labels.append(f"{r['index']}\nperm p={p:.3g}, q={q:.3g}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Adjusted high-minus-low beta (index units)")
    ax.set_title("Composite SpeechGraph index effects")
    fig.tight_layout()
    path = out_dir / "composite_index_effects_forest.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_index_boxplots(index_df: pd.DataFrame, results: pd.DataFrame, label_col: str, out_dir: str | Path, primary_model: str) -> list[Path]:
    out_dir = _ensure_dir(Path(out_dir) / "index_boxplots")
    paths: list[Path] = []
    df_res = results[(results["model"] == primary_model) & (results["family"] == "primary")]
    for _, r in df_res.iterrows():
        idx = r["index"]
        d = index_df[[label_col, idx]].dropna()
        groups = [g[idx].to_numpy(float) for _, g in d.groupby(label_col, sort=True)]
        labels = [str(k) for k, _ in d.groupby(label_col, sort=True)]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_title(f"{idx}\nadj beta={r['adj_beta_high']:.3f}, perm q={r.get('perm_q_primary_family', np.nan):.3g}")
        ax.set_ylabel("Composite index value")
        fig.tight_layout()
        path = out_dir / f"boxplot_{idx}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_index_component_heatmap(index_df: pd.DataFrame, component_table: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    primary = component_table[component_table["family"] == "primary"].copy()
    rows = []
    for _, r in primary.iterrows():
        idx = r["index"]
        pos = [x for x in str(r["positive_features"]).split(",") if x]
        neg = [x for x in str(r["negative_features"]).split(",") if x]
        for f in pos:
            rows.append({"index": idx, "feature": f, "weight": 1})
        for f in neg:
            rows.append({"index": idx, "feature": f, "weight": -1})
    mat_df = pd.DataFrame(rows)
    if mat_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No components", ha="center", va="center")
        ax.axis("off")
    else:
        mat = mat_df.pivot_table(index="feature", columns="index", values="weight", fill_value=0, aggfunc="sum")
        fig, ax = plt.subplots(figsize=(8, max(5, len(mat) * 0.25)))
        im = ax.imshow(mat.to_numpy(float), aspect="auto", vmin=-1, vmax=1)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right")
        ax.set_title("Composite index component weights")
        fig.colorbar(im, ax=ax, label="Feature weight")
    fig.tight_layout()
    path = out_dir / "composite_index_component_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sensitivity(results: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = _ensure_dir(out_dir)
    df = results[(results["family"] == "primary")].copy()
    pivot = df.pivot_table(index="index", columns="model", values="adj_beta_high")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(pivot) + 2)))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    ax.set_title("Sensitivity of adjusted beta across covariate models")
    fig.colorbar(im, ax=ax, label="Adjusted high-minus-low beta")
    fig.tight_layout()
    path = out_dir / "composite_index_sensitivity_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
