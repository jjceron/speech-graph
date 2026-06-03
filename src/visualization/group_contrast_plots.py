"""Plotting functions for SpeechGraph high/low group contrasts."""
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


def _scheme_label(activity: object, window: object) -> str:
    try:
        a = int(activity)
    except Exception:
        a = activity
    try:
        w = int(window)
    except Exception:
        w = window
    return f"A{a}-W{w}"


def plot_top_adjusted_effects(results: pd.DataFrame, output_dir: str | Path, top_n: int = 30) -> Path | None:
    out_dir = _ensure_dir(output_dir)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return None
    ok["label"] = ok.apply(lambda r: f"A{int(r.activity_number)} W{int(r.window_size)} · {r.metric}", axis=1)
    top = ok.reindex(ok["adj_std_beta_high"].abs().sort_values(ascending=False).index).head(top_n).copy()
    top = top.iloc[::-1]
    height = max(6, 0.32 * len(top) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(top["label"], top["adj_std_beta_high"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Adjusted standardized high-minus-low effect")
    ax.set_title(f"Top {len(top)} adjusted SpeechGraph group contrasts")
    for i, (_, row) in enumerate(top.iterrows()):
        q = row.get("adj_q_fdr_global", np.nan)
        text = f"q={q:.3f}" if np.isfinite(q) else "q=NA"
        x = row["adj_std_beta_high"]
        ax.text(x + (0.02 if x >= 0 else -0.02), i, text, va="center", ha="left" if x >= 0 else "right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "top_adjusted_group_contrasts.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_effect_heatmap(results: pd.DataFrame, output_dir: str | Path) -> Path | None:
    out_dir = _ensure_dir(output_dir)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return None
    ok["scheme"] = ok.apply(lambda r: _scheme_label(r.activity_number, r.window_size), axis=1)
    pivot = ok.pivot_table(index="metric", columns="scheme", values="adj_std_beta_high", aggfunc="mean")
    # Sort columns by activity and window where possible.
    def parse_scheme(s: str):
        try:
            a, w = s.replace("A", "").replace("W", "").split("-")
            return int(a), int(w)
        except Exception:
            return (999, 999)
    pivot = pivot.reindex(sorted(pivot.columns, key=parse_scheme), axis=1)
    fig_w = max(12, 0.55 * len(pivot.columns))
    fig_h = max(7, 0.42 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Adjusted standardized high-minus-low effects by activity and window")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adjusted standardized effect")
    fig.tight_layout()
    path = out_dir / "heatmap_adjusted_effects_by_activity_window.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_q_heatmap(results: pd.DataFrame, output_dir: str | Path) -> Path | None:
    out_dir = _ensure_dir(output_dir)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty or "adj_q_fdr_global" not in ok.columns:
        return None
    ok["scheme"] = ok.apply(lambda r: _scheme_label(r.activity_number, r.window_size), axis=1)
    ok["minus_log10_q"] = -np.log10(np.clip(ok["adj_q_fdr_global"].astype(float), 1e-300, 1.0))
    pivot = ok.pivot_table(index="metric", columns="scheme", values="minus_log10_q", aggfunc="mean")
    def parse_scheme(s: str):
        try:
            a, w = s.replace("A", "").replace("W", "").split("-")
            return int(a), int(w)
        except Exception:
            return (999, 999)
    pivot = pivot.reindex(sorted(pivot.columns, key=parse_scheme), axis=1)
    fig_w = max(12, 0.55 * len(pivot.columns))
    fig_h = max(7, 0.42 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Global FDR evidence by activity and window")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(global q)")
    fig.tight_layout()
    path = out_dir / "heatmap_global_fdr_evidence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_volcano(results: pd.DataFrame, output_dir: str | Path) -> Path | None:
    out_dir = _ensure_dir(output_dir)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return None
    x = ok["adj_std_beta_high"].astype(float)
    q = np.clip(ok["adj_q_fdr_global"].astype(float), 1e-300, 1.0)
    y = -np.log10(q)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.8)
    ax.axvline(0, linewidth=1)
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    ax.set_xlabel("Adjusted standardized high-minus-low effect")
    ax.set_ylabel("-log10(global FDR q)")
    ax.set_title("SpeechGraph group contrast volcano plot")
    # Annotate top absolute effects and top evidence points.
    annot = ok.assign(score=np.maximum(np.abs(x), y / max(1.0, y.max()))).sort_values("score", ascending=False).head(12)
    for _, row in annot.iterrows():
        ax.text(row["adj_std_beta_high"], -np.log10(max(row["adj_q_fdr_global"], 1e-300)),
                f"A{int(row.activity_number)}W{int(row.window_size)} {row.metric}", fontsize=7)
    fig.tight_layout()
    path = out_dir / "volcano_adjusted_group_contrasts.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_marker_counts(summary: pd.DataFrame, output_dir: str | Path) -> Path | None:
    out_dir = _ensure_dir(output_dir)
    if summary.empty:
        return None
    s = summary.copy()
    s["scheme"] = s.apply(lambda r: _scheme_label(r.activity_number, r.window_size), axis=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(s["scheme"], s["n_global_fdr_10"])
    ax.set_ylabel("Number of markers with global q < 0.10")
    ax.set_title("Candidate SpeechGraph markers by activity and window")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out_dir / "candidate_marker_counts_by_scheme.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_marker_boxplots(analysis_df: pd.DataFrame, results: pd.DataFrame, output_dir: str | Path, label_col: str, top_n: int = 8) -> list[Path]:
    out_dir = _ensure_dir(output_dir)
    ok = results[results["status"].eq("ok")].copy()
    if ok.empty:
        return []
    top = ok.reindex(ok["adj_std_beta_high"].abs().sort_values(ascending=False).index).head(top_n)
    paths: list[Path] = []
    for _, row in top.iterrows():
        activity = row["activity_number"]
        window = row["window_size"]
        metric = row["metric"]
        sub = analysis_df[(analysis_df["activity_number"] == activity) & (analysis_df["window_size"] == window)].copy()
        if metric not in sub.columns:
            continue
        groups = []
        labels = []
        for lab, g in sub.groupby(label_col):
            values = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                groups.append(values)
                labels.append(str(lab))
        if len(groups) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_title(f"A{int(activity)} W{int(window)} · {metric}\nadjusted β={row['adj_std_beta_high']:.2f}, q={row['adj_q_fdr_global']:.3f}")
        ax.set_ylabel(metric)
        fig.tight_layout()
        safe = f"boxplot_A{int(activity)}_W{int(window)}_{metric}.png".replace("/", "_")
        path = out_dir / safe
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def make_all_group_contrast_plots(
    analysis_df: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    label_col: str,
    top_n: int = 30,
) -> list[str]:
    out_dir = _ensure_dir(output_dir)
    paths: list[Path] = []
    for func in [
        lambda: plot_top_adjusted_effects(results, out_dir, top_n=top_n),
        lambda: plot_effect_heatmap(results, out_dir),
        lambda: plot_q_heatmap(results, out_dir),
        lambda: plot_volcano(results, out_dir),
        lambda: plot_marker_counts(summary, out_dir),
    ]:
        try:
            p = func()
            if p is not None:
                paths.append(p)
        except Exception as exc:
            print(f"Warning: plot failed: {exc}")
    try:
        paths.extend(plot_top_marker_boxplots(analysis_df, results, out_dir / "top_marker_boxplots", label_col=label_col, top_n=8))
    except Exception as exc:
        print(f"Warning: top marker boxplots failed: {exc}")
    return [str(p) for p in paths]
