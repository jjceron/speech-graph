from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required. Install with: py -m pip install matplotlib") from exc

from .common import safe_name


def save_heatmap(
    data: pd.DataFrame,
    path: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    vmin: float | None = -1,
    vmax: float | None = 1,
    annotate: bool = True,
    figsize_scale: tuple[float, float] = (0.55, 0.38),
) -> Path | None:
    if data.empty:
        return None
    matrix = data.to_numpy(dtype=float)
    fig_w = max(8, figsize_scale[0] * max(1, len(data.columns)) + 3)
    fig_h = max(5, figsize_scale[1] * max(1, len(data.index)) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotate and len(data.index) <= 20 and len(data.columns) <= 18:
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                value = matrix[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="r")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_horizontal_bars(
    labels: list[str],
    values: Iterable[float],
    path: Path,
    title: str,
    xlabel: str = "",
    xline_zero: bool = True,
    max_height: float = 12,
) -> Path | None:
    values = list(values)
    if not labels or not values:
        return None
    fig_h = min(max_height, max(4.5, 0.35 * len(labels) + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = list(range(len(labels)))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    if xline_zero:
        ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
    group_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Path | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 4:
        return None
    plot_df = df.loc[mask].copy()
    plot_df[x_col] = x[mask]
    plot_df[y_col] = y[mask]
    fig, ax = plt.subplots(figsize=(7, 5))
    if group_col and group_col in plot_df.columns and plot_df[group_col].nunique(dropna=True) > 1:
        for label, sub in plot_df.groupby(group_col, dropna=False):
            ax.scatter(sub[x_col], sub[y_col], alpha=0.75, label=str(label))
        ax.legend(fontsize=8)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.75)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_boxplot_by_group(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    path: Path,
    title: str,
) -> Path | None:
    if metric not in df.columns or group_col not in df.columns:
        return None
    values = pd.to_numeric(df[metric], errors="coerce")
    tmp = df[[group_col]].copy()
    tmp[metric] = values
    tmp = tmp.dropna(subset=[metric, group_col])
    groups = [(str(label), sub[metric].to_numpy()) for label, sub in tmp.groupby(group_col, dropna=False)]
    groups = [(label, arr) for label, arr in groups if len(arr) >= 3]
    if len(groups) < 2:
        return None
    labels, arrays = zip(*groups)
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(labels)), 5))
    ax.boxplot(arrays, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_line_by_window_group(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    path: Path,
    title: str,
) -> Path | None:
    if "window_size" not in df.columns or metric not in df.columns or group_col not in df.columns:
        return None
    tmp = df[["window_size", group_col]].copy()
    tmp[metric] = pd.to_numeric(df[metric], errors="coerce")
    tmp["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
    tmp = tmp.dropna(subset=["window_size", metric, group_col])
    if tmp.empty or tmp["window_size"].nunique() < 2 or tmp[group_col].nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for label, sub in tmp.groupby(group_col, dropna=False):
        agg = sub.groupby("window_size")[metric].mean().reset_index().sort_values("window_size")
        if len(agg) < 2:
            continue
        ax.plot(agg["window_size"], agg[metric], marker="o", label=str(label))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(title)
    ax.set_xlabel("window_size")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def numbered_path(out_dir: Path, prefix: str, index: int, name: str) -> Path:
    return out_dir / f"{prefix}_{index:02d}_{safe_name(name)}.png"
