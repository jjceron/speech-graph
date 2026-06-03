from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _safe(value: object) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_heatmap(ax, values: np.ndarray, fmt: str = ".4f") -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    midpoint = (float(finite.min()) + float(finite.max())) / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value > midpoint else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=color)


def _annotate_bars(ax, fmt: str = ".4f") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if not np.isfinite(width):
            continue
        y = patch.get_y() + patch.get_height() / 2
        ha = "left" if width >= 0 else "right"
        dx = 0.002 if width >= 0 else -0.002
        ax.text(width + dx, y, format(width, fmt), va="center", ha=ha, fontsize=8)


def _heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, colorbar_label: str, path: Path) -> int:
    if table.empty:
        return 0
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return 0
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(table.columns)), max(4, 0.45 * len(table.index))))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(i) for i in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    _annotate_heatmap(ax, values, fmt=".4f")
    _save(fig, path)
    return 1


def plot_r2(results: pd.DataFrame, figures_dir: Path) -> int:
    if results.empty or "r2" not in results.columns:
        return 0
    data = results.copy()
    data["r2"] = pd.to_numeric(data["r2"], errors="coerce")
    count = 0
    if {"activity_number", "window_size"}.issubset(data.columns):
        for target, sub in data.groupby("target", dropna=False):
            table = sub.pivot_table(index="activity_number", columns="window_size", values="r2", aggfunc="mean")
            table = table.sort_index().sort_index(axis=1)
            table.index = [f"A{int(i)}" for i in table.index]
            table.columns = [f"W{int(c)}" for c in table.columns]
            count += _heatmap(
                table,
                title=f"Cross-validated R²: {target}",
                xlabel="Window size",
                ylabel="Activity",
                colorbar_label="R²",
                path=figures_dir / "models" / f"r2_heatmap_{_safe(target)}.png",
            )
        summary = data.groupby("target", dropna=False)["r2"].max().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["target"].astype(str), summary["r2"])
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Best cross-validated R²")
        ax.set_xlabel("Barratt dimension")
        ax.set_title("Best activity-window model performance")
        for patch in ax.patches:
            value = patch.get_height()
            if np.isfinite(value):
                va = "bottom" if value >= 0 else "top"
                ax.text(patch.get_x() + patch.get_width() / 2, value, format(value, ".4f"), ha="center", va=va, fontsize=8)
        _save(fig, figures_dir / "models" / "r2_best_by_target.png")
        return count + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["target"].astype(str), data["r2"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Cross-validated R²")
    ax.set_xlabel("Barratt dimension")
    ax.set_title("Linear model performance")
    for patch in ax.patches:
        value = patch.get_height()
        if np.isfinite(value):
            va = "bottom" if value >= 0 else "top"
            ax.text(patch.get_x() + patch.get_width() / 2, value, format(value, ".4f"), ha="center", va=va, fontsize=8)
    _save(fig, figures_dir / "models" / "r2_by_target.png")
    return 1


def plot_predictions(predictions: pd.DataFrame, figures_dir: Path) -> int:
    if predictions.empty:
        return 0
    count = 0
    group_cols = [col for col in ["target", "activity_number", "window_size"] if col in predictions.columns]
    grouped = predictions.groupby(group_cols, dropna=False) if group_cols else [((), predictions)]
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        info = dict(zip(group_cols, keys))
        y_true = pd.to_numeric(sub["y_true"], errors="coerce")
        y_pred = pd.to_numeric(sub["y_pred"], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        if mask.sum() < 30:
            continue
        target = info.get("target", "target")
        suffix = _safe(target)
        title = f"Observed vs predicted: {target}"
        if "activity_number" in info and "window_size" in info:
            suffix += f"_A{int(info['activity_number'])}_W{int(info['window_size'])}"
            title += f" | A{int(info['activity_number'])} W{int(info['window_size'])}"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true[mask], y_pred[mask], s=18, alpha=0.55)
        lo = min(float(y_true[mask].min()), float(y_pred[mask].min()))
        hi = max(float(y_true[mask].max()), float(y_pred[mask].max()))
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        _save(fig, figures_dir / "models" / f"observed_vs_predicted_{suffix}.png")
        count += 1
    return count


def _feature_label(row: pd.Series) -> str:
    prefix = ""
    if "activity_number" in row and "window_size" in row and pd.notna(row.get("activity_number")) and pd.notna(row.get("window_size")):
        prefix = f"A{int(row['activity_number'])}-W{int(row['window_size'])}: "
    return prefix + str(row.get("feature", ""))


def plot_population_relevance(relevance: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> int:
    if relevance.empty or "mean_abs_contribution" not in relevance.columns:
        return 0
    count = 0
    for target, sub in relevance.groupby("target", dropna=False):
        top = sub.sort_values("mean_abs_contribution", ascending=False).head(top_n).iloc[::-1]
        if top.empty:
            continue
        labels = top.apply(_feature_label, axis=1)
        values = pd.to_numeric(top["mean_abs_contribution"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(top))))
        ax.barh(labels.astype(str), values)
        ax.set_xlabel("Mean absolute contribution")
        ax.set_title(f"Feature relevance: {target}")
        _annotate_bars(ax, fmt=".4f")
        _save(fig, figures_dir / "interpretability" / f"feature_relevance_population_{_safe(target)}.png")
        count += 1
    return count


def plot_group_relevance(relevance: pd.DataFrame, group_col: str, output_name: str, figures_dir: Path, top_n: int = 15) -> int:
    if relevance.empty or group_col not in relevance.columns or "mean_abs_contribution" not in relevance.columns:
        return 0
    count = 0
    for (target, group), sub in relevance.groupby(["target", group_col], dropna=False):
        top = sub.sort_values("mean_abs_contribution", ascending=False).head(top_n).iloc[::-1]
        if top.empty:
            continue
        labels = top.apply(_feature_label, axis=1)
        values = pd.to_numeric(top["mean_abs_contribution"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(top))))
        ax.barh(labels.astype(str), values)
        ax.set_xlabel("Mean absolute contribution")
        ax.set_title(f"{target} | {group_col}: {group}")
        _annotate_bars(ax, fmt=".4f")
        _save(fig, figures_dir / "interpretability" / f"feature_relevance_{output_name}_{_safe(target)}_{_safe(group)}.png")
        count += 1
    return count


def plot_correlations(corr: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> int:
    if corr.empty or "r" not in corr.columns:
        return 0
    data = corr.copy()
    data["r"] = pd.to_numeric(data["r"], errors="coerce")
    data["abs_r"] = pd.to_numeric(data.get("abs_r", data["r"].abs()), errors="coerce")
    count = 0
    for target, sub in data.groupby("target", dropna=False):
        top = sub.sort_values("abs_r", ascending=False).head(top_n).iloc[::-1]
        if top.empty:
            continue
        labels = top.apply(lambda r: f"W{r.get('scheme_window_size','')}-A{r.get('activity_number','')}-{r.get('metric','')}", axis=1)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
        ax.barh(labels.astype(str), pd.to_numeric(top["r"], errors="coerce"))
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Correlation r")
        ax.set_title(f"Top graph-Barratt correlations: {target}")
        _annotate_bars(ax, fmt=".4f")
        _save(fig, figures_dir / "analysis" / f"top_correlations_{_safe(target)}.png")
        count += 1

    needed = {"target", "activity_number", "scheme_window_size", "metric", "r"}
    if needed.issubset(data.columns):
        for (target, activity), sub in data.groupby(["target", "activity_number"], dropna=False):
            table = sub.pivot_table(index="metric", columns="scheme_window_size", values="r", aggfunc="mean")
            table = table.reindex(sorted(table.index)).sort_index(axis=1)
            table.columns = [f"W{int(c)}" for c in table.columns]
            count += _heatmap(
                table,
                title=f"Correlation r: {target} | Activity {int(activity)}",
                xlabel="Window size",
                ylabel="Feature",
                colorbar_label="r",
                path=figures_dir / "analysis" / f"correlation_heatmap_{_safe(target)}_A{int(activity)}.png",
            )
    return count


def generate_figures(run_dir: Path) -> int:
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    interp_dir = run_dir / "interpretability"
    analysis_dir = run_dir / "analysis"
    count = 0
    count += plot_r2(_read(models_dir / "linear_cv_results.csv"), figures_dir)
    count += plot_predictions(_read(models_dir / "linear_cv_predictions.csv"), figures_dir)
    count += plot_population_relevance(_read(interp_dir / "feature_relevance_population.csv"), figures_dir)
    count += plot_group_relevance(_read(interp_dir / "feature_relevance_by_age.csv"), "age_group", "by_age", figures_dir)
    schooling = _read(interp_dir / "feature_relevance_by_schooling.csv")
    group_cols = [col for col in schooling.columns if col not in {"target", "activity_number", "activity", "window_size", "scheme_window_size", "feature", "mean_contribution", "mean_abs_contribution", "n"}]
    count += plot_group_relevance(schooling, group_cols[0], "by_schooling", figures_dir) if group_cols else 0
    corr = _read(analysis_dir / "correlations_by_activity_window_min_n100.csv")
    if corr.empty:
        corr = _read(analysis_dir / "correlations_by_activity_window.csv")
    count += plot_correlations(corr, figures_dir)
    print(f"Generated {count} figures in {figures_dir}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for one run directory")
    parser.add_argument("--run-dir", default="outputs/01_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figures(Path(args.run_dir))


if __name__ == "__main__":
    main()
