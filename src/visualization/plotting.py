from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_r2(results: pd.DataFrame, figures_dir: Path) -> int:
    if results.empty or "r2" not in results.columns:
        return 0
    count = 0
    data = results.copy()
    data["r2"] = pd.to_numeric(data["r2"], errors="coerce")
    if {"activity_number", "window_size"}.issubset(data.columns):
        for target, sub in data.groupby("target", dropna=False):
            table = sub.pivot_table(index="activity_number", columns="window_size", values="r2", aggfunc="mean")
            if table.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(table.to_numpy(dtype=float), aspect="auto")
            ax.set_xticks(range(len(table.columns)))
            ax.set_xticklabels([str(int(c)) for c in table.columns])
            ax.set_yticks(range(len(table.index)))
            ax.set_yticklabels([f"A{int(i)}" for i in table.index])
            ax.set_xlabel("Window size")
            ax.set_ylabel("Activity")
            ax.set_title(f"Cross-validated R²: {target}")
            fig.colorbar(im, ax=ax, label="R²")
            _save(fig, figures_dir / "models" / f"r2_heatmap_{_safe(target)}.png")
            count += 1
        summary = data.groupby("target", dropna=False)["r2"].max().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary["target"].astype(str), summary["r2"])
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Best cross-validated R²")
        ax.set_xlabel("Barratt dimension")
        ax.set_title("Best activity-window model performance")
        _save(fig, figures_dir / "models" / "r2_best_by_target.png")
        return count + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["target"].astype(str), data["r2"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Cross-validated R²")
    ax.set_xlabel("Barratt dimension")
    ax.set_title("Linear model performance")
    _save(fig, figures_dir / "models" / "r2_by_target.png")
    return 1


def plot_predictions(predictions: pd.DataFrame, figures_dir: Path) -> int:
    if predictions.empty:
        return 0
    count = 0
    for target, sub in predictions.groupby("target", dropna=False):
        y_true = pd.to_numeric(sub["y_true"], errors="coerce")
        y_pred = pd.to_numeric(sub["y_pred"], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        if mask.sum() < 10:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true[mask], y_pred[mask], s=18, alpha=0.55)
        lo = min(float(y_true[mask].min()), float(y_pred[mask].min()))
        hi = max(float(y_true[mask].max()), float(y_pred[mask].max()))
        ax.plot([lo, hi], [lo, hi], linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Observed vs predicted: {target}")
        _save(fig, figures_dir / "models" / f"observed_vs_predicted_{_safe(target)}.png")
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
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(top))))
        ax.barh(labels.astype(str), pd.to_numeric(top["mean_abs_contribution"], errors="coerce"))
        ax.set_xlabel("Mean absolute contribution")
        ax.set_title(f"Feature relevance: {target}")
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
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(top))))
        ax.barh(labels.astype(str), pd.to_numeric(top["mean_abs_contribution"], errors="coerce"))
        ax.set_xlabel("Mean absolute contribution")
        ax.set_title(f"{target} | {group_col}: {group}")
        _save(fig, figures_dir / "interpretability" / f"feature_relevance_{output_name}_{_safe(target)}_{_safe(group)}.png")
        count += 1
    return count


def plot_correlations(corr: pd.DataFrame, figures_dir: Path, top_n: int = 25) -> int:
    if corr.empty or "r" not in corr.columns:
        return 0
    count = 0
    for target, sub in corr.groupby("target", dropna=False):
        top = sub.sort_values("abs_r", ascending=False).head(top_n).iloc[::-1]
        if top.empty:
            continue
        labels = top.apply(lambda r: f"w{r.get('scheme_window_size','')}_a{r.get('activity_number','')}_{r.get('metric','')}", axis=1)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
        ax.barh(labels.astype(str), pd.to_numeric(top["r"], errors="coerce"))
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Correlation r")
        ax.set_title(f"Top graph-Barratt correlations: {target}")
        _save(fig, figures_dir / "analysis" / f"top_correlations_{_safe(target)}.png")
        count += 1
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
