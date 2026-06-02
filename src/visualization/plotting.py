from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization. Install requirements.txt first.") from exc


def _safe_name(value: object) -> str:
    text = str(value or "").strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "plot"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path, xlabel: str) -> bool:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return False
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col]).tail(20)
    if plot_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return True


def plot_model_results(run_dir: Path, figures_dir: Path) -> list[Path]:
    paths: list[Path] = []
    results = _read(run_dir / "models" / "linear_cv_results.csv")
    predictions = _read(run_dir / "models" / "linear_cv_predictions.csv")
    if not results.empty and {"target", "r2"}.issubset(results.columns):
        plot_df = results.copy()
        plot_df["r2"] = pd.to_numeric(plot_df["r2"], errors="coerce")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(plot_df["target"].astype(str), plot_df["r2"])
        ax.axhline(0, linewidth=1)
        ax.set_title("Cross-validated R² by Barratt dimension")
        ax.set_xlabel("Target")
        ax.set_ylabel("R²")
        fig.tight_layout()
        path = figures_dir / "models" / "r2_by_target.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths.append(path)

    if not predictions.empty and {"target", "y_true", "y_pred"}.issubset(predictions.columns):
        for target, sub in predictions.groupby("target"):
            x = pd.to_numeric(sub["y_true"], errors="coerce")
            y = pd.to_numeric(sub["y_pred"], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x[mask], y[mask], alpha=0.75)
            mn = float(min(x[mask].min(), y[mask].min()))
            mx = float(max(x[mask].max(), y[mask].max()))
            ax.plot([mn, mx], [mn, mx], linewidth=1)
            ax.set_title(f"Predicted vs observed: {target}")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            fig.tight_layout()
            path = figures_dir / "models" / f"predicted_vs_observed_{_safe_name(target)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_correlations(run_dir: Path, figures_dir: Path, max_rows: int = 25) -> list[Path]:
    paths: list[Path] = []
    corr = _read(run_dir / "analysis" / "correlations_by_activity_window.csv")
    if corr.empty or not {"metric", "target", "r"}.issubset(corr.columns):
        return paths
    corr = corr.copy()
    corr["r"] = pd.to_numeric(corr["r"], errors="coerce")
    corr = corr.dropna(subset=["r"])
    corr["label"] = corr.apply(
        lambda r: f"w{r.get('scheme_window_size','')}_a{r.get('activity_number','')} {r['metric']} → {r['target']}",
        axis=1,
    )
    corr = corr.sort_values("r", key=lambda s: s.abs(), ascending=False).head(max_rows).sort_values("r")
    if corr.empty:
        return paths
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(corr))))
    ax.barh(corr["label"], corr["r"])
    ax.axvline(0, linewidth=1)
    ax.set_title("Top speech-graph correlations by activity and window")
    ax.set_xlabel("Correlation")
    fig.tight_layout()
    path = figures_dir / "analysis" / "top_correlations_by_activity_window.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_interpretability(run_dir: Path, figures_dir: Path, top_n: int = 20) -> list[Path]:
    paths: list[Path] = []
    population = _read(run_dir / "interpretability" / "contributions_population.csv")
    by_age = _read(run_dir / "interpretability" / "contributions_by_age.csv")
    by_school = _read(run_dir / "interpretability" / "contributions_by_schooling.csv")

    if not population.empty and {"target", "feature", "mean_abs_contribution"}.issubset(population.columns):
        for target, sub in population.groupby("target"):
            top = sub.sort_values("mean_abs_contribution", ascending=False).head(top_n).sort_values("mean_abs_contribution")
            path = figures_dir / "interpretability" / f"population_top_features_{_safe_name(target)}.png"
            if _barh(top, "feature", "mean_abs_contribution", f"Population feature relevance: {target}", path, "Mean absolute contribution"):
                paths.append(path)

    for table, group_col, name in [(by_age, "age_group", "age"), (by_school, "schooling_group", "schooling")]:
        if table.empty or not {"target", group_col, "feature", "mean_abs_contribution"}.issubset(table.columns):
            continue
        for (target, group), sub in table.groupby(["target", group_col], dropna=False):
            top = sub.sort_values("mean_abs_contribution", ascending=False).head(min(12, top_n)).sort_values("mean_abs_contribution")
            path = figures_dir / "interpretability" / name / f"top_features_{_safe_name(target)}_{_safe_name(group)}.png"
            if _barh(top, "feature", "mean_abs_contribution", f"Feature relevance: {target} | {name}={group}", path, "Mean absolute contribution"):
                paths.append(path)
    return paths


def generate_figures(run_dir: Path, figures_dir: Path | None = None) -> list[Path]:
    figures_dir = figures_dir or run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_model_results(run_dir, figures_dir))
    paths.extend(plot_correlations(run_dir, figures_dir))
    paths.extend(plot_interpretability(run_dir, figures_dir))
    manifest = pd.DataFrame({"figure": [str(p) for p in paths]})
    manifest.to_csv(figures_dir / "figures_manifest.csv", index=False)
    print(f"Generated {len(paths)} figures in {figures_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for one experiment run directory")
    parser.add_argument("--run-dir", default="outputs/01_run")
    parser.add_argument("--figures-dir", default="")
    parser.add_argument("--outputs-root", default="", help="Compatibility option. If provided without --run-dir, use outputs-root/01_run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if args.outputs_root and args.run_dir == "outputs/01_run":
        candidate = Path(args.outputs_root) / "01_run"
        if candidate.exists():
            run_dir = candidate
    figures_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "figures"
    generate_figures(run_dir, figures_dir)


if __name__ == "__main__":
    main()
