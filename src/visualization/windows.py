from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization. Install with: py -m pip install matplotlib") from exc

from .common import append_manifest, available_metric_cols, ensure_figures_dir, parse_csv_list, read_data_csv, safe_name, stability_csv


def plot_stability_csv(stability: pd.DataFrame, output_dir: Path, run_name: str, metrics: list[str] | None = None, max_metrics: int = 16) -> list[dict]:
    rows: list[dict] = []
    required = {"window_size", "metric", "mean"}
    if stability.empty or not required.issubset(stability.columns):
        return rows

    stability = stability.copy()
    stability["mean"] = pd.to_numeric(stability["mean"], errors="coerce")
    stability["window_size"] = pd.to_numeric(stability["window_size"], errors="coerce")
    if metrics:
        metric_names = [m for m in metrics if m in set(stability["metric"].astype(str))]
    else:
        metric_names = stability.groupby("metric")["mean"].count().sort_values(ascending=False).head(max_metrics).index.tolist()

    for metric in metric_names[:max_metrics]:
        sub = stability[stability["metric"].astype(str) == str(metric)].dropna(subset=["mean", "window_size"])
        if sub.empty:
            continue
        group_cols = [c for c in ["level", "activity"] if c in sub.columns and sub[c].nunique(dropna=True) > 1]
        if group_cols:
            groups = sub.groupby(group_cols, dropna=False)
        else:
            groups = [("all", sub)]

        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for group_label, group_df in groups:
            agg = group_df.groupby("window_size", dropna=False)["mean"].mean().reset_index().sort_values("window_size")
            if len(agg) < 1:
                continue
            label = str(group_label)
            ax.plot(agg["window_size"], agg["mean"], marker="o", label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{run_name}: estabilidad de {metric} por ventana")
        ax.set_xlabel("window_size")
        ax.set_ylabel(metric)
        if group_cols:
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"stability_{safe_name(metric)}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        rows.append({"figure": str(path), "kind": "window_stability", "metric": metric})
    return rows


def plot_data_by_window(df: pd.DataFrame, output_dir: Path, run_name: str, metrics: list[str] | None = None, max_metrics: int = 16) -> list[dict]:
    rows: list[dict] = []
    if "window_size" not in df.columns or df["window_size"].nunique(dropna=True) < 2:
        return rows
    metric_cols = available_metric_cols(df, metrics, max_metrics=max_metrics)
    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce")
        sub = df.loc[values.notna(), ["window_size"]].copy()
        sub[metric] = values[values.notna()]
        if sub.empty:
            continue
        agg = sub.groupby("window_size", dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
        agg["window_size"] = pd.to_numeric(agg["window_size"], errors="coerce")
        agg = agg.dropna(subset=["window_size"]).sort_values("window_size")
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(agg["window_size"], agg["mean"], yerr=agg["std"], marker="o", capsize=4)
        ax.set_title(f"{run_name}: media de {metric} por ventana")
        ax.set_xlabel("window_size")
        ax.set_ylabel(metric)
        fig.tight_layout()
        path = output_dir / f"mean_by_window_{safe_name(metric)}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        rows.append({"figure": str(path), "kind": "mean_by_window", "metric": metric})
    return rows


def generate_window_figures(run_dir: Path, metrics: list[str] | None = None, max_metrics: int = 16) -> list[dict]:
    out_dir = ensure_figures_dir(run_dir, "window_comparison")
    rows: list[dict] = []

    st_path = stability_csv(run_dir)
    if st_path:
        try:
            stability = pd.read_csv(st_path)
            rows.extend(plot_stability_csv(stability, out_dir, run_dir.name, metrics=metrics, max_metrics=max_metrics))
        except Exception as exc:
            print(f"Could not plot stability from {st_path}: {exc}")

    df, data_path = read_data_csv(run_dir)
    if df is not None:
        rows.extend(plot_data_by_window(df, out_dir, run_dir.name, metrics=metrics, max_metrics=max_metrics))

    append_manifest(rows, run_dir)
    print(f"[{run_dir.name}] window figures: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create window-comparison figures from existing outputs.")
    parser.add_argument("--run-dir", default="outputs/01_windows_no_random")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--max-metrics", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = parse_csv_list(args.metrics) or None
    generate_window_figures(Path(args.run_dir), metrics=metrics, max_metrics=args.max_metrics)


if __name__ == "__main__":
    main()
