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

from .common import append_manifest, available_metric_cols, ensure_figures_dir, parse_csv_list, read_data_csv, safe_name


def plot_metric_distribution(df: pd.DataFrame, metric: str, output_dir: Path, run_name: str) -> list[dict]:
    rows: list[dict] = []
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[values.notna()].copy()
    valid[metric] = values[values.notna()]
    if valid.empty:
        return rows

    if "window_size" in valid.columns and valid["window_size"].nunique(dropna=True) > 1:
        labels = []
        data = []
        for w, sub in valid.groupby("window_size", dropna=False):
            series = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if not series.empty:
                labels.append(str(w))
                data.append(series.to_numpy())
        if data:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_title(f"{run_name}: {metric} por tamaño de ventana")
            ax.set_xlabel("window_size")
            ax.set_ylabel(metric)
            fig.tight_layout()
            path = output_dir / f"{safe_name(metric)}_by_window.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            rows.append({"figure": str(path), "kind": "distribution_by_window", "metric": metric})
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(valid[metric].to_numpy(), bins=30)
        ax.set_title(f"{run_name}: distribución de {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("frecuencia")
        fig.tight_layout()
        path = output_dir / f"{safe_name(metric)}_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rows.append({"figure": str(path), "kind": "histogram", "metric": metric})

    return rows


def generate_distribution_figures(run_dir: Path, metrics: list[str] | None = None, max_metrics: int = 16) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found for distributions")
        return []

    out_dir = ensure_figures_dir(run_dir, "metric_distributions_by_window")
    metric_cols = available_metric_cols(df, metrics, max_metrics=max_metrics)
    rows: list[dict] = []
    for metric in metric_cols:
        rows.extend(plot_metric_distribution(df, metric, out_dir, run_dir.name))
    append_manifest(rows, run_dir)
    print(f"[{run_dir.name}] distribution figures: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create metric distribution figures from existing outputs.")
    parser.add_argument("--run-dir", default="outputs/01_windows_no_random")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--max-metrics", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = parse_csv_list(args.metrics) or None
    generate_distribution_figures(Path(args.run_dir), metrics=metrics, max_metrics=args.max_metrics)


if __name__ == "__main__":
    main()
