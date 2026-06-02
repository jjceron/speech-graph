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

from .common import DEFAULT_GROUP_COLS, append_manifest, available_metric_cols, ensure_figures_dir, parse_csv_list, read_data_csv, safe_name


def _make_group_labels(df: pd.DataFrame, group_col: str) -> pd.Series:
    labels = df[group_col].astype(str).fillna("NA")
    if "window_size" in df.columns and df["window_size"].nunique(dropna=True) > 1:
        labels = labels + " | w" + df["window_size"].astype(str)
    return labels


def plot_group_boxplot(df: pd.DataFrame, metric: str, group_col: str, output_dir: Path, run_name: str) -> list[dict]:
    if metric not in df.columns or group_col not in df.columns:
        return []
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[values.notna() & df[group_col].notna()].copy()
    valid[metric] = values[values.notna()]
    if valid.empty or valid[group_col].nunique(dropna=True) < 2:
        return []

    valid["_group_label"] = _make_group_labels(valid, group_col)
    grouped = []
    labels = []
    for label, sub in valid.groupby("_group_label", dropna=False):
        series = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if len(series) >= 2:
            grouped.append(series.to_numpy())
            labels.append(str(label))
    if len(grouped) < 2:
        return []

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 5))
    ax.boxplot(grouped, labels=labels, showmeans=True)
    ax.set_title(f"{run_name}: {metric} por {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    path = output_dir / f"{safe_name(metric)}_by_{safe_name(group_col)}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return [{"figure": str(path), "kind": "group_boxplot", "metric": metric, "group": group_col}]


def generate_group_figures(
    run_dir: Path,
    group_cols: list[str] | None = None,
    metrics: list[str] | None = None,
    max_metrics: int = 10,
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found for group figures")
        return []

    out_dir = ensure_figures_dir(run_dir, "group_profiles")
    group_cols = group_cols or [c for c in DEFAULT_GROUP_COLS if c in df.columns]
    metric_cols = available_metric_cols(df, metrics, max_metrics=max_metrics)

    rows: list[dict] = []
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        for metric in metric_cols:
            rows.extend(plot_group_boxplot(df, metric, group_col, out_dir, run_dir.name))
    append_manifest(rows, run_dir)
    print(f"[{run_dir.name}] group figures: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create group profile figures from existing outputs.")
    parser.add_argument("--run-dir", default="outputs/01_windows_no_random")
    parser.add_argument("--group-cols", default="")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--max-metrics", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_cols = parse_csv_list(args.group_cols) or None
    metrics = parse_csv_list(args.metrics) or None
    generate_group_figures(Path(args.run_dir), group_cols=group_cols, metrics=metrics, max_metrics=args.max_metrics)


if __name__ == "__main__":
    main()
