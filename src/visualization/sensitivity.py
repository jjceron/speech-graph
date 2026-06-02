from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    available_core_metrics,
    available_targets,
    append_manifest,
    ensure_figures_dir,
    filter_analysis_level,
    parse_csv_list,
    read_data_csv,
    safe_name,
)
from .plots_utils import save_heatmap, save_line_by_window_group
from .stats_utils import CorrConfig, correlation_table, filter_correlations


def generate_sensitivity_figures(
    run_dir: Path,
    level: str = "file",
    method: str = "spearman",
    min_n: int = 20,
    min_abs_r: float = 0.20,
    group_col: str = "Tipo",
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found")
        return []
    df = filter_analysis_level(df, level)
    if "window_size" not in df.columns or df["window_size"].nunique(dropna=True) < 2:
        print(f"[{run_dir.name}] no multiple window sizes available")
        return []
    out_dir = ensure_figures_dir(run_dir, "nlp_profile/window_sensitivity")
    rows: list[dict] = []

    metrics = available_core_metrics(df)
    targets = available_targets(df, "all")
    cfg = CorrConfig(method=method, min_n=min_n, min_abs_r=min_abs_r)

    corr_frames: list[pd.DataFrame] = []
    for ws, sub in df.groupby("window_size", dropna=False):
        corr = correlation_table(sub, metrics, targets, cfg)
        corr = filter_correlations(corr, min_abs_r=min_abs_r, min_n=min_n)
        if corr.empty:
            continue
        corr.insert(0, "window_size", ws)
        corr_frames.append(corr)
    if corr_frames:
        corr_all = pd.concat(corr_frames, ignore_index=True)
        corr_path = out_dir / "window_sensitivity_relevant_correlations.csv"
        corr_all.to_csv(corr_path, index=False)
        rows.append({"table": str(corr_path), "kind": "window_sensitivity_correlations"})

        corr_all["pair"] = corr_all["metric"] + " ~ " + corr_all["target"]
        top_pairs = corr_all.groupby("pair")["abs_r"].max().sort_values(ascending=False).head(18).index.tolist()
        pivot = corr_all[corr_all["pair"].isin(top_pairs)].pivot_table(
            index="pair", columns="window_size", values="r", aggfunc="mean"
        )
        # Stable pairs first: those appearing in several windows, then max absolute r.
        order_df = pd.DataFrame({
            "n_windows": pivot.notna().sum(axis=1),
            "max_abs_r": pivot.abs().max(axis=1),
        }).sort_values(["n_windows", "max_abs_r"], ascending=[False, False])
        pivot = pivot.loc[order_df.index]
        path = save_heatmap(
            pivot,
            out_dir / "window_sensitivity_heatmap.png",
            f"{run_dir.name}: estabilidad de correlaciones relevantes por ventana",
            xlabel="window_size",
            ylabel="métrica ~ variable",
        )
        if path:
            rows.append({"figure": str(path), "kind": "window_sensitivity_heatmap"})

    # Group trajectories only for a very small, interpretable set.
    if group_col in df.columns and df[group_col].nunique(dropna=True) > 1:
        for metric in [m for m in ["mean_z_lsc", "mean_z_lcc", "mean_lsc_ratio", "mean_nodes", "mean_density", "mean_asp"] if m in df.columns]:
            path = save_line_by_window_group(
                df,
                metric,
                group_col,
                out_dir / f"{safe_name(metric)}_by_window_and_{safe_name(group_col)}.png",
                f"{run_dir.name}: {metric} por ventana y {group_col}",
            )
            if path:
                rows.append({"figure": str(path), "kind": "metric_by_window_group", "metric": metric, "group_col": group_col})

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] window sensitivity figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate useful 10/20/30 window sensitivity figures only.")
    parser.add_argument("--run-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-abs-r", type=float, default=0.20)
    parser.add_argument("--group-col", default="Tipo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_sensitivity_figures(
        Path(args.run_dir),
        level=args.level,
        method=args.method,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
