from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    DEFAULT_GROUP_COLS,
    available_core_metrics,
    append_manifest,
    ensure_figures_dir,
    filter_analysis_level,
    parse_csv_list,
    pick_window,
    read_data_csv,
    safe_name,
)
from .plots_utils import save_boxplot_by_group, save_heatmap
from .stats_utils import zscore


def generate_group_profile_figures(
    run_dir: Path,
    window_size: int = 30,
    level: str = "file",
    group_cols: list[str] | None = None,
    max_groups: int = 2,
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found")
        return []
    df = filter_analysis_level(df, level)
    df_w = pick_window(df, window_size)
    out_dir = ensure_figures_dir(run_dir, "nlp_profile/groups")
    rows: list[dict] = []

    metrics = available_core_metrics(df_w)
    if not metrics:
        return []
    group_cols = group_cols or ["Tipo", "Educational level"]
    group_cols = [g for g in group_cols if g in df_w.columns and df_w[g].nunique(dropna=True) > 1]
    if not group_cols:
        print(f"[{run_dir.name}] no usable grouping columns")
        return []

    # Standardized group means make cross-metric profiles interpretable.
    zdf = df_w.copy()
    for metric in metrics:
        zdf[f"zstd__{metric}"] = zscore(zdf[metric])

    for group_col in group_cols[:max_groups]:
        counts = df_w[group_col].value_counts(dropna=False)
        valid_groups = counts[counts >= 5].index.tolist()
        if len(valid_groups) < 2:
            continue
        sub = zdf[zdf[group_col].isin(valid_groups)].copy()
        zmetrics = [f"zstd__{m}" for m in metrics]
        profile = sub.groupby(group_col, dropna=False)[zmetrics].mean().T
        profile.index = [x.replace("zstd__", "") for x in profile.index]
        # Keep metrics that separate at least one group from another.
        spread = profile.max(axis=1) - profile.min(axis=1)
        profile = profile.loc[spread.sort_values(ascending=False).head(10).index]
        path = save_heatmap(
            profile,
            out_dir / f"group_profile_{safe_name(group_col)}_w{window_size}.png",
            f"{run_dir.name}: perfil NLP estandarizado por {group_col} (w{window_size})",
            xlabel=group_col,
            ylabel="métricas NLP",
            vmin=None,
            vmax=None,
        )
        if path:
            rows.append({"figure": str(path), "kind": "group_profile_heatmap", "group_col": group_col})
        profile_path = out_dir / f"group_profile_{safe_name(group_col)}_w{window_size}.csv"
        profile.to_csv(profile_path)
        rows.append({"table": str(profile_path), "kind": "group_profile_table", "group_col": group_col})

        for metric in profile.index[:3]:
            path = save_boxplot_by_group(
                df_w,
                metric,
                group_col,
                out_dir / f"boxplot_{safe_name(metric)}_by_{safe_name(group_col)}_w{window_size}.png",
                f"{run_dir.name}: {metric} por {group_col} (w{window_size})",
            )
            if path:
                rows.append({"figure": str(path), "kind": "group_boxplot", "metric": metric, "group_col": group_col})

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] group profile figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create focused group profiling figures.")
    parser.add_argument("--run-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--group-cols", default="Tipo,Educational level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_group_profile_figures(
        Path(args.run_dir),
        window_size=args.window_size,
        level=args.level,
        group_cols=parse_csv_list(args.group_cols),
    )


if __name__ == "__main__":
    main()
