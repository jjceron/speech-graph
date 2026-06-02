from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    available_label_ratios,
    available_targets,
    append_manifest,
    ensure_figures_dir,
    filter_analysis_level,
    parse_csv_list,
    pick_window,
    read_data_csv,
    safe_name,
)
from .plots_utils import save_heatmap, save_horizontal_bars, save_scatter
from .stats_utils import CorrConfig, correlation_table, filter_correlations


def generate_label_figures(
    run_dir: Path,
    window_size: int = 30,
    level: str = "file",
    method: str = "spearman",
    min_nonzero: int = 8,
    min_n: int = 12,
    min_abs_r: float = 0.25,
    scatter_top_n: int = 6,
    group_col: str = "Tipo",
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found")
        return []
    df = filter_analysis_level(df, level)
    df_w = pick_window(df, window_size)
    out_dir = ensure_figures_dir(run_dir, "nlp_profile/labels")
    rows: list[dict] = []

    label_cols = available_label_ratios(df_w, min_nonzero=min_nonzero)
    if not label_cols:
        print(f"[{run_dir.name}] no label_ratio columns with min_nonzero={min_nonzero}")
        return []

    prevalence = []
    for col in label_cols:
        values = pd.to_numeric(df_w[col], errors="coerce").fillna(0)
        prevalence.append({
            "label": col.replace("label_ratio_", ""),
            "column": col,
            "mean_ratio": float(values.mean()),
            "nonzero_n": int((values > 0).sum()),
            "n": int(values.notna().sum()),
        })
    prev_df = pd.DataFrame(prevalence).sort_values(["nonzero_n", "mean_ratio"], ascending=[False, False])
    prev_path = out_dir / f"label_prevalence_w{window_size}.csv"
    prev_df.to_csv(prev_path, index=False)
    rows.append({"table": str(prev_path), "kind": "label_prevalence"})
    path = save_horizontal_bars(
        prev_df["label"].tolist(),
        prev_df["nonzero_n"].tolist(),
        out_dir / f"label_nonzero_counts_w{window_size}.png",
        f"{run_dir.name}: sujetos con etiqueta anotada (w{window_size})",
        xlabel="n con ratio > 0",
        xline_zero=False,
    )
    if path:
        rows.append({"figure": str(path), "kind": "label_prevalence"})

    targets = available_targets(df_w, "all")
    cfg = CorrConfig(method=method, min_n=min_n, min_abs_r=min_abs_r)
    corr = correlation_table(df_w, label_cols, targets, cfg)
    corr = filter_correlations(corr, min_abs_r=min_abs_r, min_n=min_n)
    if not corr.empty:
        corr_path = out_dir / f"label_relevant_correlations_w{window_size}.csv"
        corr.to_csv(corr_path, index=False)
        rows.append({"table": str(corr_path), "kind": "label_correlations"})

        top_labels = corr.groupby("metric")["abs_r"].max().sort_values(ascending=False).head(10).index.tolist()
        top_targets = corr.groupby("target")["abs_r"].max().sort_values(ascending=False).head(14).index.tolist()
        pivot = corr.pivot_table(index="metric", columns="target", values="r", aggfunc="mean").reindex(index=top_labels, columns=top_targets)
        pivot.index = [x.replace("label_ratio_", "") for x in pivot.index]
        path = save_heatmap(
            pivot,
            out_dir / f"label_correlations_heatmap_w{window_size}.png",
            f"{run_dir.name}: etiquetas [[...]] vs variables relevantes (w{window_size})",
            xlabel="variables",
            ylabel="etiquetas",
        )
        if path:
            rows.append({"figure": str(path), "kind": "label_heatmap"})

        top = corr.head(12)
        labels = [f"{r.metric.replace('label_ratio_', '')} ~ {r.target} (n={int(r.n)})" for r in top.itertuples()]
        path = save_horizontal_bars(
            labels,
            top["r"].tolist(),
            out_dir / f"label_top_correlations_w{window_size}.png",
            f"{run_dir.name}: asociaciones más fuertes de etiquetas (w{window_size})",
            xlabel="Spearman r",
        )
        if path:
            rows.append({"figure": str(path), "kind": "label_top_correlations"})

        made = 0
        for row in corr.itertuples():
            path = save_scatter(
                df_w,
                row.metric,
                row.target,
                out_dir / f"label_scatter_{made+1:02d}_{safe_name(row.metric)}_vs_{safe_name(row.target)}_w{window_size}.png",
                f"{run_dir.name}: {row.metric.replace('label_ratio_', '')} vs {row.target}\nSpearman r={row.r:.3f}, n={int(row.n)}",
                group_col=group_col if group_col in df_w.columns else None,
                xlabel=row.metric.replace("label_ratio_", "ratio_"),
            )
            if path:
                rows.append({"figure": str(path), "kind": "label_scatter", "metric": row.metric, "target": row.target})
                made += 1
            if made >= scatter_top_n:
                break

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] label figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate focused figures for essential [[...]] transcript labels.")
    parser.add_argument("--run-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-nonzero", type=int, default=8)
    parser.add_argument("--min-n", type=int, default=12)
    parser.add_argument("--min-abs-r", type=float, default=0.25)
    parser.add_argument("--scatter-top-n", type=int, default=6)
    parser.add_argument("--group-col", default="Tipo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_label_figures(
        Path(args.run_dir),
        window_size=args.window_size,
        level=args.level,
        method=args.method,
        min_nonzero=args.min_nonzero,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        scatter_top_n=args.scatter_top_n,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
