from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.activity_focus import generate_activity_focus_outputs, parse_int_list
from .common import append_manifest, ensure_figures_dir, read_data_csv, safe_name
from .plots_utils import save_heatmap, save_horizontal_bars, save_scatter


def _pivot_corr(df: pd.DataFrame, activity: str, section: str, top_rows: int = 12, top_cols: int = 12) -> pd.DataFrame:
    sub = df[(df["section"] == section) & (df["activity"] == activity)].copy()
    if sub.empty:
        return pd.DataFrame()
    metrics = sub.groupby("metric")["abs_r"].max().sort_values(ascending=False).head(top_rows).index.tolist()
    targets = sub.groupby("target")["abs_r"].max().sort_values(ascending=False).head(top_cols).index.tolist()
    mat = sub.pivot_table(index="metric", columns="target", values="r", aggfunc="mean")
    return mat.reindex(index=metrics, columns=targets)


def generate_activity_figures(
    run_dir: Path,
    window_size: int = 30,
    target_activities: str = "2,6,7",
    secondary_activities: str = "1,4,5",
    method: str = "spearman",
    min_n: int = 30,
    min_abs_r: float = 0.15,
    label_min_nonzero: int = 20,
    group_col: str = "Tipo",
    scatter_top_n: int = 4,
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found for activity figures")
        return []

    results, csv_path, report_path = generate_activity_focus_outputs(
        input_csv=data_path,
        output_dir=run_dir,
        window_size=window_size,
        target_activities=target_activities,
        secondary_activities=secondary_activities,
        method=method,
        min_n=min_n,
        min_abs_r=min_abs_r,
        label_min_nonzero=label_min_nonzero,
        group_col=group_col,
    )

    out_dir = ensure_figures_dir(run_dir, "nlp_profile/activity_focus")
    rows: list[dict] = [
        {"table": str(csv_path), "kind": "activity_focus_results"},
        {"table": str(report_path), "kind": "activity_focus_report"},
    ]

    if results.empty:
        append_manifest(rows, run_dir, subdir="nlp_profile")
        return rows

    activities = sorted(set(results.loc[results["section"].isin(["graph_correlation", "label_correlation"]), "activity"].dropna()))
    for activity in activities:
        for section, prefix, title_part in [
            ("graph_correlation", "graph", "métricas de grafos"),
            ("label_correlation", "labels", "etiquetas"),
        ]:
            mat = _pivot_corr(results, activity, section)
            if not mat.empty:
                path = save_heatmap(
                    mat,
                    out_dir / f"heatmap_{prefix}_{safe_name(activity)}_w{window_size}.png",
                    f"{run_dir.name}: {title_part} vs variables ({activity}, w{window_size})",
                    xlabel="variables",
                    ylabel="métricas/etiquetas",
                )
                if path:
                    rows.append({"figure": str(path), "kind": f"activity_{prefix}_heatmap", "activity": activity})

            sub = results[(results["section"] == section) & (results["activity"] == activity)].sort_values(["abs_r", "p"], ascending=[False, True]).head(12)
            if not sub.empty:
                labels = [f"{r.metric} ~ {r.target} (n={int(r.n)})" for r in sub.itertuples()]
                path = save_horizontal_bars(
                    labels,
                    sub["r"].tolist(),
                    out_dir / f"top_{prefix}_{safe_name(activity)}_w{window_size}.png",
                    f"{run_dir.name}: top {title_part} ({activity}, w{window_size})",
                    xlabel="Spearman r",
                )
                if path:
                    rows.append({"figure": str(path), "kind": f"activity_{prefix}_top", "activity": activity})

    # Scatter plots for the strongest activity-specific associations.
    plot_df = df.copy()
    if "window_size" in plot_df.columns:
        plot_df = plot_df[pd.to_numeric(plot_df["window_size"], errors="coerce").eq(window_size)].copy()
    if "level" in plot_df.columns:
        plot_df = plot_df[plot_df["level"].astype(str).str.lower().eq("activity")].copy()
    if "activity_number" not in plot_df.columns:
        plot_df["activity_number"] = plot_df["activity"].astype(str).str.extract(r"(\d+)")[0]
    plot_df["activity"] = plot_df["activity_number"].map(lambda x: f"Actividad{int(float(x))}" if pd.notna(x) else "NA")

    if {"section", "abs_r", "p"}.issubset(results.columns):
        top = results[results["section"].isin(["graph_correlation", "label_correlation"])].sort_values(["abs_r", "p"], ascending=[False, True]).head(scatter_top_n)
    else:
        top = pd.DataFrame()
    made = 0
    for r in top.itertuples():
        sub = plot_df[plot_df["activity"].eq(r.activity)]
        if sub.empty:
            continue
        path = save_scatter(
            sub,
            r.metric,
            r.target,
            out_dir / f"scatter_{made+1:02d}_{safe_name(r.activity)}_{safe_name(r.metric)}_vs_{safe_name(r.target)}.png",
            f"{run_dir.name}: {r.metric} vs {r.target}\n{r.activity}, r={r.r:.3f}, n={int(r.n)}",
            group_col=group_col if group_col in sub.columns else None,
        )
        if path:
            rows.append({"figure": str(path), "kind": "activity_scatter", "activity": r.activity, "metric": r.metric, "target": r.target})
            made += 1

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] activity figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate activity-focused figures and two-file report.")
    parser.add_argument("--run-dir", default="outputs/02_w30_by_activity_no_random_all")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--target-activities", default="2,6,7")
    parser.add_argument("--secondary-activities", default="1,4,5")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--label-min-nonzero", type=int, default=20)
    parser.add_argument("--group-col", default="Tipo")
    parser.add_argument("--scatter-top-n", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_activity_figures(
        Path(args.run_dir),
        window_size=args.window_size,
        target_activities=args.target_activities,
        secondary_activities=args.secondary_activities,
        method=args.method,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        label_min_nonzero=args.label_min_nonzero,
        group_col=args.group_col,
        scatter_top_n=args.scatter_top_n,
    )


if __name__ == "__main__":
    main()
