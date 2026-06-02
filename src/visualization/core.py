from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    BARRATT_ITEM_TARGETS,
    available_core_metrics,
    available_targets,
    append_manifest,
    ensure_figures_dir,
    filter_analysis_level,
    parse_csv_list,
    pick_window,
    read_data_csv,
    safe_name,
    write_text_report,
)
from .plots_utils import save_heatmap, save_horizontal_bars, save_scatter
from .stats_utils import CorrConfig, correlation_table, filter_correlations


def _correlation_matrix(corr: pd.DataFrame, metrics: list[str], targets: list[str]) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(index=metrics, columns=targets, dtype=float)
    pivot = corr.pivot_table(index="metric", columns="target", values="r", aggfunc="mean")
    return pivot.reindex(index=metrics, columns=targets)


def _top_corr_plot(corr: pd.DataFrame, out_dir: Path, run_name: str, label: str, top_n: int) -> list[dict]:
    rows: list[dict] = []
    top = corr.head(top_n).copy()
    if top.empty:
        return rows
    labels = [f"{r.metric} ~ {r.target}  (n={int(r.n)})" for r in top.itertuples()]
    path = save_horizontal_bars(
        labels,
        top["r"].tolist(),
        out_dir / f"top_correlations_{safe_name(label)}.png",
        f"{run_name}: correlaciones relevantes ({label})",
        xlabel="Spearman r",
    )
    if path:
        rows.append({"figure": str(path), "kind": "top_correlations", "target_set": label})
    return rows


def _scatter_top(corr: pd.DataFrame, df: pd.DataFrame, out_dir: Path, run_name: str, label: str, top_n: int, group_col: str | None) -> list[dict]:
    rows: list[dict] = []
    made = 0
    for row in corr.itertuples():
        metric = row.metric
        target = row.target
        title = f"{run_name}: {metric} vs {target}\nSpearman r={row.r:.3f}, n={int(row.n)}"
        path = save_scatter(
            df,
            metric,
            target,
            out_dir / f"scatter_{safe_name(label)}_{made+1:02d}_{safe_name(metric)}_vs_{safe_name(target)}.png",
            title,
            group_col=group_col,
        )
        if path:
            rows.append({"figure": str(path), "kind": "scatter", "metric": metric, "target": target, "target_set": label})
            made += 1
        if made >= top_n:
            break
    return rows


def generate_core_figures(
    run_dir: Path,
    window_size: int = 30,
    level: str = "file",
    method: str = "spearman",
    min_n: int = 20,
    min_abs_r: float = 0.20,
    top_n: int = 12,
    scatter_top_n: int = 6,
    group_col: str = "Tipo",
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found")
        return []

    df = filter_analysis_level(df, level)
    df_w = pick_window(df, window_size)
    out_dir = ensure_figures_dir(run_dir, "nlp_profile/core")
    rows: list[dict] = []

    metrics = available_core_metrics(df_w)
    if not metrics:
        print(f"[{run_dir.name}] no core metrics available")
        return []

    cfg = CorrConfig(method=method, min_n=min_n, min_abs_r=min_abs_r)
    all_corr_frames: list[pd.DataFrame] = []

    for target_set in ["demographic", "barratt", "cognitive"]:
        targets = available_targets(df_w, target_set)
        if not targets:
            continue
        corr = correlation_table(df_w, metrics, targets, cfg)
        corr = filter_correlations(corr, min_abs_r=min_abs_r, min_n=min_n)
        if corr.empty:
            continue
        corr.insert(0, "target_set", target_set)
        all_corr_frames.append(corr)

        selected_targets = corr.groupby("target")["abs_r"].max().sort_values(ascending=False).head(12).index.tolist()
        selected_metrics = corr.groupby("metric")["abs_r"].max().sort_values(ascending=False).head(12).index.tolist()
        matrix = _correlation_matrix(corr, selected_metrics, selected_targets)
        path = save_heatmap(
            matrix,
            out_dir / f"heatmap_{safe_name(target_set)}_w{window_size}.png",
            f"{run_dir.name}: métricas NLP relevantes vs {target_set} (w{window_size})",
            xlabel="variables",
            ylabel="métricas NLP",
        )
        if path:
            rows.append({"figure": str(path), "kind": "core_heatmap", "target_set": target_set})
        rows.extend(_top_corr_plot(corr, out_dir, run_dir.name, f"{target_set}_w{window_size}", top_n=top_n))
        rows.extend(_scatter_top(corr, df_w, out_dir, run_dir.name, f"{target_set}_w{window_size}", scatter_top_n, group_col if group_col in df_w.columns else None))

    # Barratt items: include only if some item-level association survives the same filter.
    item_targets = available_targets(df_w, BARRATT_ITEM_TARGETS)
    if item_targets:
        item_corr = correlation_table(df_w, metrics, item_targets, cfg)
        item_corr = filter_correlations(item_corr, min_abs_r=max(min_abs_r, 0.25), min_n=min_n)
        if not item_corr.empty:
            item_corr.insert(0, "target_set", "barratt_items")
            all_corr_frames.append(item_corr)
            top_items = item_corr.groupby("target")["abs_r"].max().sort_values(ascending=False).head(14).index.tolist()
            top_metrics = item_corr.groupby("metric")["abs_r"].max().sort_values(ascending=False).head(10).index.tolist()
            matrix = _correlation_matrix(item_corr, top_metrics, top_items)
            path = save_heatmap(
                matrix,
                out_dir / f"heatmap_barratt_items_w{window_size}.png",
                f"{run_dir.name}: ítems Barratt con señal NLP (w{window_size})",
                xlabel="ítems Barratt",
                ylabel="métricas NLP",
            )
            if path:
                rows.append({"figure": str(path), "kind": "barratt_items_heatmap"})
            rows.extend(_top_corr_plot(item_corr, out_dir, run_dir.name, f"barratt_items_w{window_size}", top_n=top_n))

    if all_corr_frames:
        all_corr = pd.concat(all_corr_frames, ignore_index=True)
        table_path = out_dir / f"relevant_correlations_w{window_size}.csv"
        all_corr.to_csv(table_path, index=False)
        rows.append({"table": str(table_path), "kind": "relevant_correlations"})
        msg = [
            f"Run: {run_dir.name}",
            f"Data: {data_path}",
            f"Level: {level}",
            f"Window: {window_size}",
            f"Method: {method}",
            f"Filters: min_n={min_n}, min_abs_r={min_abs_r}",
            "",
            "Solo se grafican correlaciones con variables de interés; se excluyen token_count, window_count, edges mecánicas, LCC redundante y métricas globales dominadas por longitud.",
            f"Correlaciones relevantes guardadas: {len(all_corr)}",
        ]
    else:
        msg = [
            f"Run: {run_dir.name}",
            "No hubo correlaciones relevantes bajo los filtros actuales.",
            f"Filtros usados: min_n={min_n}, min_abs_r={min_abs_r}",
        ]
    report_path = write_text_report(run_dir, msg, filename=f"README_core_w{window_size}.txt", subdir="nlp_profile/core")
    rows.append({"table": str(report_path), "kind": "readme"})

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] core figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate only meaningful NLP/speech-graph correlation figures.")
    parser.add_argument("--run-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-abs-r", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--scatter-top-n", type=int, default=6)
    parser.add_argument("--group-col", default="Tipo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_core_figures(
        Path(args.run_dir),
        window_size=args.window_size,
        level=args.level,
        method=args.method,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        top_n=args.top_n,
        scatter_top_n=args.scatter_top_n,
        group_col=args.group_col,
    )


if __name__ == "__main__":
    main()
