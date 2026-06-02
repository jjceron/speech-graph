from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from .common import append_manifest, ensure_figures_dir, filter_analysis_level, pick_window, read_data_csv, safe_name
from .plots_utils import save_heatmap, save_scatter


def _prepare(df: pd.DataFrame, window_size: int, level: str, metrics: list[str]) -> pd.DataFrame:
    df = filter_analysis_level(df, level)
    df = pick_window(df, window_size)
    cols = ["code"] + [m for m in metrics if m in df.columns]
    if "code" not in df.columns:
        return pd.DataFrame()
    out = df[cols].copy()
    # One row per code if duplicates exist.
    for m in metrics:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")
    return out.groupby("code", as_index=False).mean(numeric_only=True)


def generate_run_comparison(
    run_a: Path,
    run_b: Path,
    window_size: int = 30,
    level: str = "file",
    metrics: list[str] | None = None,
) -> list[dict]:
    metrics = metrics or ["mean_z_lcc", "mean_z_lsc", "mean_z_edges", "mean_lsc_ratio", "mean_nodes", "mean_density", "mean_asp"]
    df_a, _ = read_data_csv(run_a)
    df_b, _ = read_data_csv(run_b)
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        print("Could not read both runs")
        return []

    a = _prepare(df_a, window_size, level, metrics)
    b = _prepare(df_b, window_size, level, metrics)
    if a.empty or b.empty:
        return []
    merged = a.merge(b, on="code", suffixes=("_a", "_b"))
    out_dir = ensure_figures_dir(run_b, "nlp_profile/run_comparison")
    rows: list[dict] = []

    summary = []
    for metric in metrics:
        ca = f"{metric}_a"
        cb = f"{metric}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        x = pd.to_numeric(merged[ca], errors="coerce")
        y = pd.to_numeric(merged[cb], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        r, p = stats.spearmanr(x[mask], y[mask])
        summary.append({"metric": metric, "spearman_r": float(r), "p": float(p), "n": int(mask.sum())})
        path = save_scatter(
            merged,
            ca,
            cb,
            out_dir / f"compare_{safe_name(metric)}_w{window_size}.png",
            f"{metric}: {run_a.name} vs {run_b.name}\nSpearman r={r:.3f}, n={int(mask.sum())}",
            xlabel=f"{run_a.name}",
            ylabel=f"{run_b.name}",
        )
        if path:
            rows.append({"figure": str(path), "kind": "run_comparison_scatter", "metric": metric})
    if summary:
        summary_df = pd.DataFrame(summary).sort_values("spearman_r", ascending=False)
        table_path = out_dir / f"run_comparison_summary_w{window_size}.csv"
        summary_df.to_csv(table_path, index=False)
        rows.append({"table": str(table_path), "kind": "run_comparison_summary"})
        matrix = summary_df.set_index("metric")[["spearman_r"]]
        path = save_heatmap(
            matrix,
            out_dir / f"run_comparison_spearman_w{window_size}.png",
            f"Convergencia entre {run_a.name} y {run_b.name} (w{window_size})",
            xlabel="comparación",
            ylabel="métrica",
            vmin=-1,
            vmax=1,
        )
        if path:
            rows.append({"figure": str(path), "kind": "run_comparison_heatmap"})
    append_manifest(rows, run_b, subdir="nlp_profile")
    print(f"[{run_b.name}] run comparison figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare random100 vs random1000 outputs for stability/convergence.")
    parser.add_argument("--run-a", default="outputs/03_windows_random100")
    parser.add_argument("--run-b", default="outputs/04_windows_random1000")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--metrics", default="mean_z_lcc,mean_z_lsc,mean_z_edges,mean_lsc_ratio,mean_nodes,mean_density,mean_asp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_run_comparison(
        Path(args.run_a),
        Path(args.run_b),
        window_size=args.window_size,
        level=args.level,
        metrics=[x.strip() for x in args.metrics.split(",") if x.strip()],
    )


if __name__ == "__main__":
    main()
