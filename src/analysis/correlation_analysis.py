"""Correlation analysis: simple and partial Spearman correlations
between speech-graph features and Barratt impulsivity targets.

Targets: MOT, COG, MOT_V4 (items 8+13+16+21+23), COG_V1 (items 3+6).

Usage:
    py -m src.analysis.correlation_analysis --task 2 --experiment raw --adj-var "School year"
    py -m src.analysis.correlation_analysis --task 2 --experiment raw --window T2W10 --adj-var "School year"
    py -m src.analysis.correlation_analysis --task 7 --experiment zscores --adj-var "Age"
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.experiment_config import get_experiment_feature_names
from src.pipeline.speechgraph import load_metadata
from src.visualization.corr_metrics import plot_correlation_heatmaps

# New targets derived from BIS-11 items
ITEM = {
    "MOT_V4": ["8.", "13.", "16.", "21.", "23."],
    "COG_V1": ["3.", "6."],
}


def find_means_tables(metrics_dir: Path, tasks: list[int] | None = None) -> list[dict]:
    records = []
    for task_dir in sorted(metrics_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("Task"):
            continue
        task_num = int(task_dir.name.replace("Task", ""))
        if tasks is not None and task_num not in tasks:
            continue
        for fpath in sorted(task_dir.iterdir()):
            name = fpath.name
            if "means_params_table" in name:
                is_z = name.startswith("z_")
                tag = name.replace("z_means_params_table", "").replace("means_params_table", "").replace(".txt", "")
                records.append({
                    "path": fpath,
                    "task": task_num,
                    "tag": tag,
                    "type": "z" if is_z else "raw",
                })
    return records


def load_feature_table(
    path: Path,
    z_metrics: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    mapping = {}
    for col in df.columns:
        if col == "file":
            continue
        if col.startswith("z_"):
            mapping[col] = col
        else:
            mapping[col] = col
    df = df.rename(columns=mapping)
    df["file"] = df["file"].apply(lambda x: Path(str(x)).stem)
    for suffix in ("_CorrEtiq", "-CorrEtiq", " CorrEtiq"):
        df["file"] = df["file"].str.replace(suffix, "", regex=False)
    df["file"] = df["file"].str.strip()
    return df


def _compute_targets(meta: pd.DataFrame) -> pd.DataFrame:
    """Add MOT_V4 and COG_V1 columns to metadata."""
    meta = meta.copy()
    meta["MOT_V4"] = meta[ITEM["MOT_V4"]].sum(axis=1)
    meta["COG_V1"] = meta[ITEM["COG_V1"]].sum(axis=1)
    return meta


def run_correlation_analysis(
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/06_correlations",
    tasks: list[int] | None = None,
    experiment: str = "raw",
    adj_var: str = "School year",
    window: str | None = None,
) -> None:
    TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]
    PARTIAL_COLS = [f"r_partial_{t}" for t in TARGETS] + [f"p_partial_{t}" for t in TARGETS]
    SIMPLE_COLS = [f"r_{t}" for t in TARGETS] + [f"p_{t}" for t in TARGETS]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(metadata_path)
    meta = _compute_targets(meta)

    tables = find_means_tables(Path(metrics_dir), tasks=tasks)
    if not tables:
        print("No means tables found.")
        return

    if experiment == "raw":
        tables = [t for t in tables if t["type"] == "raw"]
    elif experiment == "zscores":
        tables = [t for t in tables if t["type"] == "z"]
    if not tables:
        print(f"No tables found for experiment '{experiment}'.")
        return

    if window is not None:
        tables = [t for t in tables if t["tag"] == window]
        if not tables:
            print(f"No tables found for window '{window}'.")
            return

    task_str = f"tasks {tasks}" if tasks else "all tasks"
    window_str = f", window='{window}'" if window else ""
    print(f"Processing: {task_str}, experiment='{experiment}'{window_str}")
    print(f"Found {len(tables)} means tables ({sum(1 for t in tables if t['type']=='raw')} raw, {sum(1 for t in tables if t['type']=='z')} z)")

    print(f"Targets: {TARGETS}")
    print(f"  MOT_V4 = {' + '.join(ITEM['MOT_V4'])}")
    print(f"  COG_V1 = {' + '.join(ITEM['COG_V1'])}")

    if not pd.api.types.is_numeric_dtype(meta[adj_var]):
        cats = sorted(meta[adj_var].unique())
        mapping = {v: i for i, v in enumerate(cats)}
        print(f"  Encoding '{adj_var}': {mapping}")

    all_rows = []

    for entry in tables:
        task = entry["task"]
        tag = entry["tag"]
        ftype = entry["type"]
        path = entry["path"]

        feats = load_feature_table(path)
        merged = feats.merge(meta, left_on="file", right_on="Cod", how="inner")

        window_int = int(tag.split("W")[1])
        exp_features = get_experiment_feature_names(experiment, window_int)
        feature_cols = [c for c in feats.columns if c != "file"]
        if ftype == "raw":
            feature_cols = [c for c in feature_cols if c in exp_features and not c.startswith("z_")]
        else:
            feature_cols = [c for c in feature_cols if c in exp_features and c.startswith("z_")]

        for col in feature_cols:
            valid = merged[col].notna()
            for tgt in TARGETS:
                valid = valid & merged[tgt].notna()
            valid = valid & merged[adj_var].notna()
            sub = merged[valid]
            if len(sub) < 10:
                continue

            x = sub[col].values.astype(float)
            if np.nanvar(x) < 1e-10:
                continue

            if not pd.api.types.is_numeric_dtype(sub[adj_var]):
                cats = sorted(sub[adj_var].unique())
                mapping = {v: i for i, v in enumerate(cats)}
                cov_vals = sub[adj_var].map(mapping).values.astype(float)
            else:
                cov_vals = sub[adj_var].values.astype(float)

            row = {
                "feature": col,
                "type": ftype,
                "task": task,
                "window_tag": tag,
                "n": len(sub),
            }

            for tgt in TARGETS:
                y = sub[tgt].values.astype(float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_val, p_val = spearmanr(x, y)
                row[f"r_{tgt}"] = r_val
                row[f"p_{tgt}"] = p_val

                try:
                    import pingouin as pg
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        partial_df = pd.DataFrame({"feat": x, "target": y, "cov": cov_vals})
                        pcorr = pg.partial_corr(
                            data=partial_df, x="feat", y="target", covar="cov",
                            method="spearman"
                        )
                    rp = pcorr["r"].values[0]
                    pval_col = [c for c in pcorr.columns if c.startswith("p")]
                    if not pval_col:
                        raise KeyError(f"No p-value column found in {list(pcorr.columns)}")
                    pp = pcorr[pval_col[0]].values[0]
                except Exception:
                    rp = np.nan
                    pp = np.nan

                row[f"r_partial_{tgt}"] = rp
                row[f"p_partial_{tgt}"] = pp

            all_rows.append(row)

    if not all_rows:
        print("No correlations computed.")
        return

    results = pd.DataFrame(all_rows)

    # Absolute value columns for ranking
    for tgt in TARGETS:
        results[f"abs_r_{tgt}"] = results[f"r_{tgt}"].abs()
        results[f"abs_r_partial_{tgt}"] = results[f"r_partial_{tgt}"].abs()

    ftypes_present = results["type"].unique()
    for ftype in ftypes_present:
        subset = results[results["type"] == ftype].copy()

        for target in TARGETS:
            col_r = f"abs_r_{target}"
            top = subset.sort_values(col_r, ascending=False).head(30)
            clean = top.drop(columns=[c for c in top.columns if c.startswith("abs_")])
            out_path = output_dir / f"top_simple_{target}_{ftype}.csv"
            clean.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

        for target in TARGETS:
            col_r = f"abs_r_partial_{target}"
            top = subset.sort_values(col_r, ascending=False).head(30)
            clean = top.drop(columns=[c for c in top.columns if c.startswith("abs_")])
            out_path = output_dir / f"top_partial_{target}_{ftype}.csv"
            clean.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

    full_path = output_dir / "all_correlations.csv"
    results.to_csv(full_path, index=False)
    print(f"  Saved: {full_path.name} ({len(results)} rows)")

    print()
    agg = {}
    for tgt in TARGETS:
        agg[f"mean_abs_r_{tgt}"] = (f"r_{tgt}", lambda x: x.abs().mean())
        agg[f"mean_abs_partial_{tgt}"] = (f"r_partial_{tgt}", lambda x: x.dropna().abs().mean())
    summary = results.groupby("type").agg(**{k: v for k, v in agg.items()}).round(4)
    print(summary.to_string())
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path.name}")

    try:
        import matplotlib
        plot_correlation_heatmaps(results, output_dir, adj_var=adj_var, window=window)
    except Exception as e:
        print(f"  Warning: heatmap plotting failed: {e}")

    print("\nDone. Correlation analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple and partial Spearman correlations between speech-graph features and Barratt targets (MOT, COG, MOT_V4, COG_V1)",
        prog="python -m src.analysis.correlation_analysis",
    )
    parser.add_argument(
        "--metrics-dir", default="data/processed/metrics",
        help="Directory with metrics tables",
    )
    parser.add_argument(
        "--metadata", default="data/raw/metadata.xlsx",
        help="Path to metadata Excel file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: auto-generated from task/type/window)",
    )
    parser.add_argument(
        "--task", default="all",
        help="Comma-separated task numbers (e.g. '2,7') or 'all' (default: all)",
    )
    parser.add_argument(
        "--experiment", default="raw", choices=["raw", "zscores", "rawzscore"],
        help="Feature experiment: 'raw', 'zscores', or 'rawzscore' (default: raw)",
    )
    parser.add_argument(
        "--window", default=None,
        help="Specific window tag (e.g. T2W10). When set, output goes to outputs/correlations/Task{N}/{window}_{type}/",
    )
    parser.add_argument(
        "--adj-var", default="School year",
        help="Column name in metadata for partial correlation adjustment (default: School year)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = None
    if args.task.lower() != "all":
        tasks = [int(t.strip()) for t in args.task.split(",")]

    output_dir = args.output
    if output_dir is None:
        task_str = args.task if args.task.lower() != "all" else "all"
        exp_str = args.experiment
        if args.window:
            win = args.window
            task_num = win[1:win.index("W")]
            output_dir = f"outputs/correlations/Task{task_num}/{win}_{exp_str}"
        else:
            output_dir = f"outputs/correlations/task{task_str}_{exp_str}"
        print(f"Auto output: {output_dir}")

    run_correlation_analysis(
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
        output_dir=output_dir,
        tasks=tasks,
        experiment=args.experiment,
        adj_var=args.adj_var,
        window=args.window,
    )


if __name__ == "__main__":
    main()
