"""Correlation analysis: simple and partial Spearman correlations
between speech-graph features (raw and/or z-scores) and Barratt MOT/COG,
controlling for School year.

Usage:
    py -m src.analysis.correlation_analysis --output outputs/06_correlations
    py -m src.analysis.correlation_analysis --task 2 --type raw --output outputs/06_correlations/T2_raw
    py -m src.analysis.correlation_analysis --task 2,7 --type z --output outputs/06_correlations/T27_z
    py -m src.analysis.correlation_analysis --task 2 --type raw --adj-var "School year" --output outputs/correlations/T2_raw
    py -m src.analysis.correlation_analysis --task 7 --type z --adj-var "Age" --output outputs/correlations/T7_z_age
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

from src.pipeline.speechgraph import load_metadata
from src.visualization.corr_metrics import plot_correlation_heatmaps


METRICS_OF_INTEREST = [
    "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]


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


def run_correlation_analysis(
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/06_correlations",
    tasks: list[int] | None = None,
    ftype_filter: str = "all",
    adj_var: str = "School year",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(metadata_path)

    tables = find_means_tables(Path(metrics_dir), tasks=tasks)
    if not tables:
        print("No means tables found.")
        return

    if ftype_filter != "all":
        tables = [t for t in tables if t["type"] == ftype_filter]
        if not tables:
            print(f"No tables found for type '{ftype_filter}'.")
            return

    task_str = f"tasks {tasks}" if tasks else "all tasks"
    print(f"Processing: {task_str}, type='{ftype_filter}'")
    print(f"Found {len(tables)} means tables ({sum(1 for t in tables if t['type']=='raw')} raw, {sum(1 for t in tables if t['type']=='z')} z)")

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

        feature_cols = [c for c in feats.columns if c != "file"]
        if ftype == "raw":
            feature_cols = [c for c in feature_cols if c in METRICS_OF_INTEREST]
        else:
            feature_cols = [c for c in feature_cols if c.startswith("z_") and c.replace("z_", "") in METRICS_OF_INTEREST]

        for col in feature_cols:
            valid = merged[col].notna() & merged["MOT"].notna() & merged["COG"].notna() & merged[adj_var].notna()
            sub = merged[valid]
            if len(sub) < 10:
                continue

            x = sub[col].values.astype(float)
            if np.nanvar(x) < 1e-10:
                continue
            mot = sub["MOT"].values.astype(float)
            cog = sub["COG"].values.astype(float)
            if not pd.api.types.is_numeric_dtype(sub[adj_var]):
                cats = sorted(sub[adj_var].unique())
                mapping = {v: i for i, v in enumerate(cats)}
                cov_vals = sub[adj_var].map(mapping).values.astype(float)
            else:
                cov_vals = sub[adj_var].values.astype(float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_mot, p_mot = spearmanr(x, mot)
                r_cog, p_cog = spearmanr(x, cog)

            try:
                import pingouin as pg
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    partial_df = pd.DataFrame({"feat": x, "target": mot, "cov": cov_vals})
                    pcorr_mot = pg.partial_corr(
                        data=partial_df, x="feat", y="target", covar="cov",
                        method="spearman"
                    )
                r_partial_mot = pcorr_mot["r"].values[0]
                pval_col = [c for c in pcorr_mot.columns if c.startswith("p")]
                if not pval_col:
                    raise KeyError(f"No p-value column found in {list(pcorr_mot.columns)}")
                p_partial_mot = pcorr_mot[pval_col[0]].values[0]
            except Exception:
                r_partial_mot = np.nan
                p_partial_mot = np.nan
                traceback.print_exc()

            try:
                import pingouin as pg
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    partial_df = pd.DataFrame({"feat": x, "target": cog, "cov": cov_vals})
                    pcorr_cog = pg.partial_corr(
                        data=partial_df, x="feat", y="target", covar="cov",
                        method="spearman"
                    )
                r_partial_cog = pcorr_cog["r"].values[0]
                pval_col = [c for c in pcorr_cog.columns if c.startswith("p")]
                if not pval_col:
                    raise KeyError(f"No p-value column found in {list(pcorr_cog.columns)}")
                p_partial_cog = pcorr_cog[pval_col[0]].values[0]
            except Exception:
                r_partial_cog = np.nan
                p_partial_cog = np.nan
                traceback.print_exc()

            all_rows.append({
                "feature": col,
                "type": ftype,
                "task": task,
                "window_tag": tag,
                "n": len(sub),
                "r_MOT": r_mot,
                "p_MOT": p_mot,
                "r_partial_MOT": r_partial_mot,
                "p_partial_MOT": p_partial_mot,
                "r_COG": r_cog,
                "p_COG": p_cog,
                "r_partial_COG": r_partial_cog,
                "p_partial_COG": p_partial_cog,
            })

    if not all_rows:
        print("No correlations computed.")
        return

    results = pd.DataFrame(all_rows)

    results["abs_r_MOT"] = results["r_MOT"].abs()
    results["abs_r_partial_MOT"] = results["r_partial_MOT"].abs()
    results["abs_r_COG"] = results["r_COG"].abs()
    results["abs_r_partial_COG"] = results["r_partial_COG"].abs()

    ftypes_present = results["type"].unique()
    for ftype in ftypes_present:
        subset = results[results["type"] == ftype].copy()

        for target, col_r in [("MOT", "abs_r_MOT"), ("COG", "abs_r_COG")]:
            top = subset.sort_values(col_r, ascending=False).head(30)
            clean = top.drop(columns=[c for c in top.columns if c.startswith("abs_")])
            out_path = output_dir / f"top_simple_{target}_{ftype}.csv"
            clean.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

        for target, col_r in [("MOT", "abs_r_partial_MOT"), ("COG", "abs_r_partial_COG")]:
            top = subset.sort_values(col_r, ascending=False).head(30)
            clean = top.drop(columns=[c for c in top.columns if c.startswith("abs_")])
            out_path = output_dir / f"top_partial_{target}_{ftype}.csv"
            clean.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

    full_path = output_dir / "all_correlations.csv"
    results.to_csv(full_path, index=False)
    print(f"  Saved: {full_path.name} ({len(results)} rows)")

    print()
    summary = results.groupby("type").agg(
        n_features=("feature", "count"),
        mean_abs_r_MOT=("r_MOT", lambda x: x.abs().mean()),
        mean_abs_partial_MOT=("r_partial_MOT", lambda x: x.dropna().abs().mean()),
        mean_abs_r_COG=("r_COG", lambda x: x.abs().mean()),
        mean_abs_partial_COG=("r_partial_COG", lambda x: x.dropna().abs().mean()),
    ).round(4)
    print(summary.to_string())
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path.name}")

    try:
        import matplotlib
        plot_correlation_heatmaps(results, output_dir, adj_var=adj_var)
    except Exception as e:
        print(f"  Warning: heatmap plotting failed: {e}")

    print("\nDone. Correlation analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple and partial Spearman correlations between speech-graph features and Barratt MOT/COG",
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
        "--output", default="outputs/06_correlations",
        help="Output directory for results",
    )
    parser.add_argument(
        "--task", default="all",
        help="Comma-separated task numbers to process (e.g. '2,7') or 'all' (default: all)",
    )
    parser.add_argument(
        "--type", default="all", choices=["raw", "z", "all"],
        help="Feature type: 'raw', 'z', or 'all' (default: all)",
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
    run_correlation_analysis(
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
        output_dir=args.output,
        tasks=tasks,
        ftype_filter=args.type,
        adj_var=args.adj_var,
    )


if __name__ == "__main__":
    main()
