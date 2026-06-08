"""Check collinearity among candidate features for linear regression.

Loads features from --feature-list or --corr-dir + --target + --top-k,
builds the subject-level matrix, and prints:
  - sample size
  - pairwise Spearman correlation matrix
  - VIF (Variance Inflation Factor)
  - interpretation

Usage:
    py scripts/check_collinearity.py --type raw --feature-list "lsc_T7W30,l2_T2W30"
    py scripts/check_collinearity.py --type raw --corr-dir outputs/correlations/Task7/raw_task7_schoolyear --target MOT --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.correlation_analysis import find_means_tables, load_feature_table
from src.pipeline.speechgraph import load_metadata
from src.analysis.linear_regression_mc import (
    parse_feature_id,
    load_top_features,
    load_all_features,
    resolve_feature_ids,
)
from src.analysis.linear_regression_rcv import METRICS_OF_INTEREST


def compute_vif(df: pd.DataFrame) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    vifs = {}
    cols = df.columns
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        lr = LinearRegression()
        lr.fit(df[others], df[col])
        r2 = lr.score(df[others], df[col])
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(vifs)


def resolve_mixed_feature_ids(
    feat_ids: list[str],
    raw_df: pd.DataFrame,
    z_df: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve feature IDs from mixed raw/z sources.

    Feature IDs starting with 'z_' are looked up in the z DataFrame;
    others are looked up in the raw DataFrame.
    """
    series_list = []
    for fid in feat_ids:
        feature, task, window = parse_feature_id(fid)
        is_z = fid.startswith("z_")
        source = z_df if is_z else raw_df
        if source.empty:
            print(f"  Warning: no {('z' if is_z else 'raw')} data loaded for {fid}")
            continue
        mask = (source["_task"] == task) & (source["_window"] == window)
        subset = source[mask]
        if subset.empty:
            print(f"  Warning: no data found for {fid} (task={task}, window={window})")
            continue
        if feature not in subset.columns:
            print(f"  Warning: column '{feature}' not found in "
                  f"{'z' if is_z else 'raw'} tables for {fid}")
            continue
        ser = subset.set_index("file")[feature].rename(fid)
        series_list.append(ser)

    if not series_list:
        return pd.DataFrame()
    result = pd.concat(series_list, axis=1)
    result.index.name = "file"
    result = result.dropna()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check collinearity among candidate features",
    )
    parser.add_argument("--type", default="raw", choices=["raw", "z"],
                        help="Feature type for per-window or --corr-dir mode (default: raw)")
    parser.add_argument("--task", type=int, default=None,
                        help="Task number for per-window collinearity check")
    parser.add_argument("--window", type=int, default=None,
                        help="Window size for per-window collinearity check")
    parser.add_argument("--all-metrics", action="store_true", default=False,
                        help="Use all METRICS_OF_INTEREST (default); otherwise uses them all anyway")
    parser.add_argument("--feature-list", default=None,
                        help="Comma-separated feature identifiers")
    parser.add_argument("--corr-dir", default=None,
                        help="Directory with top feature CSVs from correlation_analysis.py")
    parser.add_argument("--target", default="MOT", choices=["MOT", "COG"],
                        help="Target for top-k feature loading (default: MOT)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top features (default: 5)")
    parser.add_argument("--metrics-dir", default="data/processed/metrics",
                        help="Directory with metrics tables")
    parser.add_argument("--metadata", default="data/raw/metadata.xlsx",
                        help="Path to metadata Excel file")
    args = parser.parse_args()

    # --- Per-window mode ---
    if args.task is not None and args.window is not None:
        full_window = f"T{args.task}W{args.window}"
        print(f"Loading means table for {full_window} ({args.type})...")
        tables = find_means_tables(Path(args.metrics_dir), tasks=[args.task])
        tables = [
            t for t in tables
            if t["tag"] == full_window and t["type"] == args.type
        ]
        if not tables:
            print(f"  No means table found for {full_window} type={args.type}. Exiting.")
            sys.exit(1)
        feats = load_feature_table(tables[0]["path"])
        feature_cols = [c for c in feats.columns if c != "file"]
        if args.type == "raw":
            feature_cols = [c for c in feature_cols if c in METRICS_OF_INTEREST]
        else:
            feature_cols = [
                c for c in feature_cols
                if c.startswith("z_") and c.replace("z_", "") in METRICS_OF_INTEREST
            ]
        feat_ids = [f"{c}_{full_window}" for c in feature_cols]
        from src.analysis.linear_regression_rcv import filter_cc_features
        feat_ids = filter_cc_features(feat_ids)
        X = feats[["file"] + feature_cols].set_index("file")
        X.columns = [f"{c}_{full_window}" for c in X.columns]
        X = X[feat_ids]
        X = X.dropna()
        desc = f"{full_window} {args.type} ({X.shape[1]} features)"
    else:
        print(f"Loading metadata...")
        meta = load_metadata(args.metadata)

        print(f"Loading metric tables...")
        all_data = load_all_features(Path(args.metrics_dir), meta, ftype_filter="all")
        raw_df = all_data.get("raw", pd.DataFrame())
        z_df = all_data.get("z", pd.DataFrame())

        if args.feature_list is not None:
            feat_ids = [f.strip() for f in args.feature_list.split(",") if f.strip()]
            desc = f"manual ({len(feat_ids)} features)"
            X = resolve_mixed_feature_ids(feat_ids, raw_df, z_df)
        elif args.corr_dir is not None:
            corr_path = Path(args.corr_dir)
            feat_ids = load_top_features(corr_path, args.target, args.type, args.top_k)
            if not feat_ids:
                print(f"  No features loaded from {corr_path}. Exiting.")
                sys.exit(1)
            desc = f"top-{len(feat_ids)} from {corr_path.name}"
            df_all = all_data.get(args.type)
            if df_all is None or df_all.empty:
                print(f"  No {args.type} data loaded. Exiting.")
                sys.exit(1)
            X = resolve_feature_ids(feat_ids, df_all)
        else:
            print("  Provide --task/--window, --feature-list, or --corr-dir.")
            sys.exit(1)

    if X.empty or len(X) < 10:
        print(f"  Only {len(X)} valid subjects. Exiting.")
        sys.exit(1)

    n_subjects = len(X)
    n_features = X.shape[1]

    print(f"\n{'='*60}")
    print(f"Collinearity Check: {desc}")
    print(f"{'='*60}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Features ({n_features}): {', '.join(X.columns)}")
    print()

    corr = X.corr(method="spearman")
    print("Spearman correlation matrix:")
    print(corr.round(4).to_string())
    print()

    vif = compute_vif(X)
    print("Variance Inflation Factor (VIF):")
    for feat, v in vif.items():
        flag = ""
        if v >= 10:
            flag = "  *** HIGH collinearity"
        elif v >= 5:
            flag = "  ** MODERATE collinearity"
        print(f"  {feat:25s}  {v:.2f}{flag}")
    print()

    max_vif = vif.max()
    if max_vif >= 10:
        print(f"WARNING: max VIF = {max_vif:.2f} (>= 10). "
              "Consider removing or regularising (Ridge).")
    elif max_vif >= 5:
        print(f"NOTE: max VIF = {max_vif:.2f} (>= 5). "
              "Moderate collinearity — monitor coefficient stability.")
    else:
        print(f"No collinearity concerns (all VIF < 5).")

    high_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                high_pairs.append((corr.columns[i], corr.columns[j], r))
    if high_pairs:
        print(f"\nHigh pairwise correlations (|r| > 0.7):")
        for f1, f2, r in high_pairs:
            print(f"  {f1}  <->  {f2}  (r = {r:.3f})")


if __name__ == "__main__":
    main()
