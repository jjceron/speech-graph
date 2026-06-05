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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check collinearity among candidate features",
    )
    parser.add_argument("--type", default="raw", choices=["raw", "z"],
                        help="Feature type (default: raw)")
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

    print(f"Loading metadata...")
    meta = load_metadata(args.metadata)

    print(f"Loading {args.type} metric tables...")
    all_data = load_all_features(Path(args.metrics_dir), meta, ftype_filter=args.type)
    df_all = all_data.get(args.type)
    if df_all is None or df_all.empty:
        print(f"  No {args.type} data loaded. Exiting.")
        sys.exit(1)

    if args.feature_list is not None:
        feat_ids = [f.strip() for f in args.feature_list.split(",") if f.strip()]
        desc = f"manual ({len(feat_ids)} features)"
    elif args.corr_dir is not None:
        corr_path = Path(args.corr_dir)
        feat_ids = load_top_features(corr_path, args.target, args.type, args.top_k)
        if not feat_ids:
            print(f"  No features loaded from {corr_path}. Exiting.")
            sys.exit(1)
        desc = f"top-{len(feat_ids)} from {corr_path.name}"
    else:
        print("  Provide --feature-list or --corr-dir.")
        sys.exit(1)

    X = resolve_feature_ids(feat_ids, df_all)
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
