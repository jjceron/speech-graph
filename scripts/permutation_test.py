"""Permutation test for linear / Ridge regression models.
Runs the model on real data, then shuffles the target N times to build
a null distribution of R². Optionally saves results to --output / --run-name.

Usage:
    py scripts/permutation_test.py --target COG --type raw --feature-list "edges_T2W10,l3_T6W30,edges_T7W50" --covar "School year" --ridge --n-perm 1000
    py scripts/permutation_test.py --target MOT --type z --feature-list "z_cc_T2W20,z_l2_T2W10" --n-perm 2000
    py scripts/permutation_test.py --target COG --type raw --feature-list "edges_T2W10,l3_T6W30" --output outputs/linear_regression/perm_test --run-name cog_2feat_ols --n-perm 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.linear_regression_mc import (
    parse_feature_id,
    load_top_features,
    load_all_features,
    resolve_feature_ids,
    build_model_matrix,
)
from src.pipeline.speechgraph import load_metadata


ALPHA_GRID = np.logspace(-2, 3, 20)


def run_regression(X, y, use_ridge: bool):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if use_ridge:
        model = RidgeCV(alphas=ALPHA_GRID, scoring="neg_mean_squared_error")
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return r2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permutation test for regression models (R² significance)"
    )
    parser.add_argument("--target", default="COG", choices=["MOT", "COG"],
                        help="Target variable (default: COG)")
    parser.add_argument("--type", default="raw", choices=["raw", "z"],
                        help="Feature type (default: raw)")
    parser.add_argument("--feature-list", required=True,
                        help="Comma-separated feature identifiers")
    parser.add_argument("--covar", default=None,
                        help="Covariate column from metadata (e.g. 'School year')")
    parser.add_argument("--ridge", action="store_true",
                        help="Use RidgeCV instead of LinearRegression")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Number of permutations (default: 1000)")
    parser.add_argument("--output", default=None,
                        help="Output directory to save results (e.g. outputs/linear_regression/perm_test)")
    parser.add_argument("--run-name", default=None,
                        help="Name for the output subfolder (default: auto-generated)")
    parser.add_argument("--metrics-dir", default="data/processed/metrics",
                        help="Metrics directory")
    parser.add_argument("--metadata", default="data/raw/metadata.xlsx",
                        help="Metadata path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    feat_ids = [f.strip() for f in args.feature_list.split(",") if f.strip()]
    covar_cols = [c.strip() for c in args.covar.split(",") if c.strip()] if args.covar else None

    model_label = "RidgeCV" if args.ridge else "OLS"
    pred_label = f"{args.target}, {len(feat_ids) + (len(covar_cols) if covar_cols else 0)} predictors"

    print(f"Permutation Test - {pred_label}")
    print(f"  Model: {model_label}")
    print(f"  Features: {', '.join(feat_ids)}")
    if covar_cols:
        print(f"  Covariates: {', '.join(covar_cols)}")
    print(f"  Permutations: {args.n_perm}")
    print()

    meta = load_metadata(args.metadata)
    all_data = load_all_features(Path(args.metrics_dir), meta, ftype_filter=args.type)
    df = all_data.get(args.type)
    if df is None or df.empty:
        print(f"  No {args.type} data. Exiting.")
        sys.exit(1)

    matrix = build_model_matrix(feat_ids, df, args.target)
    if matrix.empty or len(matrix) < 20:
        print(f"  Only {len(matrix)} valid subjects. Exiting.")
        sys.exit(1)

    y = matrix[args.target]
    cols = [c for c in matrix.columns if c not in (args.target, "file")]

    if covar_cols:
        covar_df = df[["file"] + covar_cols].drop_duplicates(subset="file").set_index("file")
        matrix = matrix.join(covar_df, on="file", how="left")
        cols = cols + covar_cols
        matrix = matrix.dropna()

    X = matrix[cols]
    y = matrix[args.target]

    print(f"  Subjects: {len(X)}")
    print(f"  Predictors: {len(cols)}")
    print()

    real_r2 = run_regression(X, y, use_ridge=args.ridge)

    null_r2s = []
    for i in range(args.n_perm):
        y_shuff = y.sample(frac=1, random_state=args.seed + i).values
        null_r2 = run_regression(X, y_shuff, use_ridge=args.ridge)
        null_r2s.append(null_r2)

    null_r2s = np.array(null_r2s)
    p_value = (np.sum(null_r2s >= real_r2) + 1) / (args.n_perm + 1)
    null_mean = null_r2s.mean()
    null_std = null_r2s.std()

    significant = p_value < 0.05
    print(f"  Real R2     = {real_r2:.4f}")
    print(f"  Null mean   = {null_mean:.4f} +/- {null_std:.4f}")
    print(f"  Null 95% CI = [{np.percentile(null_r2s, 2.5):.4f}, {np.percentile(null_r2s, 97.5):.4f}]")
    print(f"  p-value     = {p_value:.4f} ({int(np.sum(null_r2s >= real_r2))} / {args.n_perm + 1})")
    print(f"  Significant at 0.05: {'YES' if significant else 'NO'}")

    print(f"\n           R2 distribution")
    print(f"  {'-' * 40}")
    n_bars = 40
    bins = np.linspace(null_r2s.min(), null_r2s.max(), 20)
    hist, edges = np.histogram(null_r2s, bins=bins)
    max_count = hist.max()
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar_len = int(h / max_count * n_bars) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {lo:6.3f} | {bar} {h}")
    print(f"         L-- Real R2 = {real_r2:.4f} {'***' if significant else ''}")

    if args.output is not None:
        out_root = Path(args.output)
        out_root.mkdir(parents=True, exist_ok=True)

        model_tag = "ridge" if args.ridge else "ols"
        n_feats = len(feat_ids) + (len(covar_cols) if covar_cols else 0)
        run_tag = args.run_name or f"{args.target.lower()}_{n_feats}feat_{model_tag}"
        run_dir = out_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        figs_dir = run_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        summary = pd.DataFrame([{
            "target": args.target,
            "feature_type": args.type,
            "model": model_label,
            "n_subjects": len(X),
            "n_predictors": len(cols),
            "features": ", ".join(feat_ids),
            "covariates": args.covar or "",
            "n_perm": args.n_perm,
            "real_r2": round(real_r2, 6),
            "null_mean": round(null_mean, 6),
            "null_std": round(null_std, 6),
            "null_ci_2.5": round(np.percentile(null_r2s, 2.5), 6),
            "null_ci_97.5": round(np.percentile(null_r2s, 97.5), 6),
            "p_value": round(p_value, 6),
            "significant": significant,
        }])
        summary.to_csv(run_dir / "summary.csv", index=False)

        null_df = pd.DataFrame({"r2_null": null_r2s})
        null_df.to_csv(run_dir / "null_distribution.csv", index=False)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(null_r2s, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(real_r2, color="crimson", linestyle="--", linewidth=2,
                       label=f"Real R2 = {real_r2:.4f}")
            ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            ax.set_xlabel("R2")
            ax.set_ylabel("Permutations")
            ax.set_title(f"Permutation test — {run_tag}", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(figs_dir / "null_histogram.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Saved: {run_dir / 'summary.csv'}")
            print(f"  Saved: {run_dir / 'null_distribution.csv'}")
            print(f"  Saved: {figs_dir / 'null_histogram.png'}")
        except Exception as e:
            print(f"  Warning: figure save failed: {e}")

        print(f"\nPermutation test results saved to: {run_dir}")


if __name__ == "__main__":
    main()
