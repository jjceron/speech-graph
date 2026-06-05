"""Monte Carlo cross-validated linear regression predicting MOT and COG
from top speech-graph features identified by correlation_analysis.py.

Usage:
    py -m src.analysis.linear_regression_mc --top-k 5 --n-iter 400 --output outputs/06_regression
    py -m src.analysis.linear_regression_mc --top-k 10 --n-iter 200 --corr-dir outputs/06_correlations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.correlation_analysis import find_means_tables, load_feature_table
from src.visualization.lr_metrics import plot_all_regression_figures
from src.pipeline.speechgraph import load_metadata


METRICS_OF_INTEREST = [
    "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]


def load_top_features(corr_dir: Path, target: str, ftype: str, top_k: int) -> list[str]:
    fname = f"top_partial_{target}_{ftype}.csv"
    fpath = corr_dir / fname
    if not fpath.exists():
        print(f"  Warning: {fpath} not found. Skipping {target} {ftype}.")
        return []
    top = pd.read_csv(fpath)
    if "feature" not in top.columns:
        print(f"  Warning: 'feature' column not found in {fpath}.")
        return []
    return top["feature"].head(top_k).tolist()


def load_all_features(
    metrics_dir: Path,
    meta: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables = find_means_tables(metrics_dir)
    raw_dfs, z_dfs = [], []

    for entry in tables:
        task = entry["task"]
        tag = entry["tag"]
        ftype = entry["type"]
        path = entry["path"]

        feats = load_feature_table(path)
        merged = feats.merge(meta, left_on="file", right_on="Cod", how="inner")
        merged["_task"] = task
        merged["_window"] = tag

        if ftype == "raw":
            raw_dfs.append(merged)
        else:
            z_dfs.append(merged)

    raw_all = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()
    z_all = pd.concat(z_dfs, ignore_index=True) if z_dfs else pd.DataFrame()
    return {"raw": raw_all, "z": z_all}


def run_montecarlo_regression(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 400,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    results = []
    all_y_true = []
    all_y_pred = []

    for i in range(n_iter):
        seed = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rho, p = spearmanr(y_test, y_pred)

        results.append({"r2": r2, "rmse": rmse, "rho": rho, "p_rho": p})
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    df = pd.DataFrame(results)

    return {
        "r2_mean": df["r2"].mean(),
        "r2_std": df["r2"].std(),
        "rmse_mean": df["rmse"].mean(),
        "rmse_std": df["rmse"].std(),
        "rho_mean": df["rho"].mean(),
        "rho_std": df["rho"].std(),
        "r2_below_zero": (df["r2"] < 0).mean(),
        "n_iter": n_iter,
        "test_size": test_size,
        "all_results": df,
        "y_true_all": all_y_true,
        "y_pred_all": all_y_pred,
    }


def run_analysis(
    corr_dir: str | Path = "outputs/06_correlations",
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/06_regression",
    top_k: int = 5,
    n_iter: int = 400,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_dir = Path(corr_dir)

    print("Loading metadata...")
    meta = load_metadata(metadata_path)

    print("Loading metric tables...")
    all_data = load_all_features(Path(metrics_dir), meta)

    targets = ["MOT", "COG"]
    ftypes = ["raw", "z"]

    summary_rows = []

    for target in targets:
        for ftype in ftypes:
            print(f"\n{'='*60}")
            print(f"Target: {target} | Features: {ftype}")
            print(f"{'='*60}")

            feat_list = load_top_features(corr_dir, target, ftype, top_k)
            if not feat_list:
                print("  No features loaded. Skipping.")
                continue

            df = all_data[ftype]
            if df.empty:
                print("  No data loaded. Skipping.")
                continue

            missing = [f for f in feat_list if f not in df.columns]
            if missing:
                print(f"  Missing features: {missing}")
                feat_list = [f for f in feat_list if f in df.columns]
            if not feat_list:
                print("  No valid features. Skipping.")
                continue

            valid = df[feat_list].notna().all(axis=1) & df[target].notna()
            sub = df[valid]
            if len(sub) < 20:
                print(f"  Only {len(sub)} valid subjects (< 20). Skipping.")
                continue

            X = sub[feat_list].values.astype(float)
            y = sub[target].values.astype(float)

            print(f"  Subjects: {len(sub)}")
            print(f"  Features ({len(feat_list)}): {feat_list}")

            result = run_montecarlo_regression(
                X, y, n_iter=n_iter, test_size=test_size, random_state=seed
            )

            print(f"  R²  = {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
            print(f"  RMSE= {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
            print(f"  ρ   = {result['rho_mean']:.4f} ± {result['rho_std']:.4f}")
            print(f"  R² < 0: {result['r2_below_zero']:.1%} of iterations")

            summary_rows.append({
                "target": target,
                "feature_type": ftype,
                "n_subjects": len(sub),
                "n_features": len(feat_list),
                "features": ", ".join(feat_list),
                "n_iter": n_iter,
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "rmse_mean": result["rmse_mean"],
                "rmse_std": result["rmse_std"],
                "rho_mean": result["rho_mean"],
                "rho_std": result["rho_std"],
                "frac_r2_below_zero": result["r2_below_zero"],
            })

            tag = f"{target}_{ftype}"

            try:
                import matplotlib
                plot_all_regression_figures(result, output_dir, tag)
            except Exception as e:
                print(f"  Warning: regression plots failed: {e}")

            iter_path = output_dir / f"mc_iterations_{tag}.csv"
            result["all_results"].to_csv(iter_path, index=False)
            print(f"  Saved: {iter_path.name}")

            pred_df = pd.DataFrame({
                "y_true": result["y_true_all"],
                "y_pred": result["y_pred_all"],
            })
            pred_path = output_dir / f"predictions_{tag}.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"  Saved: {pred_path.name}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / "regression_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path.name}")
        print(summary.round(4).to_string(index=False))

    print("\nDone. Regression analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo cross-validated linear regression predicting MOT/COG from top speech-graph features",
        prog="python -m src.analysis.linear_regression_mc",
    )
    parser.add_argument(
        "--corr-dir", default="outputs/06_correlations",
        help="Directory with top feature CSVs from correlation_analysis.py",
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
        "--output", default="outputs/06_regression",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top features to use (default: 5)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=400,
        help="Number of Monte Carlo iterations (default: 400)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test set proportion (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        corr_dir=args.corr_dir,
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
        output_dir=args.output,
        top_k=args.top_k,
        n_iter=args.n_iter,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
