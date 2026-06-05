"""Monte Carlo cross-validated linear regression predicting MOT or COG
from speech-graph features. Supports automatic top-K from a single task's
correlation output, or manual feature lists spanning multiple tasks and
windows with user-chosen feature identifiers.

Usage:
    py -m src.analysis.linear_regression_mc --target MOT --type raw --top-k 5 --n-iter 400
    py -m src.analysis.linear_regression_mc --target MOT --type raw --feature-list "lsc_T7W30,l2_T2W30" --run-name myrun --n-iter 400
    py -m src.analysis.linear_regression_mc --target all --type all --top-k 5 --n-iter 400
"""

from __future__ import annotations

import argparse
import re
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


def parse_feature_id(fid: str) -> tuple[str, int, str]:
    """Parse a feature identifier like 'lsc_T7W30'.
    
    Returns (feature_name, task_number, window_tag).
    window_tag is the full window string including the T prefix, e.g. 'T7W30'.
    """
    m = re.match(r"^(.+)_T(\d+)W(.+)$", fid)
    if not m:
        raise ValueError(
            f"Cannot parse feature identifier: '{fid}'. "
            f"Expected format like 'lsc_T7W30' (feature_T<task>W<window>)."
        )
    feature = m.group(1)
    task = int(m.group(2))
    window = f"T{m.group(2)}W{m.group(3)}"
    return feature, task, window


def load_top_features(corr_dir: Path, target: str, ftype: str, top_k: int) -> list[str]:
    fname = f"top_partial_{target}_{ftype}.csv"
    fpath = corr_dir / fname
    if not fpath.exists():
        print(f"  Warning: {fpath} not found.")
        return []
    top = pd.read_csv(fpath)
    required = {"feature", "window_tag"}
    if not required.issubset(top.columns):
        print(f"  Warning: required columns {required} not found in {fpath}.")
        return []
    top["feat_id"] = top["feature"] + "_" + top["window_tag"]
    return top["feat_id"].head(top_k).tolist()


def load_all_features(
    metrics_dir: Path,
    meta: pd.DataFrame,
    ftype_filter: str = "all",
) -> dict[str, pd.DataFrame]:
    tables = find_means_tables(metrics_dir)
    if ftype_filter != "all":
        tables = [t for t in tables if t["type"] == ftype_filter]

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


def resolve_feature_ids(feat_ids: list[str], all_data: pd.DataFrame) -> pd.DataFrame:
    """Resolve feature identifiers like 'lsc_T7W30' into a subject-level matrix.
    
    Returns a DataFrame indexed by subject ID (file) with one column per feature.
    Rows with any missing value are dropped.
    """
    series_list = []
    for fid in feat_ids:
        feature, task, window = parse_feature_id(fid)
        mask = (all_data["_task"] == task) & (all_data["_window"] == window)
        subset = all_data[mask]
        if subset.empty:
            print(f"  Warning: no data found for {fid} (task={task}, window={window})")
            continue
        if feature not in subset.columns:
            print(f"  Warning: column '{feature}' not found for {fid}")
            continue
        ser = subset.set_index("file")[feature].rename(fid)
        series_list.append(ser)

    if not series_list:
        return pd.DataFrame()

    result = pd.concat(series_list, axis=1)
    result.index.name = "file"
    result = result.dropna()
    return result


def build_model_matrix(
    feat_ids: list[str],
    all_data: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Build a subject-level matrix with features and the target column.
    
    Returns a DataFrame with columns: file, <feat_id_1>, ..., <feat_id_n>, target.
    """
    X = resolve_feature_ids(feat_ids, all_data)
    if X.empty:
        return pd.DataFrame()

    target_ser = all_data.groupby("file")[target].first()
    X[target] = target_ser
    result = X.dropna().reset_index()
    return result


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
    all_coefs = []
    feature_names = list(X.columns)

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
        rho, p_rho = spearmanr(y_test, y_pred)

        results.append({"r2": r2, "rmse": rmse, "rho": rho, "p_rho": p_rho})
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_coefs.append(model.coef_)

    df = pd.DataFrame(results)
    coefs = np.array(all_coefs)

    n_nonzero = np.sum(np.abs(coefs) > 1e-12, axis=0)
    coef_summary = pd.DataFrame({
        "feature": feature_names,
        "mean_coef": coefs.mean(axis=0),
        "std_coef": coefs.std(axis=0),
        "frac_nonzero": n_nonzero / n_iter,
    }).sort_values("mean_coef", key=abs, ascending=False)

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
        "coef_summary": coef_summary,
    }


def run_analysis(
    target: str = "all",
    ftype: str = "all",
    feature_list: str | None = None,
    run_name: str | None = None,
    corr_dir: str | Path = "outputs/correlations",
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/linear_regression",
    top_k: int = 5,
    n_iter: int = 400,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    meta = load_metadata(metadata_path)

    print("Loading metric tables...")
    type_filter = ftype if ftype != "all" else "all"
    all_data = load_all_features(Path(metrics_dir), meta, ftype_filter=type_filter)

    targets = ["MOT", "COG"] if target == "all" else [target]
    ftypes = ["raw", "z"] if ftype == "all" else [ftype]

    summary_rows = []

    for tgt in targets:
        for ft in ftypes:
            print(f"\n{'='*60}")
            print(f"Target: {tgt} | Type: {ft}")
            print(f"{'='*60}")

            df = all_data.get(ft)
            if df is None or df.empty:
                print(f"  No {ft} data available. Skipping.")
                continue

            if feature_list is not None:
                raw_feats = [f.strip() for f in feature_list.split(",") if f.strip()]
                desc = run_name or f"manual{len(raw_feats)}"
            else:
                corr_dir_path = Path(corr_dir)
                raw_feats = load_top_features(corr_dir_path, tgt, ft, top_k)
                desc = run_name or f"top{top_k}"

            if not raw_feats:
                print("  No features loaded. Skipping.")
                continue

            matrix = build_model_matrix(raw_feats, df, tgt)
            if matrix.empty or len(matrix) < 20:
                print(f"  Only {len(matrix)} valid subjects (< 20). Skipping.")
                continue

            # 'file' column comes from build_model_matrix
            y = matrix[tgt]
            X = matrix.drop(columns=[tgt, "file"])

            print(f"  Subjects: {len(matrix)}")
            print(f"  Features ({X.shape[1]}): {list(X.columns)}")

            result = run_montecarlo_regression(
                X, y, n_iter=n_iter, test_size=test_size, random_state=seed
            )

            print(f"  R2   = {result['r2_mean']:.4f} +/- {result['r2_std']:.4f}")
            print(f"  RMSE = {result['rmse_mean']:.4f} +/- {result['rmse_std']:.4f}")
            print(f"  rho  = {result['rho_mean']:.4f} +/- {result['rho_std']:.4f}")
            print(f"  R2 < 0: {result['r2_below_zero']:.1%} of iterations")
            print(f"  Top coefficients:")
            for _, row in result["coef_summary"].head(5).iterrows():
                print(f"    {row['feature']}: coef={row['mean_coef']:.4f} +/- {row['std_coef']:.4f} "
                      f"(nonzero in {row['frac_nonzero']:.0%})")

            run_dir = output_root / f"{tgt}_{desc}_{ft}"
            run_dir.mkdir(parents=True, exist_ok=True)
            figs_dir = run_dir / "figures"
            figs_dir.mkdir(parents=True, exist_ok=True)

            tag = f"{tgt}_{ft}"

            summary_rows.append({
                "target": tgt,
                "feature_type": ft,
                "descriptor": desc,
                "n_subjects": len(matrix),
                "n_features": X.shape[1],
                "features": ", ".join(X.columns),
                "n_iter": n_iter,
                "test_size": test_size,
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "rmse_mean": result["rmse_mean"],
                "rmse_std": result["rmse_std"],
                "rho_mean": result["rho_mean"],
                "rho_std": result["rho_std"],
                "frac_r2_below_zero": result["r2_below_zero"],
            })

            summary_path = run_dir / "summary.csv"
            pd.DataFrame([summary_rows[-1]]).to_csv(summary_path, index=False)
            print(f"  Saved: {summary_path.name}")

            iter_path = run_dir / "iterations.csv"
            result["all_results"].to_csv(iter_path, index=False)
            print(f"  Saved: {iter_path.name}")

            pred_df = pd.DataFrame({
                "y_true": result["y_true_all"],
                "y_pred": result["y_pred_all"],
            })
            pred_path = run_dir / "predictions.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"  Saved: {pred_path.name}")

            coef_path = run_dir / "coefficients.csv"
            result["coef_summary"].to_csv(coef_path, index=False)
            print(f"  Saved: {coef_path.name}")

            try:
                import matplotlib
                plot_all_regression_figures(result, figs_dir, tag)
            except Exception as e:
                print(f"  Warning: regression plots failed: {e}")

    if len(summary_rows) > 1:
        summary_all = pd.DataFrame(summary_rows)
        summary_all_path = output_root / "multi_run_summary.csv"
        summary_all.to_csv(summary_all_path, index=False)
        print(f"\nMulti-run summary saved: {summary_all_path.name}")
        print(summary_all.round(4).to_string(index=False))

    print("\nDone. Regression analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo cross-validated linear regression predicting "
            "MOT or COG from speech-graph features."
        ),
        prog="python -m src.analysis.linear_regression_mc",
    )
    parser.add_argument(
        "--target", default="all", choices=["MOT", "COG", "all"],
        help="Target variable (default: all)",
    )
    parser.add_argument(
        "--type", default="all", choices=["raw", "z", "all"],
        help="Feature type (default: all)",
    )
    parser.add_argument(
        "--feature-list", default=None,
        help=(
            "Comma-separated feature identifiers for manual feature selection, "
            "e.g. 'lsc_T7W30,cc_T7W40,l2_T2W30'. When provided, --corr-dir "
            "and --top-k are ignored."
        ),
    )
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Custom name for the output subfolder. When using --feature-list, "
            "the folder becomes {target}_{name}_{type}. If omitted, defaults to "
            "'manual{N}' where N is the number of features."
        ),
    )
    parser.add_argument(
        "--corr-dir", default="outputs/correlations",
        help=(
            "Directory with top feature CSVs from correlation_analysis.py. "
            "Only used when --feature-list is not provided (default: "
            "outputs/correlations)"
        ),
    )
    parser.add_argument(
        "--metrics-dir", default="data/processed/metrics",
        help="Directory with metrics tables (default: data/processed/metrics)",
    )
    parser.add_argument(
        "--metadata", default="data/raw/metadata.xlsx",
        help="Path to metadata Excel file (default: data/raw/metadata.xlsx)",
    )
    parser.add_argument(
        "--output", default="outputs/linear_regression",
        help="Root output directory (default: outputs/linear_regression)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help=(
            "Number of top features to use from --corr-dir. "
            "Only used when --feature-list is not provided (default: 5)"
        ),
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
        target=args.target,
        ftype=args.type,
        feature_list=args.feature_list,
        run_name=args.run_name,
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
