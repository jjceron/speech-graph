"""Monte Carlo cross-validated Ridge regression predicting MOT or COG
from speech-graph features. Supports automatic top-K, manual feature lists,
and optional covariates. Uses RidgeCV for automatic alpha selection
within each Monte Carlo split.

Usage:
    py -m src.analysis.linear_regression_rcv --target MOT --type raw --feature-list "lsc_T7W30,l2_T2W30" --n-iter 400
    py -m src.analysis.linear_regression_rcv --target COG --type raw --feature-list "edges_T2W10,l3_T6W30" --covar "School year" --n-iter 400
    py -m src.analysis.linear_regression_rcv --target MOT --type raw --corr-dir outputs/correlations/Task7/raw_task7_schoolyear --top-k 10 --n-iter 400

Output root (default): outputs/linear_regression/ridgecv_mc/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
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

ALPHA_GRID = np.logspace(-2, 3, 20)


def parse_feature_id(fid: str) -> tuple[str, int, str]:
    m = re.match(r"^(.+)_T(\d+)W(.+)$", fid)
    if not m:
        raise ValueError(
            f"Cannot parse feature identifier: '{fid}'. "
            f"Expected format like 'lsc_T7W30'."
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


def build_model_matrix(
    feat_ids: list[str],
    all_data: pd.DataFrame | dict[str, pd.DataFrame],
    target: str,
    covar_cols: list[str] | None = None,
) -> pd.DataFrame:
    if isinstance(all_data, dict):
        # Mixed raw+z mode: resolve each feature against its type
        raw_df = all_data.get("raw", pd.DataFrame())
        z_df = all_data.get("z", pd.DataFrame())
        X = resolve_mixed_feature_ids(feat_ids, raw_df, z_df)
        source_for_target = z_df if not z_df.empty else raw_df
    else:
        # Single-type mode
        X = resolve_feature_ids(feat_ids, all_data)
        source_for_target = all_data

    if X.empty:
        return pd.DataFrame()

    target_ser = source_for_target.groupby("file")[target].first()
    X[target] = target_ser

    if covar_cols:
        covar_df = source_for_target[["file"] + covar_cols].drop_duplicates(subset="file")
        covar_df = covar_df.set_index("file")
        X = X.join(covar_df, how="left")

    result = X.dropna().reset_index()
    return result


def run_montecarlo_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 400,
    test_size: float = 0.2,
    random_state: int = 42,
    alphas: np.ndarray | None = None,
) -> dict:
    if alphas is None:
        alphas = ALPHA_GRID

    results = []
    all_y_true = []
    all_y_pred = []
    all_coefs = []
    all_alphas = []
    feature_names = list(X.columns)

    for i in range(n_iter):
        seed = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        model = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rho, p_rho = spearmanr(y_test, y_pred)

        results.append({"r2": r2, "rmse": rmse, "rho": rho, "p_rho": p_rho})
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_coefs.append(model.coef_)
        all_alphas.append(model.alpha_)

    df_results = pd.DataFrame(results)
    coefs = np.array(all_coefs)

    n_nonzero = np.sum(np.abs(coefs) > 1e-12, axis=0)
    coef_summary = pd.DataFrame({
        "feature": feature_names,
        "mean_coef": coefs.mean(axis=0),
        "std_coef": coefs.std(axis=0),
        "frac_nonzero": n_nonzero / n_iter,
    }).sort_values("mean_coef", key=abs, ascending=False)

    alpha_summary = pd.DataFrame({
        "alpha_mean": np.mean(all_alphas),
        "alpha_std": np.std(all_alphas),
        "alpha_selected_counts": pd.Series(all_alphas).value_counts().to_dict(),
    })

    return {
        "r2_mean": df_results["r2"].mean(),
        "r2_std": df_results["r2"].std(),
        "rmse_mean": df_results["rmse"].mean(),
        "rmse_std": df_results["rmse"].std(),
        "rho_mean": df_results["rho"].mean(),
        "rho_std": df_results["rho"].std(),
        "r2_below_zero": (df_results["r2"] < 0).mean(),
        "n_iter": n_iter,
        "test_size": test_size,
        "all_results": df_results,
        "y_true_all": all_y_true,
        "y_pred_all": all_y_pred,
        "coef_summary": coef_summary,
        "alpha_summary": alpha_summary,
        "all_alphas": all_alphas,
    }


def run_analysis(
    target: str = "all",
    ftype: str = "all",
    feature_list: str | None = None,
    run_name: str | None = None,
    covar: str | None = None,
    corr_dir: str | Path = "outputs/correlations",
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/linear_regression",
    top_k: int = 5,
    n_iter: int = 400,
    test_size: float = 0.2,
    seed: int = 42,
    alphas: np.ndarray | None = None,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if alphas is None:
        alphas = ALPHA_GRID
    covar_cols = [c.strip() for c in covar.split(",") if c.strip()] if covar else None
    covar_suffix = f"_cov{len(covar_cols)}" if covar_cols else ""

    print("Loading metadata...")
    meta = load_metadata(metadata_path)

    print("Loading metric tables...")
    all_data = load_all_features(Path(metrics_dir), meta, ftype_filter="all")
    raw_df = all_data.get("raw", pd.DataFrame())
    z_df = all_data.get("z", pd.DataFrame())

    targets = ["MOT", "COG"] if target == "all" else [target]
    ftypes = ["raw", "z"] if ftype == "all" else [ftype]

    summary_rows = []

    if feature_list is not None:
        # Mixed raw+z mode: resolve each feature against its type, run once per target
        raw_feats = [f.strip() for f in feature_list.split(",") if f.strip()]
        desc = run_name or f"rcv_manual{len(raw_feats)}"
        for tgt in targets:
            print(f"\n{'='*60}")
            print(f"Target: {tgt} | Mixed features" +
                  (f" | Covariates: {covar_cols}" if covar_cols else ""))
            print(f"{'='*60}")

            matrix = build_model_matrix(
                raw_feats, {"raw": raw_df, "z": z_df}, tgt, covar_cols=covar_cols,
            )
            if matrix.empty or len(matrix) < 20:
                print(f"  Only {len(matrix)} valid subjects (< 20). Skipping.")
                continue

            y = matrix[tgt]
            all_predictors = [c for c in matrix.columns if c not in (tgt, "file")]
            X = matrix[all_predictors]

            print(f"  Subjects: {len(matrix)}")
            print(f"  Predictors ({X.shape[1]}): {list(X.columns)}")

            result = run_montecarlo_ridge(
                X, y, n_iter=n_iter, test_size=test_size, random_state=seed, alphas=alphas,
            )

            alpha_mean = np.mean(result["all_alphas"])
            print(f"\n  R2      = {result['r2_mean']:.4f} +/- {result['r2_std']:.4f}")
            print(f"  RMSE    = {result['rmse_mean']:.4f} +/- {result['rmse_std']:.4f}")
            print(f"  rho     = {result['rho_mean']:.4f} +/- {result['rho_std']:.4f}")
            print(f"  R2 < 0  = {result['r2_below_zero']:.1%} of iterations")
            print(f"  alpha   = {alpha_mean:.3f} (grid: {alphas[0]:.2f}–{alphas[-1]:.0f})")
            print(f"  Top coefficients:")
            for _, row in result["coef_summary"].head(5).iterrows():
                nz = row["frac_nonzero"]
                print(f"    {row['feature']}: coef={row['mean_coef']:.4f} +/- "
                      f"{row['std_coef']:.4f} (nonzero in {nz:.0%})")

            folder_suffix = covar_suffix
            run_dir = output_root / f"{tgt}_{desc}{folder_suffix}"
            run_dir.mkdir(parents=True, exist_ok=True)
            figs_dir = run_dir / "figures"
            figs_dir.mkdir(parents=True, exist_ok=True)

            tag = f"{tgt}_mixed"

            summary_rows.append({
                "target": tgt,
                "feature_type": "mixed",
                "descriptor": desc,
                "covariates": covar or "",
                "n_subjects": len(matrix),
                "n_predictors": X.shape[1],
                "predictors": ", ".join(X.columns),
                "n_iter": n_iter,
                "test_size": test_size,
                "alpha_mean": alpha_mean,
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "rmse_mean": result["rmse_mean"],
                "rmse_std": result["rmse_std"],
                "rho_mean": result["rho_mean"],
                "rho_std": result["rho_std"],
                "frac_r2_below_zero": result["r2_below_zero"],
            })

            pd.DataFrame([summary_rows[-1]]).to_csv(run_dir / "summary.csv", index=False)
            result["all_results"].to_csv(run_dir / "iterations.csv", index=False)

            pd.DataFrame({
                "y_true": result["y_true_all"],
                "y_pred": result["y_pred_all"],
            }).to_csv(run_dir / "predictions.csv", index=False)

            result["coef_summary"].to_csv(run_dir / "coefficients.csv", index=False)

            alpha_df = pd.DataFrame({
                "alpha": result["all_alphas"],
            })
            alpha_df.to_csv(run_dir / "alpha_selected.csv", index=False)
            print(f"  Saved: summary.csv, iterations.csv, predictions.csv, "
                  f"coefficients.csv, alpha_selected.csv")

            try:
                import matplotlib
                plot_all_regression_figures(result, figs_dir, tag)
            except Exception as e:
                print(f"  Warning: regression plots failed: {e}")
    else:
        # Original per-type mode with --corr-dir
        for tgt in targets:
            for ft in ftypes:
                print(f"\n{'='*60}")
                print(f"Target: {tgt} | Type: {ft}" +
                      (f" | Covariates: {covar_cols}" if covar_cols else ""))
                print(f"{'='*60}")

                df = all_data.get(ft)
                if df is None or df.empty:
                    print(f"  No {ft} data available. Skipping.")
                    continue

                corr_dir_path = Path(corr_dir)
                raw_feats = load_top_features(corr_dir_path, tgt, ft, top_k)
                desc = run_name or f"rcv_top{top_k}"

                if not raw_feats:
                    print("  No features loaded. Skipping.")
                    continue

                matrix = build_model_matrix(raw_feats, df, tgt, covar_cols=covar_cols)
                if matrix.empty or len(matrix) < 20:
                    print(f"  Only {len(matrix)} valid subjects (< 20). Skipping.")
                    continue

                y = matrix[tgt]
                all_predictors = [c for c in matrix.columns if c not in (tgt, "file")]
                X = matrix[all_predictors]

                print(f"  Subjects: {len(matrix)}")
                print(f"  Predictors ({X.shape[1]}): {list(X.columns)}")

                result = run_montecarlo_ridge(
                    X, y, n_iter=n_iter, test_size=test_size, random_state=seed, alphas=alphas,
                )

                alpha_mean = np.mean(result["all_alphas"])
                print(f"\n  R2      = {result['r2_mean']:.4f} +/- {result['r2_std']:.4f}")
                print(f"  RMSE    = {result['rmse_mean']:.4f} +/- {result['rmse_std']:.4f}")
                print(f"  rho     = {result['rho_mean']:.4f} +/- {result['rho_std']:.4f}")
                print(f"  R2 < 0  = {result['r2_below_zero']:.1%} of iterations")
                print(f"  alpha   = {alpha_mean:.3f} (grid: {alphas[0]:.2f}–{alphas[-1]:.0f})")
                print(f"  Top coefficients:")
                for _, row in result["coef_summary"].head(5).iterrows():
                    nz = row["frac_nonzero"]
                    print(f"    {row['feature']}: coef={row['mean_coef']:.4f} +/- "
                          f"{row['std_coef']:.4f} (nonzero in {nz:.0%})")

                folder_suffix = covar_suffix
                run_dir = output_root / f"{tgt}_{desc}_{ft}{folder_suffix}"
                run_dir.mkdir(parents=True, exist_ok=True)
                figs_dir = run_dir / "figures"
                figs_dir.mkdir(parents=True, exist_ok=True)

                tag = f"{tgt}_{ft}"

                summary_rows.append({
                    "target": tgt,
                    "feature_type": ft,
                    "descriptor": desc,
                    "covariates": covar or "",
                    "n_subjects": len(matrix),
                    "n_predictors": X.shape[1],
                    "predictors": ", ".join(X.columns),
                    "n_iter": n_iter,
                    "test_size": test_size,
                    "alpha_mean": alpha_mean,
                    "r2_mean": result["r2_mean"],
                    "r2_std": result["r2_std"],
                    "rmse_mean": result["rmse_mean"],
                    "rmse_std": result["rmse_std"],
                    "rho_mean": result["rho_mean"],
                    "rho_std": result["rho_std"],
                    "frac_r2_below_zero": result["r2_below_zero"],
                })

                pd.DataFrame([summary_rows[-1]]).to_csv(run_dir / "summary.csv", index=False)
                result["all_results"].to_csv(run_dir / "iterations.csv", index=False)

                pd.DataFrame({
                    "y_true": result["y_true_all"],
                    "y_pred": result["y_pred_all"],
                }).to_csv(run_dir / "predictions.csv", index=False)

                result["coef_summary"].to_csv(run_dir / "coefficients.csv", index=False)

                alpha_df = pd.DataFrame({
                    "alpha": result["all_alphas"],
                })
                alpha_df.to_csv(run_dir / "alpha_selected.csv", index=False)
                print(f"  Saved: summary.csv, iterations.csv, predictions.csv, "
                      f"coefficients.csv, alpha_selected.csv")

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

    print("\nDone. Ridge regression analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo cross-validated Ridge regression predicting "
            "MOT or COG from speech-graph features."
        ),
        prog="python -m src.analysis.linear_regression_rcv",
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
        help="Comma-separated feature identifiers, e.g. 'lsc_T7W30,l2_T2W30'",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Custom name for output subfolder. Default: rcv_manual{N} or rcv_top{K}",
    )
    parser.add_argument(
        "--covar", default=None,
        help="Comma-separated covariate column names from metadata, "
             "e.g. 'School year' or 'School year,Age'",
    )
    parser.add_argument(
        "--corr-dir", default="outputs/correlations",
        help="Directory with top feature CSVs from correlation_analysis.py. "
             "Only used without --feature-list.",
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
        "--output", default="outputs/linear_regression/ridgecv_mc",
        help="Root output directory (default: outputs/linear_regression/ridgecv_mc)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top features to use from --corr-dir (default: 5)",
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
    parser.add_argument(
        "--alphas", type=str, default=None,
        help="Comma-separated alpha grid, e.g. '0.01,0.1,1,10,100'. "
             "Default: logspace(-2, 3, 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alphas is not None:
        alphas = np.array([float(a.strip()) for a in args.alphas.split(",")])
    else:
        alphas = ALPHA_GRID

    run_analysis(
        target=args.target,
        ftype=args.type,
        feature_list=args.feature_list,
        run_name=args.run_name,
        covar=args.covar,
        corr_dir=args.corr_dir,
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
        output_dir=args.output,
        top_k=args.top_k,
        n_iter=args.n_iter,
        test_size=args.test_size,
        seed=args.seed,
        alphas=alphas,
    )


if __name__ == "__main__":
    main()
