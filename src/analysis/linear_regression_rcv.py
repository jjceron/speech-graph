"""Monte Carlo cross-validated Ridge regression (MMCV 70/20/10)
predicting Barratt impulsivity targets from speech-graph features.

Supports per-window experiments, manual feature lists, and top-K selection.
Uses stratified 70/20/10 splits, RidgeCV for automatic alpha selection,
and reports metrics with 95% t-distribution confidence intervals.

Usage:
    py -m src.analysis.linear_regression_rcv --task 2 --window 10 --experiment raw --all-metrics --run-name W10_allmetrics
    py -m src.analysis.linear_regression_rcv --task 2 --window 10 --experiment raw --top-k 5 --run-name W10_top5metrics
    py -m src.analysis.linear_regression_rcv --task 2 --window 10 --experiment zscores --all-metrics --run-name W10_zscores_allmetrics
    py -m src.analysis.linear_regression_rcv --task 2 --window 10 --experiment rawzscore --all-metrics --run-name W10_rawzscore_allmetrics

Output root (default): outputs/linear_regression/
"""

from __future__ import annotations

import argparse
import re
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import experiment_config as expcfg
from src.analysis.experiment_config import (
    _compute_targets, find_means_tables, load_feature_table,
)
from src.visualization.lr_metrics import plot_all_regression_figures
from src.pipeline.speechgraph import load_metadata


METRICS_OF_INTEREST = [
    "nodes", "edges", "re", "pe", "l1", "l2", "l3",
    "lcc", "lsc", "atd", "density", "diameter", "asp", "cc",
]

ALL_TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]

ALPHA_GRID = np.logspace(-2, 3, 20)


def filter_cc_features(feat_ids: list[str]) -> list[str]:
    out = []
    for fid in feat_ids:
        is_cc = fid.startswith("cc_") or fid.startswith("z_cc_")
        if is_cc:
            tag_match = re.search(r"W(\d+)", fid)
            if tag_match and int(tag_match.group(1)) < 100:
                continue
        out.append(fid)
    return out


def _mean_ci(values: np.ndarray) -> dict:
    """Return mean, std, ci_lower, ci_upper, t_critical, df using t-distribution."""
    n = len(values)
    df = n - 1
    t_crit = float(sp_stats.t.ppf(0.975, df))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    ci = t_crit * sem
    return {
        "mean": mean, "std": std,
        "ci_lower": mean - ci, "ci_upper": mean + ci,
        "t_critical": t_crit, "df": df,
    }


def generate_mmcv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 400,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate stratified MMCV splits (70/20/10) with coverage tracking."""
    n_samples = len(X)
    y_qcut = pd.qcut(y, q=5, labels=False, duplicates="drop")

    splits = []
    seen_test = set()
    seen_val = set()
    split_no = 0

    while split_no < n_splits or len(seen_test) < n_samples or len(seen_val) < n_samples:
        seed = random_state + split_no
        rng = np.random.RandomState(seed)

        sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(sss_outer.split(X, y_qcut))

        train_val_y = y_qcut.iloc[train_val_idx]
        train_val_idx_arr = np.array(train_val_idx)

        sss_inner = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size / (train_size + val_size), random_state=seed
        )
        train_idx, val_idx = next(sss_inner.split(train_val_idx_arr, train_val_y))

        train_idx = train_val_idx_arr[train_idx]
        val_idx = train_val_idx_arr[val_idx]

        seen_test.update(test_idx)
        seen_val.update(val_idx)

        splits.append((train_idx, val_idx, test_idx))
        split_no += 1

    return splits[:n_splits]


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


def run_mmcv_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: np.ndarray | None = None,
    top_k: int | None = None,
) -> dict:
    """Run MMCV Ridge regression with pre-computed splits.
    
    If top_k is set, selects top-K features by |Spearman r| on the training
    set of EACH split (within-CV selection, no data leakage).
    """
    if alphas is None:
        alphas = ALPHA_GRID

    feature_names = list(X.columns)
    n_splits = len(splits)

    val_results = []
    test_results = []
    val_y_true_all, val_y_pred_all = [], []
    test_y_true_all, test_y_pred_all = [], []
    subjects_val, subjects_test = [], []
    split_ids_val, split_ids_test = [], []
    all_alphas = []
    coef_tracker: dict[str, list[float]] = {}

    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Within-CV feature selection: compute |Spearman r| on train only
        if top_k is not None and top_k < len(feature_names):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=sp_stats.ConstantInputWarning)
                train_r = X_train.apply(
                    lambda col: abs(spearmanr(col, y_train)[0])
                ).dropna()
            current_feats = train_r.sort_values(ascending=False).head(top_k).index.tolist()
        else:
            current_feats = feature_names

        X_train_k = X_train[current_feats]
        X_val_k = X_val[current_feats]
        X_test_k = X_test[current_feats]

        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, scoring="neg_mean_squared_error"),
        )
        model.fit(X_train_k, y_train)

        # VAL predictions
        y_val_pred = model.predict(X_val_k)
        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae_val = mean_absolute_error(y_val, y_val_pred)
        rho_val, _ = spearmanr(y_val, y_val_pred)

        # TEST predictions
        y_test_pred = model.predict(X_test_k)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rho_test, _ = spearmanr(y_test, y_test_pred)

        val_results.append({
            "r2": r2_val, "rmse": rmse_val, "mae": mae_val, "rho": rho_val,
        })
        test_results.append({
            "r2": r2_test, "rmse": rmse_test, "mae": mae_test, "rho": rho_test,
        })

        val_y_true_all.extend(y_val)
        val_y_pred_all.extend(y_val_pred)
        test_y_true_all.extend(y_test)
        test_y_pred_all.extend(y_test_pred)
        subjects_val.extend(X.iloc[val_idx].index.tolist())
        subjects_test.extend(X.iloc[test_idx].index.tolist())
        split_ids_val.extend([i] * len(val_idx))
        split_ids_test.extend([i] * len(test_idx))
        all_alphas.append(model.named_steps["ridgecv"].alpha_)

        for feat, coef in zip(current_feats, model.named_steps["ridgecv"].coef_):
            coef_tracker.setdefault(feat, []).append(coef)

    df_val = pd.DataFrame(val_results)
    df_test = pd.DataFrame(test_results)

    coef_rows = []
    for feat, coefs in coef_tracker.items():
        arr = np.array(coefs)
        coef_rows.append({
            "feature": feat,
            "mean_coef": float(arr.mean()),
            "std_coef": float(arr.std()),
            "frac_nonzero": float(np.mean(np.abs(arr) > 1e-12)),
            "n_selected": len(coefs),
            "frac_selected": len(coefs) / n_splits,
        })
    coef_summary = pd.DataFrame(coef_rows).sort_values("mean_coef", key=abs, ascending=False)

    alpha_mean = float(np.mean(all_alphas))
    alpha_std = float(np.std(all_alphas))

    def _summarize_metrics(df: pd.DataFrame) -> dict:
        def _unpack(key, vals):
            ci = _mean_ci(vals)
            return {
                f"{key}_mean": ci["mean"],
                f"{key}_std": ci["std"],
                f"{key}_ci_lower": ci["ci_lower"],
                f"{key}_ci_upper": ci["ci_upper"],
                f"{key}_t_critical": ci["t_critical"],
                f"{key}_df": ci["df"],
            }
        out = {}
        out.update(_unpack("r2", df["r2"].values))
        out.update(_unpack("rmse", df["rmse"].values))
        out.update(_unpack("mae", df["mae"].values))
        out.update(_unpack("rho", df["rho"].values))
        out["r2_below_zero"] = (df["r2"] < 0).mean()
        return out

    summary = {
        "val": _summarize_metrics(df_val),
        "test": _summarize_metrics(df_test),
        "n_iter": n_splits,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "coef_summary": coef_summary,
        "all_alphas": all_alphas,
        "all_results_val": df_val,
        "all_results_test": df_test,
        "y_true_val": val_y_true_all,
        "y_pred_val": val_y_pred_all,
        "y_true_test": test_y_true_all,
        "y_pred_test": test_y_pred_all,
        "subjects_val": subjects_val,
        "subjects_test": subjects_test,
        "split_ids_val": split_ids_val,
        "split_ids_test": split_ids_test,
    }
    return summary


def run_per_window_analysis(
    task_num: int,
    window: str,
    experiment: str,
    metadata_path: str | Path,
    metrics_dir: Path,
    targets: list[str],
    covar_cols: list[str] | None = None,
    output_root: Path = Path("outputs/linear_regression"),
    n_iter: int = 400,
    seed: int = 42,
    alphas: np.ndarray | None = None,
    top_k: int | None = None,
    run_name: str | None = None,
) -> list[dict]:
    """Run per-window MMCV Ridge regression for all targets."""
    if alphas is None:
        alphas = ALPHA_GRID

    full_window = f"T{task_num}W{window}"
    metric_mode = "all_metrics" if top_k is None else f"top{top_k}_metrics"
    summary_rows = []

    for tgt in targets:
        X_matrix, y_series = expcfg.load_experiment_matrix(
            experiment=experiment,
            task=task_num,
            window=int(window),
            target=tgt,
            metrics_dir=metrics_dir,
            metadata_path=metadata_path,
            covar_cols=covar_cols,
        )
        X_matrix = X_matrix.rename(
            columns={col: f"{col}_{full_window}" for col in X_matrix.columns}
        )

        print(f"\n{'='*60}")
        print(f"Target: {tgt} | Window: W{window} | Task: {task_num} | Experiment: {experiment}" +
              (f" | Covariates: {covar_cols}" if covar_cols else "") +
              f" | Mode: {metric_mode}")
        print(f"{'='*60}")
        print(f"  Subjects: {len(y_series)}")
        print(f"  Predictors ({X_matrix.shape[1]}): {list(X_matrix.columns)}")

        # Generate MMCV splits (stratified by target quintiles)
        splits = generate_mmcv_splits(
            X_matrix, y_series, n_splits=n_iter, random_state=seed,
        )

        # Run MMCV Ridge (within-CV feature selection if top_k is set)
        result = run_mmcv_ridge(X_matrix, y_series, splits, alphas=alphas, top_k=top_k)

        # Save split assignments
        split_records = []
        for i, (tr, vl, te) in enumerate(splits):
            for idx in tr:
                split_records.append({
                    "split": i, "subject": X_matrix.index[idx], "set": "TRAIN"
                })
            for idx in vl:
                split_records.append({
                    "split": i, "subject": X_matrix.index[idx], "set": "VALIDATION"
                })
            for idx in te:
                split_records.append({
                    "split": i, "subject": X_matrix.index[idx], "set": "TEST"
                })
        df_splits = pd.DataFrame(split_records)

        task_str = f"task{task_num}"
        exp_root = run_dir = output_root / task_str / (run_name or f"W{window}_{experiment}_{metric_mode}")
        run_dir = exp_root / tgt
        run_dir.mkdir(parents=True, exist_ok=True)
        figs_dir = run_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)

        # Print results
        for set_name in ("val", "test"):
            s = result[set_name]
            print(f"\n  [{set_name.upper()}] "
                  f"R2 = {s['r2_mean']:.4f} [{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}] "
                  f"| RMSE = {s['rmse_mean']:.3f} [{s['rmse_ci_lower']:.3f}, {s['rmse_ci_upper']:.3f}] "
                  f"| MAE = {s['mae_mean']:.3f} [{s['mae_ci_lower']:.3f}, {s['mae_ci_upper']:.3f}] "
                  f"| rho = {s['rho_mean']:.4f} [{s['rho_ci_lower']:.4f}, {s['rho_ci_upper']:.4f}]")
        print(f"  R2 < 0 (test): {result['test']['r2_below_zero']:.1%} of iterations")
        print(f"  alpha = {result['alpha_mean']:.3f} +/- {result['alpha_std']:.3f}")
        print(f"  Top coefficients:")
        for _, row in result["coef_summary"].head(5).iterrows():
            nz = row["frac_nonzero"]
            print(f"    {row['feature']}: coef={row['mean_coef']:.4f} +/- "
                  f"{row['std_coef']:.4f} (nonzero in {nz:.0%})")

        # Build summary row (using TEST as the primary metric)
        s_test = result["test"]
        summary_rows.append({
            "target": tgt,
            "window": full_window,
            "experiment": experiment,
            "covariates": ", ".join(covar_cols) if covar_cols else "",
            "n_subjects": len(y_series),
            "n_candidate_predictors": X_matrix.shape[1],
            "n_selected_per_split": top_k if top_k is not None else X_matrix.shape[1],
            "predictors": ", ".join(X_matrix.columns),
            "n_iter": n_iter,
            "alpha_mean": result["alpha_mean"],
            "alpha_std": result["alpha_std"],
            "r2_mean": s_test["r2_mean"],
            "r2_std": s_test["r2_std"],
            "r2_ci_lower": s_test["r2_ci_lower"],
            "r2_ci_upper": s_test["r2_ci_upper"],
            "rmse_mean": s_test["rmse_mean"],
            "rmse_std": s_test["rmse_std"],
            "rmse_ci_lower": s_test["rmse_ci_lower"],
            "rmse_ci_upper": s_test["rmse_ci_upper"],
            "mae_mean": s_test["mae_mean"],
            "mae_std": s_test["mae_std"],
            "mae_ci_lower": s_test["mae_ci_lower"],
            "mae_ci_upper": s_test["mae_ci_upper"],
            "rho_mean": s_test["rho_mean"],
            "rho_std": s_test["rho_std"],
            "rho_ci_lower": s_test["rho_ci_lower"],
            "rho_ci_upper": s_test["rho_ci_upper"],
            "frac_r2_below_zero": s_test["r2_below_zero"],
        })

        # Save summary_TEST.csv and summary_VALIDATION.csv
        for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
            s = result[set_name]
            pd.DataFrame([{
                "set": label,
                "r2_mean": s["r2_mean"], "r2_std": s["r2_std"],
                "r2_ci_lower": s["r2_ci_lower"], "r2_ci_upper": s["r2_ci_upper"],
                "rmse_mean": s["rmse_mean"], "rmse_std": s["rmse_std"],
                "rmse_ci_lower": s["rmse_ci_lower"], "rmse_ci_upper": s["rmse_ci_upper"],
                "mae_mean": s["mae_mean"], "mae_std": s["mae_std"],
                "mae_ci_lower": s["mae_ci_lower"], "mae_ci_upper": s["mae_ci_upper"],
                "rho_mean": s["rho_mean"], "rho_std": s["rho_std"],
                "rho_ci_lower": s["rho_ci_lower"], "rho_ci_upper": s["rho_ci_upper"],
                "r2_below_zero": s["r2_below_zero"],
            }]).to_csv(run_dir / f"summary_{label}.csv", index=False)

        # Save ci_summary.csv (long-format, one row per metric x set)
        ci_rows = []
        for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
            s = result[set_name]
            for metric_key, metric_label in [
                ("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE"), ("rho", "rho"),
            ]:
                ci_rows.append({
                    "target": tgt,
                    "window": full_window,
                    "set": label,
                    "metric": metric_label,
                    "mean": round(s[f"{metric_key}_mean"], 6),
                    "std": round(s[f"{metric_key}_std"], 6),
                    "ci_lower": round(s[f"{metric_key}_ci_lower"], 6),
                    "ci_upper": round(s[f"{metric_key}_ci_upper"], 6),
                    "t_critical": round(s[f"{metric_key}_t_critical"], 4),
                    "df": int(s[f"{metric_key}_df"]),
                    "significant": (s[f"{metric_key}_ci_lower"] > 0) or (s[f"{metric_key}_ci_upper"] < 0),
                })
        pd.DataFrame(ci_rows).to_csv(run_dir / "ci_summary.csv", index=False)

        result["all_results_test"].to_csv(run_dir / "iterations.csv", index=False)
        pd.DataFrame({
            "split": result["split_ids_test"],
            "subject": result["subjects_test"],
            "set": "TEST",
            "y_true": result["y_true_test"],
            "y_pred": result["y_pred_test"],
        }).to_csv(run_dir / "predictions.csv", index=False)
        result["coef_summary"].to_csv(run_dir / "coefficients.csv", index=False)
        pd.DataFrame({"alpha": result["all_alphas"]}).to_csv(
            run_dir / "alpha_selected.csv", index=False
        )
        df_splits.to_csv(run_dir / "mmcv_splits.csv", index=False)

        print(f"  Saved: summary_VALIDATION.csv, summary_TEST.csv, ci_summary.csv, "
              f"iterations.csv, predictions.csv, coefficients.csv, "
              f"alpha_selected.csv, mmcv_splits.csv")

        try:
            import matplotlib
            plot_all_regression_figures(result, figs_dir, f"{tgt}_{full_window}", has_val=True)
        except Exception as e:
            print(f"  Warning: regression plots failed: {e}")

    if len(summary_rows) > 1:
        summary_all = pd.DataFrame(summary_rows)
        summary_all_path = exp_root / "multi_run_summary.csv"
        summary_all.to_csv(summary_all_path, index=False)
        # Also save combined ci_summary across all targets
        all_ci = []
        for tgt in targets:
            ci_path = exp_root / tgt / "ci_summary.csv"
            if ci_path.exists():
                all_ci.append(pd.read_csv(ci_path))
        if all_ci:
            combined = pd.concat(all_ci, ignore_index=True)
            combined.to_csv(exp_root / "ci_summary.csv", index=False)
            print(f"  Combined ci_summary.csv saved: {exp_root / 'ci_summary.csv'}")
        print(f"\nMulti-run summary saved: {summary_all_path}")
        print(summary_all.round(4).to_string(index=False))

    return summary_rows


def run_analysis(
    target: str = "all",
    experiment: str = "raw",
    feature_list: str | None = None,
    window: str | None = None,
    task: int | None = None,
    run_name: str | None = None,
    covar: str | None = None,
    corr_dir: str | Path = "outputs/correlations",
    metrics_dir: str | Path = "data/processed/metrics",
    metadata_path: str | Path = "data/raw/metadata.xlsx",
    output_dir: str | Path = "outputs/linear_regression",
    all_metrics: bool = False,
    top_k: int | None = 0,
    n_iter: int = 400,
    seed: int = 42,
    alphas: np.ndarray | None = None,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if alphas is None:
        alphas = ALPHA_GRID
    covar_cols = [c.strip() for c in covar.split(",") if c.strip()] if covar else None

    # Determine targets
    if target == "all":
        targets = ALL_TARGETS
    elif target == "old":
        targets = ["MOT", "COG"]
    else:
        targets = [t.strip() for t in target.split(",")]

    if window is not None:
        # Per-window mode
        if all_metrics:
            effective_top_k = None
        else:
            effective_top_k = top_k if top_k and top_k > 0 else None
        if task is None:
            print("Error: --task is required when using --window.")
            return
        summary_rows = run_per_window_analysis(
            task_num=task,
            window=window,
            experiment=experiment,
            metadata_path=metadata_path,
            metrics_dir=Path(metrics_dir),
            targets=targets,
            covar_cols=covar_cols,
            output_root=output_root,
            n_iter=n_iter,
            seed=seed,
            alphas=alphas,
            top_k=effective_top_k,
            run_name=run_name,
        )
        return

    # Legacy modes: feature-list or corr-dir
    print("Loading metadata...")
    meta = load_metadata(metadata_path)
    meta = _compute_targets(meta)

    print("Loading metric tables...")
    all_data = load_all_features(Path(metrics_dir), meta, ftype_filter="all")
    raw_df = all_data.get("raw", pd.DataFrame())
    z_df = all_data.get("z", pd.DataFrame())

    ftypes = {"raw": ["raw"], "zscores": ["z"], "rawzscore": ["raw", "z"]}[experiment]
    summary_rows = []

    if feature_list is not None:
        raw_feats = [f.strip() for f in feature_list.split(",") if f.strip()]
        raw_feats = filter_cc_features(raw_feats)
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

            splits = generate_mmcv_splits(
                X, y, n_splits=n_iter, random_state=seed,
            )
            result = run_mmcv_ridge(X, y, splits, alphas=alphas, top_k=None)

            covar_suffix = f"_cov{len(covar_cols)}" if covar_cols else ""

            for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
                s = result[set_name]
                print(f"\n  [{label}] "
                      f"R2 = {s['r2_mean']:.4f} [{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}]"
                      f" | rho = {s['rho_mean']:.4f} [{s['rho_ci_lower']:.4f}, {s['rho_ci_upper']:.4f}]")
            print(f"  alpha = {result['alpha_mean']:.3f} +/- {result['alpha_std']:.3f}")
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

            s_test = result["test"]
            summary_rows.append({
                "target": tgt, "feature_type": "mixed", "descriptor": desc,
                "covariates": covar or "", "n_subjects": len(matrix),
                "n_predictors": X.shape[1], "predictors": ", ".join(X.columns),
                "n_iter": n_iter, "alpha_mean": result["alpha_mean"],
                "r2_mean": s_test["r2_mean"], "r2_std": s_test["r2_std"],
                "r2_ci_lower": s_test["r2_ci_lower"], "r2_ci_upper": s_test["r2_ci_upper"],
                "rmse_mean": s_test["rmse_mean"], "rmse_std": s_test["rmse_std"],
                "rmse_ci_lower": s_test["rmse_ci_lower"], "rmse_ci_upper": s_test["rmse_ci_upper"],
                "mae_mean": s_test["mae_mean"], "mae_std": s_test["mae_std"],
                "mae_ci_lower": s_test["mae_ci_lower"], "mae_ci_upper": s_test["mae_ci_upper"],
                "rho_mean": s_test["rho_mean"], "rho_std": s_test["rho_std"],
                "rho_ci_lower": s_test["rho_ci_lower"], "rho_ci_upper": s_test["rho_ci_upper"],
                "frac_r2_below_zero": s_test["r2_below_zero"],
            })

            for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
                s = result[set_name]
                pd.DataFrame([{
                    "set": label, "r2_mean": s["r2_mean"], "r2_std": s["r2_std"],
                    "r2_ci_lower": s["r2_ci_lower"], "r2_ci_upper": s["r2_ci_upper"],
                    "rmse_mean": s["rmse_mean"], "rmse_std": s["rmse_std"],
                    "rmse_ci_lower": s["rmse_ci_lower"], "rmse_ci_upper": s["rmse_ci_upper"],
                    "mae_mean": s["mae_mean"], "mae_std": s["mae_std"],
                    "mae_ci_lower": s["mae_ci_lower"], "mae_ci_upper": s["mae_ci_upper"],
                    "rho_mean": s["rho_mean"], "rho_std": s["rho_std"],
                    "rho_ci_lower": s["rho_ci_lower"], "rho_ci_upper": s["rho_ci_upper"],
                    "r2_below_zero": s["r2_below_zero"],
                }]).to_csv(run_dir / f"summary_{label}.csv", index=False)

            result["all_results_test"].to_csv(run_dir / "iterations.csv", index=False)
            pd.DataFrame({
                "y_true": result["y_true_test"], "y_pred": result["y_pred_test"],
            }).to_csv(run_dir / "predictions.csv", index=False)
            result["coef_summary"].to_csv(run_dir / "coefficients.csv", index=False)
            pd.DataFrame({"alpha": result["all_alphas"]}).to_csv(
                run_dir / "alpha_selected.csv", index=False
            )
            print(f"  Saved: summary_VALIDATION.csv, summary_TEST.csv, iterations.csv, "
                  f"predictions.csv, coefficients.csv, alpha_selected.csv")

            try:
                import matplotlib
                plot_all_regression_figures(result, figs_dir, tag, has_val=True)
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
                legacy_k = top_k if top_k and top_k > 0 else 5
                raw_feats = load_top_features(corr_dir_path, tgt, ft, legacy_k)
                raw_feats = filter_cc_features(raw_feats)
                desc = run_name or f"rcv_top{legacy_k}"

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

                splits = generate_mmcv_splits(
                    X, y, n_splits=n_iter, random_state=seed,
                )
                result = run_mmcv_ridge(X, y, splits, alphas=alphas, top_k=None)

                covar_suffix = f"_cov{len(covar_cols)}" if covar_cols else ""

                for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
                    s = result[set_name]
                    print(f"\n  [{label}] "
                          f"R2 = {s['r2_mean']:.4f} [{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}]"
                          f" | rho = {s['rho_mean']:.4f} [{s['rho_ci_lower']:.4f}, {s['rho_ci_upper']:.4f}]")
                print(f"  alpha = {result['alpha_mean']:.3f} +/- {result['alpha_std']:.3f}")
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

                s_test = result["test"]
                summary_rows.append({
                    "target": tgt, "feature_type": ft, "descriptor": desc,
                    "covariates": covar or "", "n_subjects": len(matrix),
                    "n_predictors": X.shape[1], "predictors": ", ".join(X.columns),
                    "n_iter": n_iter, "alpha_mean": result["alpha_mean"],
                    "r2_mean": s_test["r2_mean"], "r2_std": s_test["r2_std"],
                    "r2_ci_lower": s_test["r2_ci_lower"], "r2_ci_upper": s_test["r2_ci_upper"],
                    "rmse_mean": s_test["rmse_mean"], "rmse_std": s_test["rmse_std"],
                    "rmse_ci_lower": s_test["rmse_ci_lower"], "rmse_ci_upper": s_test["rmse_ci_upper"],
                    "mae_mean": s_test["mae_mean"], "mae_std": s_test["mae_std"],
                    "mae_ci_lower": s_test["mae_ci_lower"], "mae_ci_upper": s_test["mae_ci_upper"],
                    "rho_mean": s_test["rho_mean"], "rho_std": s_test["rho_std"],
                    "rho_ci_lower": s_test["rho_ci_lower"], "rho_ci_upper": s_test["rho_ci_upper"],
                    "frac_r2_below_zero": s_test["r2_below_zero"],
                })

                for set_name, label in [("val", "VALIDATION"), ("test", "TEST")]:
                    s = result[set_name]
                    pd.DataFrame([{
                        "set": label, "r2_mean": s["r2_mean"], "r2_std": s["r2_std"],
                        "r2_ci_lower": s["r2_ci_lower"], "r2_ci_upper": s["r2_ci_upper"],
                        "rmse_mean": s["rmse_mean"], "rmse_std": s["rmse_std"],
                        "rmse_ci_lower": s["rmse_ci_lower"], "rmse_ci_upper": s["rmse_ci_upper"],
                        "mae_mean": s["mae_mean"], "mae_std": s["mae_std"],
                        "mae_ci_lower": s["mae_ci_lower"], "mae_ci_upper": s["mae_ci_upper"],
                        "rho_mean": s["rho_mean"], "rho_std": s["rho_std"],
                        "rho_ci_lower": s["rho_ci_lower"], "rho_ci_upper": s["rho_ci_upper"],
                        "r2_below_zero": s["r2_below_zero"],
                    }]).to_csv(run_dir / f"summary_{label}.csv", index=False)

                result["all_results_test"].to_csv(run_dir / "iterations.csv", index=False)
                pd.DataFrame({
                    "y_true": result["y_true_test"], "y_pred": result["y_pred_test"],
                }).to_csv(run_dir / "predictions.csv", index=False)
                result["coef_summary"].to_csv(run_dir / "coefficients.csv", index=False)
                pd.DataFrame({"alpha": result["all_alphas"]}).to_csv(
                    run_dir / "alpha_selected.csv", index=False
                )
                print(f"  Saved: summary_VALIDATION.csv, summary_TEST.csv, iterations.csv, "
                      f"predictions.csv, coefficients.csv, alpha_selected.csv")

                try:
                    import matplotlib
                    plot_all_regression_figures(result, figs_dir, tag, has_val=True)
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
            "MMCV Ridge regression (70/20/10) predicting Barratt impulsivity "
            "targets from speech-graph features. Reports metrics with 95% CI."
        ),
        prog="python -m src.analysis.linear_regression_rcv",
    )
    parser.add_argument(
        "--task", type=int, default=None,
        help="Task number, e.g. 2, 6, 7 (required when using --window)",
    )
    parser.add_argument(
        "--window", default=None,
        help="Window number for per-window experiment, e.g. '10' for W10. "
             "Use with --task to specify which task.",
    )
    parser.add_argument(
        "--targets", default="all",
        help="Comma-separated target(s): MOT, COG, MOT_V4, COG_V1, old, or all (default: all)",
    )
    parser.add_argument(
        "--experiment", default="raw", choices=["raw", "zscores", "rawzscore"],
        help="Feature experiment: 'raw', 'zscores', or 'rawzscore' (default: raw)",
    )
    parser.add_argument(
        "--all-metrics", action="store_true",
        help="Use all METRICS_OF_INTEREST metrics (mutually exclusive with --top-k)",
    )
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Number of top partial-correlation features to use. "
             "Mutually exclusive with --all-metrics (default: 0).",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Output subfolder name, e.g. 'W10_allmetrics'. "
             "If omitted, auto-generated from window+type+mode.",
    )
    parser.add_argument(
        "--feature-list", default=None,
        help="[Legacy] Comma-separated feature identifiers, e.g. 'lsc_T7W30,l2_T2W30'",
    )
    parser.add_argument(
        "--covar", default=None,
        help="Comma-separated covariate column names from metadata, "
             "e.g. 'School year' or 'School year,Age'",
    )
    parser.add_argument(
        "--corr-dir", default="outputs/correlations",
        help="Directory with top feature CSVs. Legacy mode only.",
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
        "--output", default="outputs/linear_regression",
        help="Root output directory (default: outputs/linear_regression)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=400,
        help="Number of MMCV splits (default: 400)",
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

    if args.window is not None and args.task is None:
        print("Error: --task is required when using --window")
        sys.exit(1)
    if args.window is not None and args.all_metrics and args.top_k > 0:
        print("Error: --all-metrics and --top-k are mutually exclusive")
        sys.exit(1)
    if args.alphas is not None:
        alphas = np.array([float(a.strip()) for a in args.alphas.split(",")])
    else:
        alphas = ALPHA_GRID

    run_analysis(
        target=args.targets,
        experiment=args.experiment,
        feature_list=args.feature_list,
        window=args.window,
        task=args.task,
        run_name=args.run_name,
        covar=args.covar,
        corr_dir=args.corr_dir,
        metrics_dir=args.metrics_dir,
        metadata_path=args.metadata,
        output_dir=args.output,
        all_metrics=args.all_metrics,
        top_k=args.top_k if not args.all_metrics else 0,
        n_iter=args.n_iter,
        seed=args.seed,
        alphas=alphas,
    )


if __name__ == "__main__":
    main()
