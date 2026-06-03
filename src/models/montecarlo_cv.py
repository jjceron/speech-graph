from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import parse_csv_list, resolve_targets
from src.graphs import MODEL_METRICS

DEFAULT_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
CONTEXT_COLUMNS = ["target", "activity_number", "activity", "window_size", "scheme_window_size"]


def parse_alphas(text: str | None) -> list[float]:
    values = [float(part.strip()) for part in str(text or "").split(",") if part.strip()]
    return values or DEFAULT_ALPHAS


def _corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return float("nan"), float("nan")
    if method == "spearman":
        r, p = stats.spearmanr(x[mask], y[mask])
    else:
        r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def _pipeline(alphas: list[float]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.asarray(alphas, dtype=float))),
        ]
    )


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def canonical_nlp_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in MODEL_METRICS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
                features.append(col)
    return features


def resolve_control_columns(df: pd.DataFrame, controls: Iterable[str]) -> list[str]:
    out = []
    for col in controls:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
                out.append(col)
    return out


def model_feature_sets(df: pd.DataFrame, controls: list[str], requested: list[str]) -> dict[str, list[str]]:
    nlp = canonical_nlp_features(df)
    features: dict[str, list[str]] = {}
    for name in requested:
        clean = name.strip().lower()
        if clean in {"nlp", "nlp_only", "speechgraph"}:
            features["nlp_only"] = nlp
        elif clean in {"school_year", "schoolyear", "controls", "control", "control_only"}:
            features["school_year_only"] = controls
        elif clean in {"school_year_nlp", "nlp_school_year", "controls_nlp", "control_nlp", "combined"}:
            features["school_year_plus_nlp"] = controls + nlp
    if not features:
        features = {"nlp_only": nlp, "school_year_only": controls, "school_year_plus_nlp": controls + nlp}
    return {name: cols for name, cols in features.items() if cols}


def _context_from_group(sub: pd.DataFrame, target: str) -> dict:
    context = {"target": target}
    for col in ["activity_number", "activity", "window_size", "scheme_window_size"]:
        if col in sub.columns:
            vals = sub[col].dropna().unique()
            if len(vals) == 1:
                val = vals[0]
                if col in {"activity_number", "window_size", "scheme_window_size"}:
                    try:
                        val = int(val)
                    except Exception:
                        pass
                context[col] = val
    return context


def _activity_window_groups(df: pd.DataFrame) -> list[tuple[tuple[int, int], pd.DataFrame]]:
    if "activity_number" not in df.columns:
        raise ValueError("Expected activity_number in input CSV")
    window_col = "window_size" if "window_size" in df.columns else "scheme_window_size" if "scheme_window_size" in df.columns else None
    if window_col is None:
        raise ValueError("Expected window_size or scheme_window_size in input CSV")
    work = df.copy()
    work["activity_number"] = pd.to_numeric(work["activity_number"], errors="coerce")
    work[window_col] = pd.to_numeric(work[window_col], errors="coerce")
    work = work[work["activity_number"].notna() & work[window_col].notna()]
    groups: list[tuple[tuple[int, int], pd.DataFrame]] = []
    for (activity, window), sub in work.groupby(["activity_number", window_col], dropna=False):
        sub = sub.copy()
        sub["activity_number"] = int(activity)
        sub["window_size"] = int(window)
        sub["scheme_window_size"] = int(window)
        groups.append(((int(activity), int(window)), sub))
    return groups


def _make_splitter(n_repeats: int, test_size: float, random_state: int, groups: pd.Series | None):
    if groups is not None and groups.nunique(dropna=True) >= 2:
        return GroupShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state), groups.astype(str)
    return ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state), None


def _fit_predict_one(
    x: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, Pipeline, float, float, float, float]:
    pipe = _pipeline(alphas)
    pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
    pred = pipe.predict(x.iloc[test_idx])
    y_train_mean = float(y.iloc[train_idx].mean())
    base = np.repeat(y_train_mean, len(test_idx))
    y_test = y.iloc[test_idx].to_numpy(dtype=float)
    rmse_model = float(mean_squared_error(y_test, pred) ** 0.5)
    rmse_base = float(mean_squared_error(y_test, base) ** 0.5)
    mae_model = float(mean_absolute_error(y_test, pred))
    mae_base = float(mean_absolute_error(y_test, base))
    return pred, pipe, rmse_model, rmse_base, mae_model, mae_base


def _linear_shap_summary(
    pipe: Pipeline,
    x_test: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    x_scaled = scaler.transform(imputer.transform(x_test))
    coef = np.asarray(model.coef_, dtype=float).reshape(1, -1)
    # Exact SHAP values for a linear model in the standardized feature space.
    # The training background has mean ~0 after StandardScaler, so shap_j = x_std_j * beta_j.
    shap_values = x_scaled * coef
    rows = []
    for j, feature in enumerate(feature_cols):
        values = shap_values[:, j]
        rows.append(
            {
                "feature": feature,
                "mean_shap": float(np.nanmean(values)),
                "mean_abs_shap": float(np.nanmean(np.abs(values))),
                "std_shap": float(np.nanstd(values)),
                "n_values": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_mc_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["target", "activity_number", "activity", "window_size", "scheme_window_size", "model_set"] if col in results.columns]
    def q025(s): return float(np.nanpercentile(s, 2.5)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    def q975(s): return float(np.nanpercentile(s, 97.5)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    summary = (
        results.groupby(group_cols, dropna=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_test", "mean"),
            features=("features", "first"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            std_r2=("r2", "std"),
            p025_r2=("r2", q025),
            p975_r2=("r2", q975),
            prop_r2_gt0=("r2", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_rmse_model=("rmse_model", "mean"),
            mean_rmse_baseline=("rmse_baseline", "mean"),
            mean_delta_rmse=("delta_rmse", "mean"),
            mean_mae_model=("mae_model", "mean"),
            mean_mae_baseline=("mae_baseline", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_pearson_r=("pearson_r", "mean"),
            mean_spearman_r=("spearman_r", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["target", "mean_r2"], ascending=[True, False])


def summarize_shap(shap_by_repeat: pd.DataFrame) -> pd.DataFrame:
    if shap_by_repeat.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["target", "activity_number", "activity", "window_size", "scheme_window_size", "model_set", "feature"] if col in shap_by_repeat.columns]
    out = (
        shap_by_repeat.groupby(group_cols, dropna=False)
        .agg(
            repeats=("repeat", "nunique"),
            mean_shap=("mean_shap", "mean"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_rank_abs_shap=("rank_abs_shap", "mean"),
            prop_top3=("rank_abs_shap", lambda s: float((pd.to_numeric(s, errors="coerce") <= 3).mean())),
            n_values=("n_values", "sum"),
        )
        .reset_index()
    )
    return out.sort_values(["target", "activity_number", "window_size", "model_set", "mean_abs_shap"], ascending=[True, True, True, True, False])


def compare_model_sets(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    keys = [col for col in ["target", "activity_number", "activity", "window_size", "scheme_window_size", "repeat"] if col in results.columns]
    piv = results.pivot_table(index=keys, columns="model_set", values=["r2", "rmse_model", "mae_model"], aggfunc="mean")
    if piv.empty:
        return pd.DataFrame()
    piv.columns = [f"{a}__{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    rows = []
    comparison_pairs = [
        ("school_year_plus_nlp", "school_year_only", "added_nlp_over_school_year"),
        ("school_year_plus_nlp", "nlp_only", "added_school_year_over_nlp"),
        ("nlp_only", "school_year_only", "nlp_vs_school_year"),
    ]
    for plus, base, label in comparison_pairs:
        r2_plus = f"r2__{plus}"
        r2_base = f"r2__{base}"
        if r2_plus not in piv.columns or r2_base not in piv.columns:
            continue
        df = piv[keys].copy()
        df["comparison"] = label
        df["delta_r2"] = piv[r2_plus] - piv[r2_base]
        rmse_plus = f"rmse_model__{plus}"
        rmse_base = f"rmse_model__{base}"
        if rmse_plus in piv.columns and rmse_base in piv.columns:
            df["delta_rmse"] = piv[rmse_base] - piv[rmse_plus]
        mae_plus = f"mae_model__{plus}"
        mae_base = f"mae_model__{base}"
        if mae_plus in piv.columns and mae_base in piv.columns:
            df["delta_mae"] = piv[mae_base] - piv[mae_plus]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    group_cols = [col for col in keys if col != "repeat"] + ["comparison"]
    def q025(s): return float(np.nanpercentile(s, 2.5)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    def q975(s): return float(np.nanpercentile(s, 97.5)) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan
    summary = (
        long.groupby(group_cols, dropna=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_delta_r2=("delta_r2", "mean"),
            median_delta_r2=("delta_r2", "median"),
            p025_delta_r2=("delta_r2", q025),
            p975_delta_r2=("delta_r2", q975),
            prop_delta_r2_gt0=("delta_r2", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_delta_rmse=("delta_rmse", "mean") if "delta_rmse" in long.columns else ("delta_r2", "mean"),
            mean_delta_mae=("delta_mae", "mean") if "delta_mae" in long.columns else ("delta_r2", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["target", "comparison", "mean_delta_r2"], ascending=[True, True, False])


def run_monte_carlo_cv(
    input_csv: Path,
    output_dir: Path,
    targets_text: str = "Total,NPLAN,MOT,COG",
    control_cols_text: str = "School year",
    model_sets_text: str = "nlp,school_year,school_year_nlp",
    n_repeats: int = 400,
    test_size: float = 0.2,
    alphas: list[float] | None = None,
    random_state: int = 42,
    group_col: str = "code",
    min_n: int = 30,
    save_predictions: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    shap_dir = output_dir / "shap"
    models_dir.mkdir(parents=True, exist_ok=True)
    shap_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    target_map = resolve_targets(df, targets_text)
    if not target_map:
        raise ValueError(f"No target columns found for: {targets_text}")
    alphas = alphas or DEFAULT_ALPHAS
    requested_model_sets = parse_csv_list(model_sets_text)
    controls_requested = parse_csv_list(control_cols_text)

    result_rows: list[dict] = []
    pred_rows: list[dict] = []
    shap_rows: list[pd.DataFrame] = []

    groups = _activity_window_groups(df)
    total_models = len(groups) * len(target_map)
    model_counter = 0
    for (_, _), sub in groups:
        controls = resolve_control_columns(sub, controls_requested)
        feature_sets = model_feature_sets(sub, controls, requested_model_sets)
        numeric_cols = sorted(set(sum(feature_sets.values(), [])) | set(target_map.values()))
        sub = _coerce_numeric(sub, numeric_cols)
        if group_col in sub.columns:
            sub = sub.drop_duplicates(subset=[group_col], keep="first").reset_index(drop=True)
        for target_label, target_col in target_map.items():
            model_counter += 1
            context = _context_from_group(sub, target_label)
            print(f"[{model_counter}/{total_models}] {target_label} A{context.get('activity_number')} W{context.get('window_size')}")
            for model_set, feature_cols in feature_sets.items():
                feature_cols = [col for col in feature_cols if col in sub.columns]
                usable = sub[feature_cols + [target_col] + ([group_col] if group_col in sub.columns else [])].copy()
                usable[target_col] = pd.to_numeric(usable[target_col], errors="coerce")
                for col in feature_cols:
                    usable[col] = pd.to_numeric(usable[col], errors="coerce")
                mask = usable[target_col].notna()
                usable = usable.loc[mask].reset_index(drop=True)
                meta = sub.loc[mask].reset_index(drop=True)
                valid_features = [col for col in feature_cols if usable[col].notna().sum() >= 3 and usable[col].nunique(dropna=True) > 1]
                if len(usable) < min_n or usable[target_col].nunique(dropna=True) < 2 or not valid_features:
                    result_rows.append(context | {"model_set": model_set, "repeat": 0, "n_train": 0, "n_test": len(usable), "features": len(valid_features), "r2": np.nan})
                    continue
                x = usable[valid_features]
                y = usable[target_col].astype(float)
                split_groups = meta[group_col] if group_col in meta.columns else None
                splitter, clean_groups = _make_splitter(n_repeats, test_size, random_state, split_groups)
                split_iter = splitter.split(x, y, clean_groups) if clean_groups is not None else splitter.split(x, y)
                for repeat, (train_idx, test_idx) in enumerate(split_iter, start=1):
                    pred, pipe, rmse_model, rmse_base, mae_model, mae_base = _fit_predict_one(x, y, train_idx, test_idx, alphas)
                    y_test = y.iloc[test_idx].to_numpy(dtype=float)
                    r2 = float(r2_score(y_test, pred)) if len(test_idx) >= 2 else np.nan
                    pr, pp = _corr(y_test, pred, method="pearson")
                    sr, sp = _corr(y_test, pred, method="spearman")
                    estimator = pipe.named_steps["model"]
                    chosen_alpha = float(getattr(estimator, "alpha_", np.nan))
                    result_rows.append(
                        context
                        | {
                            "model_set": model_set,
                            "repeat": int(repeat),
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            "features": int(len(valid_features)),
                            "alpha": chosen_alpha,
                            "r2": r2,
                            "rmse_model": rmse_model,
                            "rmse_baseline": rmse_base,
                            "delta_rmse": rmse_base - rmse_model,
                            "mae_model": mae_model,
                            "mae_baseline": mae_base,
                            "delta_mae": mae_base - mae_model,
                            "pearson_r": pr,
                            "pearson_p": pp,
                            "spearman_r": sr,
                            "spearman_p": sp,
                        }
                    )
                    sh = _linear_shap_summary(pipe, x.iloc[test_idx], valid_features)
                    if not sh.empty:
                        sh["rank_abs_shap"] = sh["mean_abs_shap"].rank(ascending=False, method="min")
                        for key, value in (context | {"model_set": model_set, "repeat": int(repeat)}).items():
                            sh[key] = value
                        shap_rows.append(sh)
                    if save_predictions:
                        pred_meta_cols = [col for col in [group_col, "Cod", "Age", "School year", "Educational level", "Gender"] if col in meta.columns]
                        pred_meta = meta.iloc[test_idx][pred_meta_cols].reset_index(drop=True)
                        for i in range(len(test_idx)):
                            pred_rows.append(
                                context
                                | {
                                    "model_set": model_set,
                                    "repeat": int(repeat),
                                    "row_index": int(test_idx[i]),
                                    "y_true": float(y_test[i]),
                                    "y_pred": float(pred[i]),
                                }
                                | {col: pred_meta.iloc[i][col] for col in pred_meta_cols}
                            )

    results = pd.DataFrame(result_rows)
    summary = summarize_mc_results(results)
    comparisons = compare_model_sets(results)
    shap_by_repeat = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    shap_summary = summarize_shap(shap_by_repeat)
    paths = {
        "mc_results_by_repeat": models_dir / "mc_cv_results_by_repeat.csv",
        "mc_summary": models_dir / "mc_cv_summary.csv",
        "mc_comparisons": models_dir / "mc_model_comparisons.csv",
        "shap_by_repeat": shap_dir / "shap_by_repeat.csv",
        "shap_summary": shap_dir / "shap_summary.csv",
        "manifest": output_dir / "mc_cv_manifest.json",
    }
    results.to_csv(paths["mc_results_by_repeat"], index=False)
    summary.to_csv(paths["mc_summary"], index=False)
    comparisons.to_csv(paths["mc_comparisons"], index=False)
    shap_by_repeat.to_csv(paths["shap_by_repeat"], index=False)
    shap_summary.to_csv(paths["shap_summary"], index=False)
    if save_predictions:
        pred_path = models_dir / "mc_cv_predictions.csv"
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        paths["predictions"] = pred_path
    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "targets": target_map,
        "control_cols_requested": controls_requested,
        "model_sets_requested": requested_model_sets,
        "n_repeats": n_repeats,
        "test_size": test_size,
        "alphas": alphas,
        "random_state": random_state,
        "group_col": group_col,
        "min_n": min_n,
        "shap_method": "exact_linear_shap_on_standardized_ridge_features",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Monte Carlo CV summary saved: {paths['mc_summary']}")
    print(f"SHAP summary saved: {paths['shap_summary']}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo CV for SpeechGraph NLP predictors and Barratt dimensions")
    parser.add_argument("--input-csv", default="outputs/02_run/analysis/activity_window_features_02.csv")
    parser.add_argument("--output-dir", default="outputs/02_run")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--control-cols", default="School year")
    parser.add_argument("--model-sets", default="nlp,school_year,school_year_nlp")
    parser.add_argument("--n-repeats", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-col", default="code")
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monte_carlo_cv(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        targets_text=args.targets,
        control_cols_text=args.control_cols,
        model_sets_text=args.model_sets,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        alphas=parse_alphas(args.alphas),
        random_state=args.random_state,
        group_col=args.group_col,
        min_n=args.min_n,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()
