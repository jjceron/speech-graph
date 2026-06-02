from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis import ID_COLUMNS, resolve_targets, safe_corr
from src.graphs import MODEL_METRICS
from src.models.contributions import add_age_group, linear_contributions, schooling_column, summarize_by_column, summarize_population

EXTRA_EXCLUDE = {
    "Cod", "_join_code", "_merge", "file", "level", "activity", "activity_number", "activity_index",
    "start_time", "end_time", "Gender", "School", "School year", "Educational level", "Tipo", "Grupo", "Group", "Age", "Edad",
}


def _metric_pattern(metric: str) -> str:
    return rf"(?:^a\d+_{metric}$|^w\d+_a\d+_(?:mean|std|global)_{metric}$)"


def _is_model_feature(col: str, feature_set: str) -> bool:
    if feature_set == "by_activity":
        return any(re.fullmatch(rf"a\d+_{metric}", col) for metric in MODEL_METRICS)
    if feature_set == "full":
        return any(re.fullmatch(rf"w\d+_a\d+_(?:mean|std|global)_{metric}", col) for metric in MODEL_METRICS)
    return any(re.fullmatch(_metric_pattern(metric), col) for metric in MODEL_METRICS)


def select_features(df: pd.DataFrame, targets: dict[str, str], feature_set: str = "by_activity") -> list[str]:
    excluded = set(ID_COLUMNS) | EXTRA_EXCLUDE | set(targets.values())
    features: list[str] = []
    for col in df.columns:
        if col in excluded or col.startswith("Unnamed"):
            continue
        if not _is_model_feature(col, feature_set):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            features.append(col)
    if not features and feature_set == "by_activity":
        return select_features(df, targets, feature_set="auto")
    return features


def _splitter(groups: pd.Series, n_rows: int, n_splits: int):
    clean = groups.astype(str).fillna("")
    unique = clean.nunique(dropna=True)
    if unique >= 2:
        return GroupKFold(n_splits=min(n_splits, unique)), clean
    return KFold(n_splits=min(n_splits, n_rows), shuffle=True, random_state=42), None


def _pipeline(model_type: str, alpha: float, alphas: list[float] | None) -> Pipeline:
    if model_type == "linear":
        estimator = LinearRegression()
    elif alphas:
        estimator = RidgeCV(alphas=np.asarray(alphas, dtype=float))
    else:
        estimator = Ridge(alpha=alpha)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def cross_validated_model(
    df: pd.DataFrame,
    target_label: str,
    target_col: str,
    feature_cols: list[str],
    group_col: str,
    n_splits: int,
    model_type: str,
    alpha: float,
    alphas: list[float] | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable_cols = feature_cols + [target_col] + ([group_col] if group_col in df.columns else [])
    usable = df[usable_cols].copy()
    usable[target_col] = pd.to_numeric(usable[target_col], errors="coerce")
    for col in feature_cols:
        usable[col] = pd.to_numeric(usable[col], errors="coerce")
    mask = usable[target_col].notna()
    work = df.loc[mask].reset_index(drop=True)
    usable = usable.loc[mask].reset_index(drop=True)
    if len(usable) < 4 or usable[target_col].nunique(dropna=True) < 2:
        result = {"target": target_label, "target_column": target_col, "n_subjects": int(len(usable)), "features": len(feature_cols), "r2": np.nan}
        return result, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    groups = work[group_col] if group_col in work.columns else pd.Series(range(len(work)))
    splitter, split_groups = _splitter(groups, len(work), n_splits)
    x = usable[feature_cols]
    y = usable[target_col].astype(float)
    predictions = np.full(len(work), np.nan)
    coef_rows: list[dict] = []
    contrib_frames: list[pd.DataFrame] = []

    split_iter = splitter.split(x, y, split_groups) if split_groups is not None else splitter.split(x, y)
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        pipe = _pipeline(model_type, alpha, alphas)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(x.iloc[test_idx])
        predictions[test_idx] = pred
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        estimator = pipe.named_steps["model"]
        x_test_scaled = scaler.transform(imputer.transform(x.iloc[test_idx]))
        coefficients = np.asarray(estimator.coef_, dtype=float)
        chosen_alpha = float(getattr(estimator, "alpha_", alpha if model_type == "ridge" else np.nan))
        for feature, coefficient in zip(feature_cols, coefficients):
            coef_rows.append({"target": target_label, "fold": fold, "feature": feature, "coefficient": float(coefficient), "alpha": chosen_alpha})
        contrib_frames.append(linear_contributions(x_test_scaled, coefficients, feature_cols, work.iloc[test_idx].reset_index(drop=True), target_label, fold))

    meta_cols = [col for col in ["code", "Cod", "Age", "Edad", "School year", "Educational level", "Escolaridad", "Gender", "Tipo"] if col in work.columns]
    pred_df = work[meta_cols].copy()
    pred_df["target"] = target_label
    pred_df["target_column"] = target_col
    pred_df["y_true"] = y.to_numpy()
    pred_df["y_pred"] = predictions
    r2 = r2_score(y, predictions) if np.isfinite(predictions).all() else np.nan
    pearson_r, pearson_p, _ = safe_corr(pred_df["y_true"], pred_df["y_pred"], method="pearson")
    spearman_r, spearman_p, _ = safe_corr(pred_df["y_true"], pred_df["y_pred"], method="spearman")
    result = {
        "target": target_label,
        "target_column": target_col,
        "n_subjects": int(len(work)),
        "features": int(len(feature_cols)),
        "model": model_type,
        "alpha": float(alpha) if model_type == "ridge" and not alphas else np.nan,
        "r2": float(r2),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
    return result, pred_df, pd.DataFrame(coef_rows), pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame()


def parse_alphas(text: str | None) -> list[float] | None:
    values = [float(part.strip()) for part in str(text or "").split(",") if part.strip()]
    return values or None


def run_linear_regression(
    input_csv: Path,
    run_dir: Path,
    targets: str = "Total,NPLAN,MOT,COG",
    group_col: str = "code",
    n_splits: int = 5,
    feature_set: str = "by_activity",
    model_type: str = "ridge",
    alpha: float = 1.0,
    alphas: list[float] | None = None,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    interp_dir = run_dir / "interpretability"
    models_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    target_map = resolve_targets(df, targets)
    if not target_map:
        raise ValueError(f"None of the requested targets were found: {targets}")
    feature_cols = select_features(df, target_map, feature_set=feature_set)
    if not feature_cols:
        raise ValueError(f"No usable features found for feature_set={feature_set}")

    results: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    coef_frames: list[pd.DataFrame] = []
    contrib_frames: list[pd.DataFrame] = []
    for target_label, target_col in target_map.items():
        result, predictions, coefficients, contributions = cross_validated_model(
            df, target_label, target_col, feature_cols, group_col, n_splits, model_type, alpha, alphas
        )
        results.append(result)
        if not predictions.empty:
            pred_frames.append(predictions)
        if not coefficients.empty:
            coef_frames.append(coefficients)
        if not contributions.empty:
            contrib_frames.append(contributions)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    coefficients_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    contributions_df = pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame()
    contributions_df, age_group_col = add_age_group(contributions_df)
    schooling_col = schooling_column(contributions_df)

    paths = {
        "results": models_dir / "linear_cv_results.csv",
        "predictions": models_dir / "linear_cv_predictions.csv",
        "coefficients": models_dir / "linear_cv_coefficients.csv",
        "contributions": interp_dir / "feature_contributions_rows.csv",
        "population": interp_dir / "feature_relevance_population.csv",
        "age": interp_dir / "feature_relevance_by_age.csv",
        "schooling": interp_dir / "feature_relevance_by_schooling.csv",
    }
    results_df.to_csv(paths["results"], index=False)
    predictions_df.to_csv(paths["predictions"], index=False)
    coefficients_df.to_csv(paths["coefficients"], index=False)
    contributions_df.to_csv(paths["contributions"], index=False)
    summarize_population(contributions_df).to_csv(paths["population"], index=False)
    summarize_by_column(contributions_df, age_group_col).to_csv(paths["age"], index=False)
    summarize_by_column(contributions_df, schooling_col).to_csv(paths["schooling"], index=False)

    print("Linear regression completed.")
    print(results_df.to_string(index=False))
    print(f"Feature set: {feature_set}. Features used: {len(feature_cols)}")
    print(f"Model outputs: {models_dir}")
    print(f"Interpretability outputs: {interp_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe linear regressors for Barratt dimensions")
    parser.add_argument("--input-csv", default="outputs/01_run/analysis/subject_level_features.csv")
    parser.add_argument("--run-dir", default="outputs/01_run")
    parser.add_argument("--output-dir", default="", help="Deprecated alias. Use --run-dir.")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--group-col", default="code")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--feature-set", default="by_activity", choices=["by_activity", "full", "auto"])
    parser.add_argument("--model", default="ridge", choices=["ridge", "linear"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alphas", default="", help="Optional comma-separated alphas for RidgeCV inside each fold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir)
    run_linear_regression(
        input_csv=Path(args.input_csv),
        run_dir=run_dir,
        targets=args.targets,
        group_col=args.group_col,
        n_splits=args.n_splits,
        feature_set=args.feature_set,
        model_type=args.model,
        alpha=args.alpha,
        alphas=parse_alphas(args.alphas),
    )


if __name__ == "__main__":
    main()
