from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.stats import BARRATT_TARGETS, build_subject_level_features, correlations_subject_level, demographic_columns, resolve_target_columns
from src.models.interpretability import (
    contribution_frame,
    make_age_groups,
    make_schooling_groups,
    summarize_group,
    summarize_population,
)


NON_FEATURE_COLUMNS = {
    "code", "file", "level", "activity", "activity_number", "activity_index", "start_time", "end_time",
    "window_size", "window_step", "scheme_window_size", "random_times", "token_count", "segment_count",
    "window_count", "valid_window", "_merge", "_join_code", "Cod",
    *BARRATT_TARGETS,
}
BAD_FEATURE_PATTERNS = (
    "window_start", "window_end", "window_index", "segment_index", "window_size_actual",
    "token_count", "unique_edges", "clustering", "repeated_edges", "ratio", "parallel_edges",
)


def _numeric_feature_columns(df: pd.DataFrame, targets: dict[str, str]) -> list[str]:
    exclude = set(NON_FEATURE_COLUMNS) | set(targets.values()) | set(demographic_columns(df).values())
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        low = col.lower()
        if any(pattern in low for pattern in BAD_FEATURE_PATTERNS):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 5 and values.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def _model(model_type: str, alpha: float) -> Pipeline:
    estimator = Ridge(alpha=alpha) if model_type == "ridge" else LinearRegression()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan, np.nan
    if method == "pearson":
        return tuple(map(float, stats.pearsonr(x[mask], y[mask])))
    return tuple(map(float, stats.spearmanr(x[mask], y[mask])))


def _cv_splits(n: int, n_splits: int) -> KFold:
    if n < 4:
        raise ValueError("At least 4 subjects are required for cross-validation.")
    return KFold(n_splits=min(n_splits, n), shuffle=True, random_state=42)


def _prepare_subject_table(input_csv: Path, targets_text: str) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(input_csv)
    targets = resolve_target_columns(df, targets_text)
    if not targets:
        raise ValueError(f"No requested target columns found. Requested: {targets_text}")
    if "activity_number" in df.columns and "scheme_window_size" in df.columns:
        subject = build_subject_level_features(df, targets)
        targets = resolve_target_columns(subject, targets.keys())
        return subject, targets
    if "code" not in df.columns:
        raise ValueError("Input table must include a code column.")
    return df.drop_duplicates("code").copy(), targets


def _fit_target(
    df: pd.DataFrame,
    target_name: str,
    target_col: str,
    feature_cols: list[str],
    n_splits: int,
    model_type: str,
    alpha: float,
    save_row_contributions: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df[["code", target_col, *feature_cols, *demographic_columns(df).values()]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[target_col]).reset_index(drop=True)
    if len(work) < 4 or work[target_col].nunique(dropna=True) < 2:
        raise ValueError(f"Target {target_name} has insufficient data after filtering.")

    x = work[feature_cols]
    y = work[target_col].astype(float)
    splitter = _cv_splits(len(work), n_splits)
    predictions = np.full(len(work), np.nan)
    coefficients_rows: list[dict] = []
    contribution_frames: list[pd.DataFrame] = []

    metadata = work[[c for c in ["code", *demographic_columns(work).values()] if c in work.columns]].copy()
    metadata["age_group"] = make_age_groups(work)
    metadata["schooling_group"] = make_schooling_groups(work)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        pipe = _model(model_type, alpha)
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(x.iloc[test_idx])
        predictions[test_idx] = pred

        estimator = pipe.named_steps["model"]
        coefficients = np.asarray(estimator.coef_, dtype=float)
        for feature, coef in zip(feature_cols, coefficients):
            coefficients_rows.append({"target": target_name, "fold": fold, "feature": feature, "coefficient": float(coef)})

        imputed = pipe.named_steps["imputer"].transform(x.iloc[test_idx])
        scaled = pipe.named_steps["scaler"].transform(imputed)
        contribution_frames.append(
            contribution_frame(
                scaled,
                coefficients,
                feature_cols,
                metadata.iloc[test_idx].reset_index(drop=True),
                target_name,
                fold,
            )
        )

    pred_df = metadata[["code", "age_group", "schooling_group"]].copy()
    pred_df["target"] = target_name
    pred_df["target_column"] = target_col
    pred_df["y_true"] = y.to_numpy()
    pred_df["y_pred"] = predictions

    pearson_r, pearson_p = _safe_corr(pred_df["y_true"], pred_df["y_pred"], "pearson")
    spearman_r, spearman_p = _safe_corr(pred_df["y_true"], pred_df["y_pred"], "spearman")
    result = {
        "target": target_name,
        "target_column": target_col,
        "n_subjects": int(len(work)),
        "features": int(len(feature_cols)),
        "model": model_type,
        "alpha": alpha if model_type == "ridge" else np.nan,
        "r2": float(r2_score(y, predictions)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
    coefficients_df = pd.DataFrame(coefficients_rows)
    contrib_df = pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame()
    if not save_row_contributions:
        row_contrib = pd.DataFrame()
    else:
        row_contrib = contrib_df
    return result, pred_df, coefficients_df, contrib_df if not contrib_df.empty else row_contrib


def run_linear_regression(
    input_csv: Path,
    output_dir: Path | None = None,
    run_dir: Path | None = None,
    targets: str = "Total,NPLAN,MOT,COG",
    n_splits: int = 5,
    model_type: str = "ridge",
    alpha: float = 1.0,
    save_row_contributions: bool = False,
) -> dict[str, Path]:
    if run_dir is None:
        run_dir = input_csv.parent.parent if input_csv.parent.name == "analysis" else input_csv.parent
    if output_dir is None:
        output_dir = run_dir / "models"
    interpretability_dir = run_dir / "interpretability"
    analysis_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    interpretability_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    subject, target_map = _prepare_subject_table(input_csv, targets)
    subject.to_csv(analysis_dir / "subject_level_features_for_model.csv", index=False)
    correlations_subject_level(subject, target_map).to_csv(analysis_dir / "correlations_subject_level_model_input.csv", index=False)
    feature_cols = _numeric_feature_columns(subject, target_map)
    if not feature_cols:
        raise ValueError("No usable NLP feature columns found after filtering.")

    results: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    coef_frames: list[pd.DataFrame] = []
    contrib_frames: list[pd.DataFrame] = []

    for target, target_col in target_map.items():
        result, pred, coef, contrib = _fit_target(subject, target, target_col, feature_cols, n_splits, model_type, alpha, save_row_contributions)
        results.append(result)
        pred_frames.append(pred)
        coef_frames.append(coef)
        contrib_frames.append(contrib)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    coefficients_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    contributions_df = pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame()

    paths = {
        "results": output_dir / "linear_cv_results.csv",
        "predictions": output_dir / "linear_cv_predictions.csv",
        "coefficients": output_dir / "linear_cv_coefficients.csv",
        "population": interpretability_dir / "contributions_population.csv",
        "age": interpretability_dir / "contributions_by_age.csv",
        "schooling": interpretability_dir / "contributions_by_schooling.csv",
    }
    results_df.to_csv(paths["results"], index=False)
    predictions_df.to_csv(paths["predictions"], index=False)
    coefficients_df.to_csv(paths["coefficients"], index=False)
    summarize_population(contributions_df).to_csv(paths["population"], index=False)
    summarize_group(contributions_df, "age_group").to_csv(paths["age"], index=False)
    summarize_group(contributions_df, "schooling_group").to_csv(paths["schooling"], index=False)
    if save_row_contributions:
        paths["rows"] = interpretability_dir / "contributions_rows.csv"
        contributions_df.to_csv(paths["rows"], index=False)

    print("Linear regression completed.")
    print(results_df.to_string(index=False))
    print(f"Model outputs: {output_dir}")
    print(f"Interpretability outputs: {interpretability_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-level leakage-safe linear regressors for Barratt dimensions")
    parser.add_argument("--input-csv", default="outputs/01_run/analysis/subject_level_features.csv")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--targets", default="Total,NPLAN,MOT,COG")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model", default="ridge", choices=["ridge", "linear"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--save-row-contributions", action="store_true")
    parser.add_argument("--group-col", default="code", help="Accepted for backwards compatibility; subject-level CV uses one row per subject.")
    parser.add_argument("--feature-mode", default="standard", help="Accepted for backwards compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_linear_regression(
        input_csv=Path(args.input_csv),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        targets=args.targets,
        n_splits=args.n_splits,
        model_type=args.model,
        alpha=args.alpha,
        save_row_contributions=args.save_row_contributions,
    )


if __name__ == "__main__":
    main()
