"""Optuna regression experiments with configurable RFE for speech-graph features.

The model is selected only with validation metrics. Test metrics are computed
once for the best validation trial and saved as the final report.

Usage:
    py -m src.analysis.regression_optuna --task 2 --window 10 --experiment raw --rfe fixed --targets all --optimize mae --n-trials 300 --n-iter 400 --optimize-splits 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from functools import partial

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:sklearn.utils.parallel"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=Warning, module="xgboost")
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import optuna
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, QuantileRegressor, Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]

from src.analysis import experiment_config as expcfg

try:
    from sklearn.metrics import d2_absolute_error_score
except Exception:  # pragma: no cover
    d2_absolute_error_score = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


METRIC_NAMES = ["mae", "d2mae", "rmse", "mape", "r2", "median_ae", "max_error", "rho"]
LOWER_IS_BETTER = {"mae", "rmse", "mape", "median_ae", "max_error"}
HIGHER_IS_BETTER = {"d2mae", "r2", "rho"}

ALL_REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "QuantileRegressor",
    "SVR",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "BaggingRegressor",
    "StackingRegressor",
    "GaussianProcessRegressor",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "XGBRegressor",
]

DEFAULT_REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "QuantileRegressor",
    "SVR",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "BaggingRegressor",
    "StackingRegressor",
    "GaussianProcessRegressor",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
    "XGBRegressor",
]


class TrialTimeout(Exception):
    pass


def setup_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "experiment.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrialProgressCallback:
    """Reports optimization progress every `report_every` trials."""
    def __init__(self, n_trials: int, report_every: int = 25) -> None:
        self.n_trials = n_trials
        self.report_every = report_every
        self._start: float | None = None

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self._start is None:
            self._start = time.time()
        if trial.number == 0 or trial.number % self.report_every != 0:
            return
        done = trial.number + 1
        best = study.best_value
        bt = study.best_trial.number
        elapsed = time.time() - self._start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (self.n_trials - done) / rate if rate > 0 else 0
        logger.info(
            "Trial %4d/%d | best: %.6f (trial %d) | elapsed: %s | remaining: %s",
            done, self.n_trials, best, bt,
            _format_duration(elapsed),
            _format_duration(remaining),
        )


class EarlyStoppingCallback:
    """Stops the study when best objective does not improve for `patience` trials."""
    def __init__(self, patience: int = 75, min_improvement: float = 1e-4) -> None:
        self.patience = patience
        self.min_improvement = min_improvement
        self._best_value: float | None = None
        self._trials_without_improvement = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number == 0 or study.best_value is None or self._best_value is None:
            self._best_value = study.best_value
            self._trials_without_improvement = 0
            return
        improvement = self._best_value - study.best_value if study.direction == optuna.study.StudyDirection.MINIMIZE else study.best_value - self._best_value
        if improvement > self.min_improvement:
            self._best_value = study.best_value
            self._trials_without_improvement = 0
        else:
            self._trials_without_improvement += 1
        if self._trials_without_improvement >= self.patience:
            logger.info(
                "Early stopping after %d trials without improvement | best: %.6f (trial %d)",
                self.patience, study.best_value, study.best_trial.number,
            )
            study.stop()


def _format_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def as_1d(y: Any) -> np.ndarray:
    return np.asarray(y).reshape(-1)


def safe_d2mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """D2-MAE compares absolute error against a median baseline."""
    if d2_absolute_error_score is not None:
        return float(d2_absolute_error_score(y_true, y_pred))
    denominator = np.sum(np.abs(y_true - np.median(y_true)))
    if denominator == 0:
        return np.nan
    return float(1.0 - np.sum(np.abs(y_true - y_pred)) / denominator)


def calculate_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Regression metrics used for model comparison and reporting."""
    yt = as_1d(y_true).astype(float)
    yp = as_1d(y_pred).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(yt, yp)

    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "d2mae": safe_d2mae(yt, yp),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": float(mean_absolute_percentage_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "median_ae": float(median_absolute_error(yt, yp)),
        "max_error": float(max_error(yt, yp)),
        "rho": float(rho) if np.isfinite(rho) else np.nan,
    }


def summarize_metric_rows(rows: list[dict[str, float]], suffix: str) -> dict[str, float]:
    """Mean, standard deviation and 95% t interval per metric."""
    df = pd.DataFrame(rows)
    out: dict[str, float] = {f"n_splits_{suffix}": len(df)}

    for metric in METRIC_NAMES:
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
        if len(values) == 0:
            mean = std = ci = t_crit = dfree = np.nan
        elif len(values) == 1:
            mean = float(values[0])
            std = ci = t_crit = np.nan
            dfree = 0
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            t_crit = float(sp_stats.t.ppf(0.975, df=len(values) - 1))
            ci = float(t_crit * sp_stats.sem(values))
            dfree = len(values) - 1

        out[f"{metric}_mean_{suffix}"] = mean
        out[f"{metric}_std_{suffix}"] = std
        out[f"{metric}_ci_lower_{suffix}"] = mean - ci if np.isfinite(ci) else np.nan
        out[f"{metric}_ci_upper_{suffix}"] = mean + ci if np.isfinite(ci) else np.nan
        out[f"{metric}_samperror_{suffix}"] = ci
        out[f"{metric}_t_critical_{suffix}"] = t_crit
        out[f"{metric}_df_{suffix}"] = dfree

    out[f"r2_below_zero_{suffix}"] = float((df["r2"] < 0).mean()) if "r2" in df else np.nan
    return out


def metric_to_objective(value: float, metric: str) -> float:
    if metric in LOWER_IS_BETTER:
        return float(value)
    if metric in HIGHER_IS_BETTER:
        return -float(value)
    raise ValueError(f"Unknown optimization metric: {metric}")


def objective_direction(metric: str) -> str:
    return "minimize"


def get_regressor(trial: optuna.trial.BaseTrial, name: str, random_state: int):
    """Return the final regressor and whether SVR is non-linear."""
    is_svr_non_linear = False

    if name == "LinearRegression":
        reg = LinearRegression()

    elif name == "Ridge":
        reg = Ridge(
            alpha=trial.suggest_float("ridge_alpha", 1e-1, 1),
            solver=trial.suggest_categorical("ridge_solver", ["auto", "svd", "lsqr", "sparse_cg"]),
            random_state=random_state,
        )

    elif name == "ElasticNet":
        reg = ElasticNet(
            alpha=trial.suggest_float("elastic_alpha", 1e-1, 1),
            l1_ratio=trial.suggest_categorical("elastic_l1_ratio", [0, 0.5, 1]),
            random_state=random_state,
            max_iter=20000,
        )

    elif name == "QuantileRegressor":
        reg = QuantileRegressor(
            quantile=0.5,
            alpha=trial.suggest_float("quantile_alpha", 1e-1, 1),
            solver="highs",
        )

    elif name == "SVR":
        kernel = trial.suggest_categorical("svr_kernel", ["linear", "rbf", "poly"])
        reg = SVR(
            C=trial.suggest_float("svr_C", 1e-3, 100.0, log=True),
            epsilon=trial.suggest_float("svr_epsilon", 1e-3, 100.0, log=True),
            kernel=kernel,
            degree=trial.suggest_int("svr_degree", 2, 5) if kernel == "poly" else 3,
            gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
        )
        is_svr_non_linear = kernel != "linear"

    elif name == "RandomForestRegressor":
        reg = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 10, 200),
            max_depth=trial.suggest_int("rf_max_depth", 2, 8),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 8),
            random_state=random_state,
            n_jobs=1,
        )

    elif name == "ExtraTreesRegressor":
        reg = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 50, 500),
            max_depth=trial.suggest_int("et_max_depth", 2, 20),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 8),
            criterion="friedman_mse",
            random_state=random_state,
            n_jobs=1,
        )

    elif name == "BaggingRegressor":
        reg = BaggingRegressor(
            n_estimators=trial.suggest_int("bag_n_estimators", 50, 500),
            random_state=random_state,
            n_jobs=1,
        )

    elif name == "StackingRegressor":
        final_name = trial.suggest_categorical(
            "stack_final_estimator", ["Ridge", "RandomForestRegressor", "ExtraTreesRegressor", "LinearRegression"]
        )
        if final_name == "Ridge":
            final_estimator = Ridge(random_state=random_state)
        elif final_name == "RandomForestRegressor":
            final_estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
        elif final_name == "ExtraTreesRegressor":
            final_estimator = ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
        else:
            final_estimator = LinearRegression()
        reg = StackingRegressor(
            estimators=[("ridge", Ridge(random_state=random_state)), ("svr", SVR(kernel="linear"))],
            final_estimator=final_estimator,
            cv=None,
        )

    elif name == "GaussianProcessRegressor":
        reg = GaussianProcessRegressor(
            alpha=trial.suggest_float("gpr_alpha", 1e-10, 1e-1, log=True),
            random_state=random_state,
        )

    elif name == "KNeighborsRegressor":
        reg = KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 1, 20),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("knn_metric", ["euclidean", "manhattan", "minkowski"]),
        )

    elif name == "DecisionTreeRegressor":
        reg = DecisionTreeRegressor(
            max_depth=trial.suggest_int("dt_max_depth", 2, 20),
            min_samples_split=trial.suggest_int("dt_min_samples_split", 2, 20),
            random_state=random_state,
        )

    elif name == "XGBRegressor":
        if XGBRegressor is None:
            raise optuna.TrialPruned("xgboost is not installed")
        reg = XGBRegressor(
            n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 10),
            learning_rate=trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0, step=0.1),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0, step=0.1),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
            booster=trial.suggest_categorical("xgb_booster", ["gbtree", "dart", "gblinear"]),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )

    else:
        raise ValueError(f"Unknown regressor: {name}")

    return reg, is_svr_non_linear


def get_rfe_estimator(
    trial: optuna.trial.BaseTrial,
    regressor_name: str,
    final_regressor: Any,
    is_svr_non_linear: bool,
    random_state: int,
):
    """Use the final model for RFE when possible, otherwise use a proxy."""
    incompatible = {
        "KNeighborsRegressor",
        "GaussianProcessRegressor",
        "BaggingRegressor",
        "StackingRegressor",
    }
    if regressor_name not in incompatible and not is_svr_non_linear:
        return clone(final_regressor), {"mode": "original", "type": regressor_name}

    proxy = trial.suggest_categorical("rfe_proxy_type", ["ExtraTrees", "LinearSVC"])
    if proxy == "ExtraTrees":
        rfe_et_n_est = trial.suggest_int("rfe_proxy_et_n_estimators", 20, 100, step=10)
        rfe_et_depth = trial.suggest_int("rfe_proxy_et_max_depth", 3, 10)
        rfe_et_min_samples = trial.suggest_int("rfe_proxy_et_min_samples_split", 2, 8)
        estimator = ExtraTreesRegressor(
            n_estimators=rfe_et_n_est,
            max_depth=rfe_et_depth,
            min_samples_split=rfe_et_min_samples,
            criterion="friedman_mse",
            random_state=random_state,
            n_jobs=1,
        )
        rfe_params = {
            "mode": "proxy", "type": "ExtraTrees",
            "n_estimators": rfe_et_n_est,
            "max_depth": rfe_et_depth,
            "min_samples_split": rfe_et_min_samples,
        }
    else:
        rfe_svc_C = trial.suggest_float("rfe_proxy_svc_C", 1e-3, 10.0, log=True)
        rfe_svr_epsilon = trial.suggest_float("svr_epsilon", 1e-3, 100, log=True)
        estimator = SVR(C=rfe_svc_C, kernel="linear", epsilon=rfe_svr_epsilon)
        rfe_params = {
            "mode": "proxy", "type": "LinearSVC",
            "C": rfe_svc_C,
            "epsilon": rfe_svr_epsilon,
        }
    return estimator, rfe_params


def fit_global_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    trial: optuna.trial.BaseTrial,
    rfe_estimator: Any,
    use_scaler: bool,
    max_rfe_features: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit imputation, optional scaling and RFE before MMCV."""
    n_candidates = X.shape[1]
    if n_candidates < 3:
        raise optuna.TrialPruned("Need at least 3 candidate features for RFE")

    upper = min(n_candidates - 1, max_rfe_features or n_candidates - 1)
    n_features = trial.suggest_int("rfe_n_features", 2, upper)

    imputer = SimpleImputer(strategy="mean")
    X_proc = imputer.fit_transform(X)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_proc)

    selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features, step=1)
    selector.fit(X_proc, as_1d(y))

    selected_features = X.columns[selector.support_].tolist()
    selected_ranks = pd.DataFrame(
        {
            "feature": X.columns,
            "selected": selector.support_,
            "ranking": selector.ranking_,
        }
    ).sort_values(["selected", "ranking", "feature"], ascending=[False, True, True])

    X_selected = pd.DataFrame(X_proc[:, selector.support_], columns=selected_features, index=X.index)
    metadata = {
        "n_candidate_features": int(n_candidates),
        "rfe_n_features": int(n_features),
        "rfe_selected_features": selected_features,
        "rfe_ranking": selected_ranks,
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
    }
    return X_selected, metadata


def make_final_pipeline(regressor: Any) -> Pipeline:
    return Pipeline([("regressor", regressor)])


def build_trial_artifacts(
    trial: optuna.trial.BaseTrial,
    X: pd.DataFrame,
    y: pd.Series,
    regressors: list[str],
    random_state: int,
    max_rfe_features: int | None,
    train_idx: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Pipeline, dict[str, Any]]:
    regressor_name = trial.suggest_categorical("regressor", regressors)

    force_scaler = {"SVR", "KNeighborsRegressor", "GaussianProcessRegressor"}
    if regressor_name in force_scaler:
        use_scaler = True
    else:
        use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    regressor, is_svr_non_linear = get_regressor(trial, regressor_name, random_state)
    rfe_estimator, rfe_info = get_rfe_estimator(
        trial, regressor_name, regressor, is_svr_non_linear, random_state
    )

    if train_idx is not None:
        X_fit = X.iloc[train_idx]
        y_fit = y.iloc[train_idx]
    else:
        X_fit = X
        y_fit = y

    X_model, rfe_meta = fit_global_rfe(
        X=X_fit,
        y=y_fit,
        trial=trial,
        rfe_estimator=rfe_estimator,
        use_scaler=use_scaler,
        max_rfe_features=max_rfe_features,
    )

    if train_idx is not None:
        X_all_proc = rfe_meta["imputer"].transform(X)
        if rfe_meta["scaler"] is not None:
            X_all_proc = rfe_meta["scaler"].transform(X_all_proc)
        X_model = pd.DataFrame(
            X_all_proc[:, rfe_meta["selector"].support_],
            columns=rfe_meta["rfe_selected_features"],
            index=X.index,
        )

    pipeline = make_final_pipeline(regressor)
    metadata = {
        "regressor": regressor_name,
        "use_scaler": bool(use_scaler),
        "rfe_estimator": rfe_info,
        **rfe_meta,
    }
    return X_model, pipeline, metadata


def objective_regression(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    optimize_metric: str,
    regressors: list[str],
    timeout_sec: int | None,
    random_state: int,
    max_rfe_features: int | None,
    train_idx: np.ndarray | None = None,
) -> float:
    start = time.time()
    X_model, pipeline, metadata = build_trial_artifacts(
        trial=trial,
        X=X,
        y=y,
        regressors=regressors,
        random_state=random_state,
        max_rfe_features=max_rfe_features,
        train_idx=train_idx,
    )

    trial.set_user_attr("regressor_final", metadata["regressor"])
    trial.set_user_attr("use_scaler_final", metadata["use_scaler"])
    trial.set_user_attr("rfe_clf", metadata["rfe_estimator"])
    trial.set_user_attr("rfe_n_features", metadata["rfe_n_features"])
    trial.set_user_attr("rfe_selected_features", metadata["rfe_selected_features"])

    validation_rows: list[dict[str, float]] = []

    for step, (train_idx, val_idx, _test_idx) in enumerate(splits, start=1):
        model = clone(pipeline)
        model.fit(X_model.iloc[train_idx], as_1d(y.iloc[train_idx]))
        val_pred = model.predict(X_model.iloc[val_idx])
        validation_rows.append(calculate_metrics(y.iloc[val_idx], val_pred))

        partial = float(pd.DataFrame(validation_rows)["d2mae"].mean())
        trial.report(partial, step)

        if trial.should_prune():
            trial.set_user_attr("termination_reason", "pruned")
            raise optuna.TrialPruned()
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            trial.set_user_attr("termination_reason", "timeout")
            raise optuna.TrialPruned()

    validation_summary = summarize_metric_rows(validation_rows, "val")
    for key, value in validation_summary.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("termination_reason", "completed")

    return metric_to_objective(validation_summary[f"{optimize_metric}_mean_val"], optimize_metric)


def objective_regression_split_rfe(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    optimize_metric: str,
    regressors: list[str],
    timeout_sec: int | None,
    random_state: int,
    max_rfe_features: int | None,
) -> float:
    start = time.time()

    regressor_name = trial.suggest_categorical("regressor", regressors)

    force_scaler = {"SVR", "KNeighborsRegressor", "GaussianProcessRegressor"}
    if regressor_name in force_scaler:
        use_scaler = True
    else:
        use_scaler = trial.suggest_categorical("use_scaler", [True, False])

    regressor, is_svr_non_linear = get_regressor(trial, regressor_name, random_state)

    n_candidates = X.shape[1]
    if n_candidates < 3:
        raise optuna.TrialPruned("Need at least 3 candidate features for RFE")
    upper = min(n_candidates - 1, max_rfe_features or n_candidates - 1)
    n_features = trial.suggest_int("rfe_n_features", 2, upper)

    trial.set_user_attr("regressor_final", regressor_name)
    trial.set_user_attr("use_scaler_final", use_scaler)
    trial.set_user_attr("rfe_n_features", n_features)

    validation_rows: list[dict[str, float]] = []

    for step, (train_idx, val_idx, _test_idx) in enumerate(splits, start=1):
        X_train_raw = X.iloc[train_idx]
        y_train = as_1d(y.iloc[train_idx])
        X_val_raw = X.iloc[val_idx]
        y_val = as_1d(y.iloc[val_idx])

        imputer = SimpleImputer(strategy="mean")
        X_tr = imputer.fit_transform(X_train_raw)
        X_va = imputer.transform(X_val_raw)

        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        rfe_estimator, _ = get_rfe_estimator(
            trial, regressor_name, regressor, is_svr_non_linear, random_state
        )
        step_size = max(1, n_candidates // 6)
        selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features, step=step_size)
        selector.fit(X_tr, y_train)

        X_tr_sel = X_tr[:, selector.support_]
        X_va_sel = X_va[:, selector.support_]

        model = clone(make_final_pipeline(regressor))
        model.fit(X_tr_sel, y_train)
        val_pred = as_1d(model.predict(X_va_sel))
        validation_rows.append(calculate_metrics(y_val, val_pred))

        partial = float(pd.DataFrame(validation_rows)["d2mae"].mean())
        trial.report(partial, step)

        if trial.should_prune():
            trial.set_user_attr("termination_reason", "pruned")
            raise optuna.TrialPruned()
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            trial.set_user_attr("termination_reason", "timeout")
            raise optuna.TrialPruned()

    validation_summary = summarize_metric_rows(validation_rows, "val")
    for key, value in validation_summary.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("termination_reason", "completed")

    return metric_to_objective(validation_summary[f"{optimize_metric}_mean_val"], optimize_metric)


def evaluate_fixed_trial(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    regressors: list[str],
    random_state: int,
    max_rfe_features: int | None,
    train_idx: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the best validation configuration on validation and test splits."""
    fixed_trial = optuna.trial.FixedTrial(params)
    X_model, pipeline, metadata = build_trial_artifacts(
        trial=fixed_trial,
        X=X,
        y=y,
        regressors=regressors,
        random_state=random_state,
        max_rfe_features=max_rfe_features,
        train_idx=train_idx,
    )

    validation_rows: list[dict[str, float]] = []
    test_rows: list[dict[str, float]] = []
    pred_rows: list[dict[str, Any]] = []

    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        model = clone(pipeline)
        model.fit(X_model.iloc[train_idx], as_1d(y.iloc[train_idx]))

        for set_name, indices, rows in [
            ("VALIDATION", val_idx, validation_rows),
            ("TEST", test_idx, test_rows),
        ]:
            pred = as_1d(model.predict(X_model.iloc[indices]))
            truth = as_1d(y.iloc[indices])
            rows.append(calculate_metrics(truth, pred))
            for subject, yt, yp in zip(X_model.index[indices], truth, pred):
                pred_rows.append(
                    {
                        "split": split_id,
                        "subject": subject,
                        "set": set_name,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    }
                )

    return {
        "validation_iterations": pd.DataFrame(validation_rows),
        "test_iterations": pd.DataFrame(test_rows),
        "validation_summary": summarize_metric_rows(validation_rows, "val"),
        "test_summary": summarize_metric_rows(test_rows, "test"),
        "predictions": pd.DataFrame(pred_rows),
        "selected_features": metadata["rfe_selected_features"],
        "rfe_ranking": metadata["rfe_ranking"],
        "metadata": {k: v for k, v in metadata.items() if k not in {"imputer", "scaler", "selector", "rfe_ranking"}},
    }


def evaluate_fixed_trial_split_rfe(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    regressors: list[str],
    random_state: int,
    max_rfe_features: int | None,
) -> dict[str, Any]:
    fixed_trial = optuna.trial.FixedTrial(params)
    regressor_name = fixed_trial.params["regressor"]
    use_scaler = fixed_trial.params["use_scaler"]
    if regressor_name in {"SVR", "KNeighborsRegressor", "GaussianProcessRegressor"}:
        use_scaler = True

    regressor, is_svr_non_linear = get_regressor(fixed_trial, regressor_name, random_state)

    n_candidates = X.shape[1]
    n_features = fixed_trial.params["rfe_n_features"]

    validation_rows: list[dict[str, float]] = []
    test_rows: list[dict[str, float]] = []
    pred_rows: list[dict[str, Any]] = []
    rankings_list: list[pd.Series] = []

    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        X_train_raw = X.iloc[train_idx]
        y_train = as_1d(y.iloc[train_idx])

        imputer = SimpleImputer(strategy="mean")
        X_tr = imputer.fit_transform(X_train_raw)

        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)

        rfe_estimator, _ = get_rfe_estimator(
            fixed_trial, regressor_name, regressor, is_svr_non_linear, random_state
        )
        selector = RFE(estimator=rfe_estimator, n_features_to_select=n_features, step=1)
        selector.fit(X_tr, y_train)

        rankings_list.append(pd.Series(selector.ranking_, index=X.columns))
        selected = X.columns[selector.support_].tolist()

        model = clone(make_final_pipeline(regressor))

        for set_name, indices, rows in [
            ("TRAIN", None, None),
            ("VALIDATION", val_idx, validation_rows),
            ("TEST", test_idx, test_rows),
        ]:
            if set_name == "TRAIN":
                X_cur = imputer.transform(X_train_raw)
                if use_scaler:
                    X_cur = scaler.transform(X_cur)
                X_cur = X_cur[:, selector.support_]
                model.fit(X_cur, y_train)
            else:
                X_raw = X.iloc[indices]
                X_cur = imputer.transform(X_raw)
                if use_scaler:
                    X_cur = scaler.transform(X_cur)
                X_cur = X_cur[:, selector.support_]
                y_true = as_1d(y.iloc[indices])
                y_pred = as_1d(model.predict(X_cur))
                rows.append(calculate_metrics(y_true, y_pred))
                for subject, yt, yp in zip(X.index[indices], y_true, y_pred):
                    pred_rows.append({
                        "split": split_id, "subject": subject, "set": set_name,
                        "y_true": float(yt), "y_pred": float(yp),
                    })

    rankings_df = pd.DataFrame(rankings_list)
    feature_stability = pd.DataFrame({
        "feature": X.columns,
        "selected_frequency": (rankings_df == 1).mean(),
        "mean_ranking": rankings_df.mean(),
    }).sort_values("selected_frequency", ascending=False).reset_index(drop=True)

    return {
        "validation_iterations": pd.DataFrame(validation_rows),
        "test_iterations": pd.DataFrame(test_rows),
        "validation_summary": summarize_metric_rows(validation_rows, "val"),
        "test_summary": summarize_metric_rows(test_rows, "test"),
        "predictions": pd.DataFrame(pred_rows),
        "selected_features": list(X.columns),
        "rfe_ranking": feature_stability,
        "feature_stability": feature_stability,
        "metadata": {
            "regressor": regressor_name,
            "use_scaler": use_scaler,
            "rfe_mode": "split-wise",
            "rfe_n_features": n_features,
        },
    }


def split_assignments(
    X: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        for label, indices in [("TRAIN", train_idx), ("VALIDATION", val_idx), ("TEST", test_idx)]:
            for idx in indices:
                rows.append({"split": split_id, "subject": X.index[idx], "set": label})
    return pd.DataFrame(rows)


def encode_covariates(df: pd.DataFrame, covar_cols: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if not covar_cols:
        return out
    for col in covar_cols:
        if col not in out.columns:
            raise KeyError(f"Covariate not found in metadata: {col}")
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col])[0].astype(float)
    return out


def generate_mmcv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 400,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    """Generate stratified MMCV splits (70/20/10) with full coverage tracking.

    Returns (splits, splits_df) where splits is a list of (train, val, test)
    index tuples and splits_df is a DataFrame with columns:
    COMBINATION, INDEX, COD, SET.
    """
    n_samples = len(X)
    y_qcut = pd.qcut(y, q=5, labels=False, duplicates="drop")

    splits = []
    results = []
    seen_test: set[int] = set()
    seen_val: set[int] = set()
    split_no = 0

    while split_no < n_splits or len(seen_test) < n_samples or len(seen_val) < n_samples:
        rng_seed = np.random.randint(0, 10**6)

        sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng_seed)
        train_val_idx, test_idx = next(sss_outer.split(X, y_qcut))

        train_val_y = y_qcut.iloc[train_val_idx]
        train_val_idx_arr = np.array(train_val_idx)

        sss_inner = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size / (train_size + val_size), random_state=rng_seed
        )
        train_idx, val_idx = next(sss_inner.split(train_val_idx_arr, train_val_y))

        train_idx = train_val_idx_arr[train_idx]
        val_idx = train_val_idx_arr[val_idx]

        seen_test.update(test_idx)
        seen_val.update(val_idx)

        splits.append((train_idx, val_idx, test_idx))

        cods = X.index.to_numpy()
        for idx in train_idx:
            results.append([split_no, idx, cods[idx], "TRAIN"])
        for idx in val_idx:
            results.append([split_no, idx, cods[idx], "VALIDATION"])
        for idx in test_idx:
            results.append([split_no, idx, cods[idx], "TEST"])

        split_no += 1

    splits_df = pd.DataFrame(results, columns=["COMBINATION", "INDEX", "COD", "SET"])
    return splits, splits_df


def load_per_window_matrix(
    task: int,
    window: str,
    experiment: str,
    target: str,
    metrics_dir: str | Path,
    metadata_path: str | Path,
    covar_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the per-window X/y matrix using experiment_config."""
    full_window = f"T{task}W{window}"
    X, y = expcfg.load_experiment_matrix(
        experiment=experiment,
        task=task,
        window=int(window),
        target=target,
        metrics_dir=metrics_dir,
        metadata_path=metadata_path,
        covar_cols=covar_cols,
    )
    X = X.rename(columns={col: f"{col}_{full_window}" for col in X.columns})
    return X, y


def normalize_experiment(value: str) -> str:
    valid = {"raw", "zscores", "rawzscore"}
    if value.lower().strip() in valid:
        return value.lower().strip()
    raise ValueError(f"--experiment must be one of {valid}")


def choose_targets(value: str) -> list[str]:
    value = value.strip()
    if value == "all":
        return ALL_TARGETS
    if value == "old":
        return ["MOT", "COG"]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_regressors(value: str) -> list[str]:
    if value == "default":
        regressors = DEFAULT_REGRESSORS[:]
    else:
        regressors = [item.strip() for item in value.split(",") if item.strip()]
        unknown = sorted(set(regressors) - set(ALL_REGRESSORS))
        if unknown:
            raise ValueError(f"Unknown regressors: {unknown}")

    if XGBRegressor is None:
        regressors = [name for name in regressors if name != "XGBRegressor"]

    if not regressors:
        raise ValueError("No regressors available")
    return regressors


def save_excel_report(
    path: Path,
    trials_df: pd.DataFrame,
    final_report: dict[str, Any],
    splits_df: pd.DataFrame,
) -> None:
    selected_features_df = pd.DataFrame({"feature": final_report["selected_features"]})
    final_summary = pd.DataFrame([{**final_report["validation_summary"], **final_report["test_summary"]}])
    best_metadata = pd.DataFrame(
        [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
          for k, v in final_report["metadata"].items()}]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        trials_df.to_excel(writer, sheet_name="optuna_trials", index=False)
        best_metadata.to_excel(writer, sheet_name="best_config", index=False)
        final_summary.to_excel(writer, sheet_name="final_summary", index=False)
        final_report["validation_iterations"].to_excel(writer, sheet_name="validation_iterations", index=False)
        final_report["test_iterations"].to_excel(writer, sheet_name="test_iterations", index=False)
        selected_features_df.to_excel(writer, sheet_name="selected_features", index=False)
        final_report["rfe_ranking"].to_excel(writer, sheet_name="rfe_ranking", index=False)
        if "feature_stability" in final_report:
            final_report["feature_stability"].to_excel(writer, sheet_name="feature_stability", index=False)
        final_report["predictions"].to_excel(writer, sheet_name="predictions", index=False)
        splits_df.to_excel(writer, sheet_name="mmcv_splits", index=False)


def save_study_outputs(
    study: optuna.Study,
    output_dir: Path,
    experiment_name: str,
    final_report: dict[str, Any],
    splits_df: pd.DataFrame,
) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f"optuna_trials_{experiment_name}.csv", index=False)
    save_excel_report(
        output_dir / f"optuna_trials_{experiment_name}.xlsx",
        trials_df=trials_df,
        final_report=final_report,
        splits_df=splits_df,
    )

    final_report["validation_iterations"].to_csv(output_dir / "final_validation_iterations.csv", index=False)
    final_report["test_iterations"].to_csv(output_dir / "final_test_iterations.csv", index=False)
    final_report["predictions"].to_csv(output_dir / "final_predictions.csv", index=False)
    final_report["rfe_ranking"].to_csv(output_dir / "rfe_ranking.csv", index=False)
    if "feature_stability" in final_report:
        final_report["feature_stability"].to_csv(output_dir / "feature_stability.csv", index=False)
    pd.DataFrame({"feature": final_report["selected_features"]}).to_csv(
        output_dir / "selected_features.csv", index=False
    )
    splits_df.to_csv(output_dir / "mmcv_splits.csv", index=False)

    json_payload = {
        "best_trial_number": study.best_trial.number,
        "best_value_internal_minimized": study.best_value,
        "best_params": study.best_trial.params,
        "best_metadata": final_report["metadata"],
        "validation_summary": final_report["validation_summary"],
        "test_summary": final_report["test_summary"],
        "selected_features": final_report["selected_features"],
    }
    (output_dir / "best_trial_final_report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def run_one_target(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    experiment_name: str,
    optimize_metric: str,
    regressors: list[str],
    n_trials: int,
    n_iter: int,
    seed: int,
    timeout_sec: int | None,
    max_rfe_features: int | None,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    rfe_mode: str = "global",
    optimize_splits: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating %d MMCV splits…", n_iter)
    splits, splits_df = generate_mmcv_splits(X, y, n_splits=n_iter)
    logger.info("MMCV splits generated (%d train/val/test triples)", len(splits))

    train_idx = splits[0][0] if rfe_mode == "fixed" else None

    # Determinar splits para optimización
    if optimize_splits is None:
        optimize_splits = n_iter
    n_opt = max(min(optimize_splits, n_iter), 20)

    splits_opt = splits[:n_opt]
    if rfe_mode == "fixed":
        splits_opt = splits_opt[1:]  # quitar split 0 (reservado para RFE)
        n_opt = len(splits_opt)
    logger.info(
        "Using %d splits for optimization (RFE=%s)",
        n_opt, rfe_mode,
    )

    db_path = output_dir / f"optuna_trials_{experiment_name}.db"
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
        ),
        study_name=f"optimizacion_{experiment_name}",
        storage=f"sqlite:///{db_path}",
        direction=objective_direction(optimize_metric),
        load_if_exists=True,
    )

    n_splits_opt = len(splits_opt)
    logger.info(
        "Starting optimization: %d trials, %d splits per trial, RFE=%s",
        n_trials, n_splits_opt, rfe_mode,
    )

    if rfe_mode == "split-wise":
        obj_func = partial(
            objective_regression_split_rfe,
            X=X,
            y=y,
            splits=splits_opt,
            optimize_metric=optimize_metric,
            regressors=regressors,
            timeout_sec=timeout_sec,
            random_state=seed,
            max_rfe_features=max_rfe_features,
        )
    else:
        obj_func = partial(
            objective_regression,
            X=X,
            y=y,
            splits=splits_opt,
            optimize_metric=optimize_metric,
            regressors=regressors,
            timeout_sec=timeout_sec,
            random_state=seed,
            max_rfe_features=max_rfe_features,
            train_idx=train_idx,
        )

    study.optimize(
        obj_func,
        n_trials=n_trials,
        callbacks=[TrialProgressCallback(n_trials=n_trials, report_every=50), EarlyStoppingCallback(patience=75, min_improvement=1e-4)],
        catch=(ValueError, FloatingPointError, np.linalg.LinAlgError, TrialTimeout),
        show_progress_bar=True,
    )

    if study.best_trial is None:
        raise RuntimeError("No completed trial is available")

    logger.info("Best trial found (trial %d). Re-evaluating on all splits…", study.best_trial.number)

    if rfe_mode == "split-wise":
        final_report = evaluate_fixed_trial_split_rfe(
            params=study.best_trial.params,
            X=X,
            y=y,
            splits=splits,
            regressors=regressors,
            random_state=seed,
            max_rfe_features=max_rfe_features,
        )
    else:
        final_report = evaluate_fixed_trial(
            params=study.best_trial.params,
            X=X,
            y=y,
            splits=splits,
            regressors=regressors,
            random_state=seed,
            max_rfe_features=max_rfe_features,
            train_idx=train_idx,
        )
    save_study_outputs(study, output_dir, experiment_name, final_report, splits_df)

    val = final_report["validation_summary"]
    test = final_report["test_summary"]
    logger.info("Best validation trial: %d", study.best_trial.number)
    logger.info("RFE mode: %s | Selected features: %d", rfe_mode, len(final_report["selected_features"]))
    logger.info(
        "VALIDATION %s: %.6f | TEST %s: %.6f",
        optimize_metric,
        val[f"{optimize_metric}_mean_val"],
        optimize_metric,
        test[f"{optimize_metric}_mean_test"],
    )
    logger.info("Saved outputs in: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="py -m src.analysis.regression_optuna",
        description="Optuna multi-regressor MMCV with global RFE as a required hyperparameter.",
    )
    parser.add_argument("--task", type=int, required=True, choices=[2, 6, 7])
    parser.add_argument("--window", required=True, help="Window number, e.g. 10 for W10")
    parser.add_argument("--experiment", default="raw", choices=["raw", "zscores", "rawzscore"],
                        help="Experiment: raw, zscores, or rawzscore (default: raw)")
    parser.add_argument("--rfe", default="global", choices=["global", "fixed", "split-wise"],
                        help="RFE mode: global (all data, leaky), fixed (train of split 0, no leakage), or split-wise (RFE per split, reports feature stability) (default: global)")
    parser.add_argument("--targets", default="all", help="MOT, COG, MOT_V4, COG_V1, old, or all")
    parser.add_argument("--covar", default=None, help="Comma-separated metadata covariates")
    parser.add_argument("--optimize", default="mae", choices=sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))
    parser.add_argument("--regressors", default="default", help="default or comma-separated regressor names")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--max-rfe-features", type=int, default=100)
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido: reduce splits en optimización")
    parser.add_argument("--optimize-splits", type=int, default=200,
                        help="Splits usados durante optimización (default: 200; con --fast: n_iter//4, min 50)")
    parser.add_argument("--pruner-startup-trials", type=int, default=100)
    parser.add_argument("--pruner-warmup-steps", type=int, default=15)
    parser.add_argument("--metrics-dir", default="data/processed/metrics")
    parser.add_argument("--metadata", default="data/raw/metadata.xlsx")
    parser.add_argument("--output", default="outputs/regression_optuna")
    parser.add_argument("--run-name", default=None, help="Optional experiment name suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = normalize_experiment(args.experiment)
    targets = choose_targets(args.targets)
    regressors = parse_regressors(args.regressors)
    covar_cols = [item.strip() for item in args.covar.split(",") if item.strip()] if args.covar else None

    if args.fast:
        args.optimize_splits = max(50, args.n_iter // 4)

    output_root = Path(args.output)
    setup_logging(log_dir=output_root / "logs")

    logger.info("=" * 60)
    logger.info("Regression Optuma experiment with RFE")
    logger.info("=" * 60)
    logger.info(
        "Task=%s | Window=W%s | Experiment=%s | RFE=%s | Targets=%s",
        args.task, args.window, experiment, args.rfe, targets,
    )
    logger.info(
        "Optimization metric=%s | Trials=%d | MMCV splits=%d (opt=%s)",
        args.optimize, args.n_trials, args.n_iter,
        args.optimize_splits or args.n_iter,
    )
    logger.info("Regressors=%s", regressors)
    if args.fast:
        logger.info("Fast mode: ON (excluye Bagging, Stacking, GPR)")

    t_total_start = time.time()

    for target in targets:
        t_target_start = time.time()
        logger.info("-" * 60)
        logger.info("Loading data for target=%s", target)
        X, y = load_per_window_matrix(
            task=args.task,
            window=args.window,
            experiment=experiment,
            target=target,
            metrics_dir=args.metrics_dir,
            metadata_path=args.metadata,
            covar_cols=covar_cols,
        )
        logger.info(
            "Target=%s | Subjects=%d | Candidate predictors=%d",
            target, len(y), X.shape[1],
        )

        suffix = f"_{args.run_name}" if args.run_name else ""
        experiment_name = f"task{args.task}_W{args.window}_{experiment}_{target}_rfe{args.rfe}_{args.optimize}{suffix}"
        output_dir = Path(args.output) / f"task{args.task}" / f"W{args.window}_{experiment}_{args.rfe}" / target

        run_one_target(
            X=X,
            y=y,
            output_dir=output_dir,
            experiment_name=experiment_name,
            optimize_metric=args.optimize,
            regressors=regressors,
            n_trials=args.n_trials,
            n_iter=args.n_iter,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
            max_rfe_features=args.max_rfe_features,
            pruner_startup_trials=args.pruner_startup_trials,
            pruner_warmup_steps=args.pruner_warmup_steps,
            rfe_mode=args.rfe,
            optimize_splits=args.optimize_splits,
        )

        elapsed_target = time.time() - t_target_start
        logger.info(
            "Target=%s completed in %.1f s (%.1f min)",
            target, elapsed_target, elapsed_target / 60,
        )

    elapsed_total = time.time() - t_total_start
    logger.info("=" * 60)
    logger.info("ALL TARGETS COMPLETE | Total: %.1f s (%.1f min)", elapsed_total, elapsed_total / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
