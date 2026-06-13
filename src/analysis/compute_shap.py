"""Compute SHAP values for all completed regression experiments.

Usage:
    py -m src.analysis.compute_shap --all --task 2
    py -m src.analysis.compute_shap --all --task 2 --task 6
    py -m src.analysis.compute_shap --task 2 --window 30 --experiment rawzscore --target COG
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent
if str(SRC.parent) not in sys.path:
    sys.path.insert(0, str(SRC.parent))

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    BaggingRegressor,
    StackingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, QuantileRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

import shap

from src.analysis import experiment_config as expcfg
from src.analysis.linear_regression_rcv import ALL_TARGETS

RNG = np.random.RandomState(42)

TRIAL_TO_SKL_PARAM: dict[str, dict[str, str]] = {
    "Ridge": {"ridge_alpha": "alpha", "ridge_solver": "solver"},
    "ElasticNet": {"elastic_alpha": "alpha", "elastic_l1_ratio": "l1_ratio"},
    "QuantileRegressor": {"quantile_alpha": "alpha"},
    "SVR": {"svr_C": "C", "svr_epsilon": "epsilon", "svr_kernel": "kernel", "svr_degree": "degree", "svr_gamma": "gamma"},
    "RandomForestRegressor": {"rf_n_estimators": "n_estimators", "rf_max_depth": "max_depth", "rf_min_samples_split": "min_samples_split", "rf_min_samples_leaf": "min_samples_leaf", "rf_max_features": "max_features"},
    "ExtraTreesRegressor": {"et_n_estimators": "n_estimators", "et_max_depth": "max_depth", "et_min_samples_split": "min_samples_split", "et_min_samples_leaf": "min_samples_leaf", "et_max_features": "max_features"},
    "BaggingRegressor": {"bag_n_estimators": "n_estimators", "bag_max_samples": "max_samples", "bag_max_features": "max_features"},
    "GaussianProcessRegressor": {"gpr_alpha": "alpha"},
    "KNeighborsRegressor": {"knn_n_neighbors": "n_neighbors", "knn_weights": "weights", "knn_metric": "metric"},
    "DecisionTreeRegressor": {"dt_max_depth": "max_depth", "dt_min_samples_split": "min_samples_split", "dt_min_samples_leaf": "min_samples_leaf"},
    "XGBRegressor": {"xgb_n_estimators": "n_estimators", "xgb_max_depth": "max_depth", "xgb_learning_rate": "learning_rate", "xgb_subsample": "subsample", "xgb_colsample_bytree": "colsample_bytree", "xgb_reg_alpha": "reg_alpha", "xgb_reg_lambda": "reg_lambda", "xgb_booster": "booster"},
}

REGRESSOR_CLS: dict[str, Any] = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "QuantileRegressor": QuantileRegressor,
    "SVR": SVR,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "BaggingRegressor": BaggingRegressor,
    "StackingRegressor": StackingRegressor,
    "GaussianProcessRegressor": GaussianProcessRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "XGBRegressor": XGBRegressor,
}

REGRESSOR_FIXED: dict[str, dict[str, Any]] = {
    "QuantileRegressor": {"quantile": 0.5, "solver": "highs"},
    "LinearRegression": {},
}


def build_regressor(params: dict[str, Any], random_state: int = 42) -> Any:
    name = params["regressor"]
    cls = REGRESSOR_CLS.get(name)
    if cls is None:
        raise ValueError(f"Unknown regressor: {name}")
    kwargs = dict(REGRESSOR_FIXED.get(name, {}))
    mapping = TRIAL_TO_SKL_PARAM.get(name, {})
    for trial_key, skl_key in mapping.items():
        if trial_key in params:
            kwargs[skl_key] = params[trial_key]
    try:
        kwargs["random_state"] = random_state
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("random_state")
        return cls(**kwargs)


def select_explainer(model, name: str, X_background: np.ndarray | pd.DataFrame):
    tree_families = {"RandomForestRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor", "XGBRegressor"}
    linear_families = {"LinearRegression", "Ridge", "ElasticNet"}
    if name in tree_families:
        if name == "XGBRegressor":
            bg = shap.sample(X_background, min(50, len(X_background)))
            return shap.KernelExplainer(model.predict, bg)
        return shap.TreeExplainer(model)
    elif name in linear_families:
        return shap.LinearExplainer(model, X_background)
    else:
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(model.predict, bg)


def compute_shap_for(
    task: int,
    window: int,
    experiment: str,
    target: str,
    base_dir: str | Path = "outputs/regression_optuna",
    force: bool = False,
) -> bool:
    exp_dir = Path(base_dir) / f"task{task}" / f"W{window}_{experiment}_fixed" / target
    report_path = exp_dir / "best_trial_final_report.json"
    out_csv = exp_dir / "shap_values.csv"
    out_json = exp_dir / "shap_summary.json"
    out_feat = exp_dir / "shap_feature_values.csv"

    if not report_path.exists():
        return False
    if not force and out_csv.exists() and out_json.exists() and out_feat.exists():
        return True

    with open(report_path) as f:
        report = json.load(f)

    params = report.get("best_params", {})
    if not params:
        return False

    selected_features = report.get("selected_features", [])
    if not selected_features:
        return False

    try:
        X_full, y = expcfg.load_experiment_matrix(
            experiment=experiment,
            task=task,
            window=window,
            target=target,
        )
    except (FileNotFoundError, ValueError) as e:
        return False

    full_window = f"T{task}W{window}"
    X_full = X_full.rename(columns={col: f"{col}_{full_window}" for col in X_full.columns})

    available = [f for f in selected_features if f in X_full.columns]
    if not available:
        return False
    X = X_full[available]

    regressor = build_regressor(params)
    use_scaler = params.get("use_scaler", False)
    steps = [("imputer", SimpleImputer(strategy="mean"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", regressor))
    pipeline = Pipeline(steps)
    pipeline.fit(X, y)

    preprocessor = Pipeline(steps[:-1])
    X_transformed = preprocessor.fit_transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=available, index=X.index)

    with redirect_stderr(StringIO()):
        explainer = select_explainer(regressor, params["regressor"], X_transformed_df)
        shap_values = explainer.shap_values(X_transformed_df)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    shap_df = pd.DataFrame(shap_values, columns=available, index=X.index)
    shap_df["y_true"] = y.values
    shap_df["y_pred"] = pipeline.predict(X)
    shap_df.index.name = "subject"
    shap_df.to_csv(out_csv)

    feat_df = X_transformed_df.copy()
    feat_df.index.name = "subject"
    feat_df.to_csv(out_feat)

    summary = {
        "mean_abs_shap": {f: float(np.abs(shap_values[:, i]).mean()) for i, f in enumerate(available)},
        "mean_shap": {f: float(shap_values[:, i].mean()) for i, f in enumerate(available)},
        "regressor": params["regressor"],
        "n_subjects": len(X),
        "n_features": len(available),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    return True


def get_completed_experiments(task: int) -> list[tuple[int, str]]:
    base_dir = Path("outputs/regression_optuna")
    task_dir = base_dir / f"task{task}"
    if not task_dir.exists():
        return []
    completed = []
    for d in sorted(task_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("W"):
            continue
        parts = name.split("_")
        if len(parts) >= 2:
            window = int(parts[0].replace("W", ""))
            exp = "_".join(parts[1:-1])
            completed.append((window, exp))
    return completed


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values for regression experiments")
    parser.add_argument("--all", action="store_true", help="Process all completed experiments")
    parser.add_argument("--task", type=int, nargs="+", default=[2], help="Task number(s)")
    parser.add_argument("--window", type=int, help="Window size")
    parser.add_argument("--experiment", type=str, help="Experiment type (raw, rawzscore, zscores)")
    parser.add_argument("--target", type=str, help="Target variable (or 'all' for all targets)")
    parser.add_argument("--force", action="store_true", help="Recompute even if files exist")
    args = parser.parse_args()

    if args.all:
        tasks = args.task
        total = 0
        for task in tasks:
            experiments = get_completed_experiments(task)
            if not experiments:
                continue
            for window, experiment in experiments:
                for target in ALL_TASKS.get(task, ALL_TASKS[2]):
                    ok = compute_shap_for(task, window, experiment, target, force=args.force)
                    if ok:
                        total += 1
        print(f"Done. SHAP computed for {total} combinations.")
    else:
        if not args.window or not args.experiment or not args.target:
            parser.error("--window, --experiment, --target required when not using --all")
        task = (args.task or [2])[0]
        targets = ALL_TASKS.get(task, ALL_TASKS[2]) if args.target == "all" else [args.target]
        total = 0
        for target in targets:
            ok = compute_shap_for(task, args.window, args.experiment, target, force=args.force)
            if ok:
                total += 1
        print(f"Done. SHAP computed for {total}/{len(targets)} targets (task{task} W{args.window} {args.experiment}).")


ALL_TASKS = {
    2: ["COG", "COG_V1", "MOT", "MOT_V4"],
    6: ["COG", "COG_V1", "MOT", "MOT_V4"],
}

if __name__ == "__main__":
    main()
