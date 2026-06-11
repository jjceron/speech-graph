from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "regression_optuna"
ALL_TARGETS = ["MOT", "COG", "MOT_V4", "COG_V1"]
EXPERIMENTS = ["raw", "zscores", "rawzscore"]
WINDOWS = ["10", "20", "30", "40"]


def get_task() -> int:
    return st.session_state.get("task", 2)


def set_task(task: int) -> None:
    st.session_state.task = task


def list_tasks() -> list[int]:
    tasks = []
    if BASE_DIR.exists():
        for d in BASE_DIR.iterdir():
            if d.is_dir() and d.name.startswith("task"):
                try:
                    tasks.append(int(d.name.replace("task", "")))
                except ValueError:
                    pass
    return sorted(tasks) if tasks else [2]


def _exp_dir(window: str, experiment: str) -> Path:
    return BASE_DIR / f"task{get_task()}" / f"W{window}_{experiment}_fixed"


def has_experiment(window: str, experiment: str) -> bool:
    return _exp_dir(window, experiment).exists()


def list_completed() -> list[tuple[str, str]]:
    completed = []
    for w in WINDOWS:
        for e in EXPERIMENTS:
            if has_experiment(w, e) and all(
                (_exp_dir(w, e) / t).exists() for t in ALL_TARGETS
            ):
                completed.append((w, e))
    return completed


def has_target(window: str, experiment: str, target: str) -> bool:
    return (_exp_dir(window, experiment) / target).exists()


@st.cache_data
def load_best_report(window: str, experiment: str, target: str) -> dict[str, Any] | None:
    path = _exp_dir(window, experiment) / target / "best_trial_final_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_test_iterations(window: str, experiment: str, target: str) -> pd.DataFrame | None:
    path = _exp_dir(window, experiment) / target / "final_test_iterations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_val_iterations(window: str, experiment: str, target: str) -> pd.DataFrame | None:
    path = _exp_dir(window, experiment) / target / "final_validation_iterations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_predictions(window: str, experiment: str, target: str) -> pd.DataFrame | None:
    path = _exp_dir(window, experiment) / target / "final_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_rfe_ranking(window: str, experiment: str, target: str) -> pd.DataFrame | None:
    path = _exp_dir(window, experiment) / target / "rfe_ranking.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_optuna_trials(window: str, experiment: str, target: str) -> pd.DataFrame | None:
    exp_dir = _exp_dir(window, experiment) / target
    for f in exp_dir.glob(f"optuna_trials_task{get_task()}_*.csv"):
        return pd.read_csv(f)
    return None


@st.cache_data
def load_selected_features(window: str, experiment: str, target: str) -> list[str] | None:
    path = _exp_dir(window, experiment) / target / "selected_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.iloc[:, 0].tolist()


def load_all_reports() -> dict[tuple[str, str, str], dict]:
    reports = {}
    for w, e in list_completed():
        for t in ALL_TARGETS:
            r = load_best_report(w, e, t)
            if r:
                reports[(w, e, t)] = r
    return reports
