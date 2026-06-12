from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "regression_optuna"

EXPECTED_WINDOWS = ["10", "20", "30", "40"]
EXPECTED_EXPERIMENTS = ["raw", "zscores", "rawzscore"]


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


def _exp_dir(window: str, experiment: str, task: int | None = None) -> Path:
    if task is None:
        task = get_task()
    return BASE_DIR / f"task{task}" / f"W{window}_{experiment}_fixed"


def get_windows(task: int | None = None) -> list[str]:
    if task is None:
        task = get_task()
    task_dir = BASE_DIR / f"task{task}"
    if not task_dir.exists():
        return []
    windows = set()
    for d in task_dir.iterdir():
        if d.is_dir():
            m = re.match(r"W(\d+)_", d.name)
            if m:
                windows.add(m.group(1))
    return sorted(windows, key=int)


def get_experiments(task: int | None = None) -> list[str]:
    if task is None:
        task = get_task()
    task_dir = BASE_DIR / f"task{task}"
    if not task_dir.exists():
        return []
    experiments = set()
    for d in task_dir.iterdir():
        if d.is_dir():
            m = re.match(r"W\d+_(.+)_fixed", d.name)
            if m:
                experiments.add(m.group(1))
    return sorted(experiments)


def get_targets(task: int | None = None, window: str | None = None, experiment: str | None = None) -> list[str]:
    if task is None:
        task = get_task()
    if window is not None and experiment is not None:
        exp_dir = _exp_dir(window, experiment, task=task)
        if exp_dir.exists():
            return sorted([d.name for d in exp_dir.iterdir() if d.is_dir()])
    targets = set()
    for w in get_windows(task=task):
        for e in get_experiments(task=task):
            d = _exp_dir(w, e, task=task)
            if d.exists():
                for sub in d.iterdir():
                    if sub.is_dir():
                        targets.add(sub.name)
    return sorted(targets)


def has_experiment(window: str, experiment: str, task: int | None = None) -> bool:
    return _exp_dir(window, experiment, task=task).exists()


def list_completed() -> list[tuple[str, str]]:
    task = get_task()
    targets = get_targets(task=task)
    if not targets:
        return []
    completed = []
    for w in get_windows(task=task):
        for e in get_experiments(task=task):
            exp_dir = _exp_dir(w, e, task=task)
            if exp_dir.exists() and all((exp_dir / t).exists() for t in targets):
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
    task = get_task()
    reports = {}
    for w, e in list_completed():
        for t in get_targets(task=task, window=w, experiment=e):
            r = load_best_report(w, e, t)
            if r:
                reports[(w, e, t)] = r
    return reports
