from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.experiment_config import _compute_targets, load_feature_table
from src.pipeline.speechgraph import load_metadata

METRICS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "metrics"
METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "metadata.xlsx"

TASK_DIR_MAP = {
    "Task2": "Task2",
    "Task6-A": "Task6",
    "Task6-B": "Task6",
    "Task7": "Task7",
}

dict_task_type_red: dict[str, dict[str, list[int]]] = {
    "Task2": {"Estructural": [10, 20, 30, 40], "Semántico": [2, 3, 4]},
    "Task6-A": {"Estructural": [30, 40, 50], "Semántico": [3, 4, 5]},
    "Task6-B": {"Estructural": [150, 160, 170, 180, 190, 200], "Semántico": [11, 12, 13]},
    "Task7": {"Estructural": [20, 30, 40, 50], "Semántico": [2, 3, 4]},
}

dict_name_files: dict[str, str] = {
    "Estructural clásico": "means_params_tableT*W##",
    "Estructural limpieza agresiva": "means_params_tableT*W##_all_pos",
    "Estructural clásico random graphs": "z_means_params_table_T*W##",
    "Estructural limpieza agresiva random graphs": "z_means_params_table_T*W##_all_pos",
    "Semántico action relation": "means_params_table_T*W##_ap",
    "Semántico prediction relation": "means_params_table_T*W##_pa",
    "Semántico all relation": "means_params_table_T*W##_semantic",
    "Semántico action relation random graphs": "z_means_params_table_T*W##_ap_er",
    "Semántico prediction relation random graphs": "z_means_params_table_T*W##_pa_er",
    "Semántico all relation random graphs": "z_means_params_table_T*W##_semantic_er",
    "Semántico action relation weight": "means_params_table_T*W##_ap_weight",
    "Semántico prediction relation weight": "means_params_table_T*W##_pa_weight",
    "Semántico all relation weight": "means_params_table_T*W##_semantic_weight",
    "Semántico action relation random graphs weight": "z_means_params_table_T*W##_ap",
    "Semántico prediction relation random graphs weight": "z_means_params_table_T*W##_pa",
    "Semántico all relation random graphs weight": "z_means_params_table_T*W##_semantic",
}

NETWORK_TYPE_OPTIONS = [
    "Estructural clásico",
    "Estructural limpieza agresiva",
    "Estructural clásico random graphs",
    "Estructural limpieza agresiva random graphs",
    "Semántico action relation",
    "Semántico prediction relation",
    "Semántico all relation",
    "Semántico action relation random graphs",
    "Semántico prediction relation random graphs",
    "Semántico all relation random graphs",
    "Semántico action relation weight",
    "Semántico prediction relation weight",
    "Semántico all relation weight",
    "Semántico action relation random graphs weight",
    "Semántico prediction relation random graphs weight",
    "Semántico all relation random graphs weight",
]

METADATA_VARS = ["TOTAL", "NPLAN", "COG", "MOT", "MOT_V4", "COG_V1", "School year", "Age"]

GRAPH_FEATURES = ["nodes", "edges", "re", "pe", "l1", "l2", "l3", "lcc", "lsc", "atd", "density", "diameter", "asp", "cc"]

Z_GRAPH_FEATURES = [f"z_{f}" for f in GRAPH_FEATURES]

ALL_VARS = METADATA_VARS + GRAPH_FEATURES


def get_network_category(network_type: str) -> str:
    if network_type.startswith("Estructural"):
        return "Estructural"
    return "Semántico"


def is_random_graph(network_type: str) -> bool:
    return "random graphs" in network_type


def has_weight(network_type: str) -> bool:
    return network_type.endswith("weight")


def extract_task_number(task_label: str) -> int:
    m = re.match(r"Task(\d+)", task_label)
    return int(m.group(1)) if m else 2


def build_file_list(task_label: str, network_type: str) -> list[dict[str, Any]]:
    category = get_network_category(network_type)
    windows = dict_task_type_red.get(task_label, {}).get(category, [])
    pattern = dict_name_files.get(network_type, "")
    task_num = extract_task_number(task_label)

    task_dir = METRICS_DIR / TASK_DIR_MAP[task_label]
    files = []
    for w in windows:
        filename = pattern.replace("*", str(task_num)).replace("##", str(w)) + ".txt"
        filepath = task_dir / filename
        if filepath.exists():
            files.append({"path": filepath, "label": f"T{task_num}W{w}"})
        else:
            alt_filename = filename.replace("z_means_params_table_T", "z_means_params_tableT")
            alt_filepath = task_dir / alt_filename
            if alt_filepath.exists():
                files.append({"path": alt_filepath, "label": f"T{task_num}W{w}"})
    return files


def load_and_prepare_metadata() -> pd.DataFrame:
    meta = load_metadata(str(METADATA_PATH))
    meta = _compute_targets(meta)
    return meta


def load_metric_file(filepath: Path) -> pd.DataFrame:
    return load_feature_table(filepath)


def get_graph_feature_columns(df: pd.DataFrame, network_type: str) -> list[str]:
    if is_random_graph(network_type):
        return [c for c in Z_GRAPH_FEATURES if c in df.columns]
    return [c for c in GRAPH_FEATURES if c in df.columns]


def strip_z_prefix(col: str) -> str:
    return col[2:] if col.startswith("z_") else col
