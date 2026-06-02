from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_RUN_ORDER = [
    "00_smoke_test",
    "01_windows_no_random",
    "02_windows_by_activity",
    "03_windows_random100",
    "04_windows_random1000",
]

RANDOM_RUN_ORDER = ["03_windows_random100", "04_windows_random1000"]

# Metrics that are methodologically useful for the NLP/speech-graph analysis.
# We intentionally exclude mechanical size variables such as token_count,
# window_count, global_edges, mean_edges, and LCC when it is equivalent to nodes.
CORE_METRICS = [
    "mean_z_lcc",
    "mean_z_lsc",
    "mean_z_edges",
    "mean_lsc_ratio",
    "mean_lsc",
    "mean_nodes",
    "mean_repeated_edges_ratio",
    "mean_density",
    "mean_asp",
    "mean_diameter",
    "mean_clustering",
    "mean_l2",
    "mean_l3",
]

RANDOM_CORE_METRICS = ["mean_z_lcc", "mean_z_lsc", "mean_z_edges"]

GRAPH_INTERPRETATION_METRICS = [
    "mean_lsc_ratio",
    "mean_lsc",
    "mean_nodes",
    "mean_repeated_edges_ratio",
    "mean_density",
    "mean_asp",
    "mean_diameter",
    "mean_clustering",
    "mean_l2",
    "mean_l3",
]

BARRATT_TARGETS = [
    "TOTAL",
    "Barratt (pre)",
    "TOTAL_zscore",
    "NPLAN",
    "MOT",
    "COG",
    "COG_zscore",
    "MOT_zscore",
]

COGNITIVE_TARGETS = [
    "Naming task",
    "COHERENCIA NARRATIVA",
    "Conceptualization task",
    "Reading comprehension task",
    "Verbal fluency tasks",
    "FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS",
    "FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES",
    "FLUIDEZ VERBAL - PRUEBA FONETICA",
]

DEMOGRAPHIC_TARGETS = ["Age", "School year"]

BARRATT_ITEM_TARGETS = [f"{i}." for i in range(1, 27)]

TARGET_SETS = {
    "barratt": BARRATT_TARGETS,
    "cognitive": COGNITIVE_TARGETS,
    "demographic": DEMOGRAPHIC_TARGETS,
    "barratt_items": BARRATT_ITEM_TARGETS,
}

DEFAULT_GROUP_COLS = ["Tipo", "Educational level", "School", "Gender", "level", "activity"]

# Tags that reflect annotation/clinical-discourse events. We exclude parser artifacts.
PREFERRED_LABEL_RATIOS = [
    "label_ratio_SIN_RESPUESTA",
    "label_ratio_SIN_PREGUNTA",
    "label_ratio_PAUSA",
    "label_ratio_EE",
    "label_ratio_IF",
    "label_ratio_PS",
    "label_ratio_DI",
    "label_ratio_DP",
    "label_ratio_PNC",
    "label_ratio_IM",
    "label_ratio_ES",
]

BAD_LABEL_PATTERNS = ("STARTTIME", "ENDTIME")

MECHANICAL_COLUMNS = {
    "token_count",
    "global_token_count",
    "global_edges",
    "global_lcc",
    "global_lcc_ratio",
    "window_count",
    "mean_token_count",
    "mean_edges",
    "mean_lcc",
    "mean_lcc_ratio",
    "std_token_count",
    "std_edges",
    "std_lcc",
    "std_lcc_ratio",
}

ID_COLUMNS = {
    "code",
    "Cod",
    "file",
    "level",
    "activity",
    "activity_index",
    "start_time",
    "end_time",
    "_merge",
}


def safe_name(value: object, max_len: int = 120) -> str:
    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len] or "value")


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def discover_run_dirs(outputs_root: Path, runs: Sequence[str] | None = None, random_only: bool = False) -> list[Path]:
    preferred = RANDOM_RUN_ORDER if random_only else DEFAULT_RUN_ORDER
    if runs:
        return [outputs_root / run for run in runs if (outputs_root / run).exists()]

    ordered = [outputs_root / run for run in preferred if (outputs_root / run).exists()]
    seen = {p.resolve() for p in ordered}
    if not outputs_root.exists():
        return ordered
    extras = [p for p in sorted(outputs_root.iterdir()) if p.is_dir() and p.resolve() not in seen]
    extras = [p for p in extras if any(p.glob("graph_metrics*.csv")) or (p / "analysis").exists()]
    return ordered + ([] if random_only else extras)


def select_data_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "graph_metrics_all_windows_with_meta.csv",
        run_dir / "graph_metrics_all_windows.csv",
        run_dir / "graph_metrics_with_meta.csv",
        run_dir / "graph_metrics.csv",
    ]
    candidates.extend(sorted(run_dir.glob("graph_metrics_*_with_meta.csv")))
    candidates.extend(sorted(run_dir.glob("graph_metrics_w*_s*.csv")))
    candidates.extend(sorted(run_dir.glob("graph_metrics*.csv")))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def read_data_csv(run_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    path = select_data_csv(run_dir)
    if not path:
        return None, None
    try:
        return pd.read_csv(path), path
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None, path


def filter_analysis_level(df: pd.DataFrame, level: str = "file") -> pd.DataFrame:
    if level and "level" in df.columns and level != "all":
        sub = df[df["level"].astype(str).str.lower() == level.lower()].copy()
        if not sub.empty:
            return sub
    return df.copy()


def pick_window(df: pd.DataFrame, window_size: int | str = 30) -> pd.DataFrame:
    if "window_size" not in df.columns or str(window_size).lower() == "all":
        return df.copy()
    ws = pd.to_numeric(df["window_size"], errors="coerce")
    sub = df[ws == int(window_size)].copy()
    return sub if not sub.empty else df.copy()


def available_core_metrics(df: pd.DataFrame, include_nonrandom: bool = True) -> list[str]:
    metrics: list[str] = []
    for metric in CORE_METRICS:
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().sum() >= 3:
            metrics.append(metric)
    if not include_nonrandom:
        metrics = [m for m in metrics if m in RANDOM_CORE_METRICS]
    return metrics


def available_targets(df: pd.DataFrame, target_set: str | Iterable[str] = "all") -> list[str]:
    if isinstance(target_set, str):
        if target_set == "all":
            requested = DEMOGRAPHIC_TARGETS + BARRATT_TARGETS + COGNITIVE_TARGETS
        else:
            requested = TARGET_SETS.get(target_set, parse_csv_list(target_set))
    else:
        requested = list(target_set)
    targets: list[str] = []
    for col in requested:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 3:
            targets.append(col)
    return targets


def available_label_ratios(df: pd.DataFrame, min_nonzero: int = 8) -> list[str]:
    cols: list[str] = []
    preferred = [c for c in PREFERRED_LABEL_RATIOS if c in df.columns]
    extras = [c for c in df.columns if c.startswith("label_ratio_") and c not in preferred]
    for col in preferred + extras:
        upper = col.upper()
        if any(pattern in upper for pattern in BAD_LABEL_PATTERNS):
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if (values > 0).sum() >= min_nonzero:
            cols.append(col)
    return cols


def ensure_figures_dir(run_dir: Path, subdir: str = "nlp_profile") -> Path:
    path = run_dir / "figures" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_manifest(rows: list[dict], run_dir: Path, subdir: str = "nlp_profile") -> None:
    if not rows:
        return
    manifest_path = run_dir / "figures" / subdir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        out = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            subset=[c for c in ["figure", "table", "kind"] if c in pd.concat([old, new_df], ignore_index=True).columns],
            keep="last",
        )
    else:
        out = new_df
    out.to_csv(manifest_path, index=False)


def write_text_report(run_dir: Path, lines: list[str], filename: str = "README_figures.txt", subdir: str = "nlp_profile") -> Path:
    out_dir = ensure_figures_dir(run_dir, subdir)
    path = out_dir / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def is_meaningful_metric(col: str) -> bool:
    if col in MECHANICAL_COLUMNS:
        return False
    if col.startswith("global_"):
        # global metrics are usually dominated by transcript length; keep only ratios if needed.
        return col.endswith("_ratio") and col not in {"global_lcc_ratio"}
    if col.startswith("std_"):
        return False
    if col.startswith("mean_") or col.startswith("label_ratio_") or col.startswith("emotion_"):
        return True
    return col.startswith("z_")


def clean_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return out

# ---------------------------------------------------------------------------
# Backward-compatible helpers for legacy visualization subcommands.
# The default `py -m src.visualization` no longer uses these because they can
# create exploratory/noisy plots, but direct legacy modules still work.
# ---------------------------------------------------------------------------
DEFAULT_METRICS = CORE_METRICS


def available_metric_cols(df: pd.DataFrame, requested: Iterable[str] | None = None, max_metrics: int = 20) -> list[str]:
    if requested:
        metrics = [m for m in requested if m in df.columns]
        if metrics:
            return metrics[:max_metrics]
    metrics = [m for m in CORE_METRICS if m in df.columns]
    if len(metrics) >= max_metrics:
        return metrics[:max_metrics]
    for col in df.columns:
        if col in metrics or not is_meaningful_metric(col):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            metrics.append(col)
        if len(metrics) >= max_metrics:
            break
    return metrics


def correlation_csv(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "analysis" / "correlations_by_window.csv",
        run_dir / "analysis" / "correlations.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((run_dir / "analysis").glob("*correlation*.csv")) if (run_dir / "analysis").exists() else []
    return matches[0] if matches else None


def stability_csv(run_dir: Path) -> Path | None:
    candidates = [run_dir / "analysis" / "window_metric_stability.csv"]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((run_dir / "analysis").glob("*stability*.csv")) if (run_dir / "analysis").exists() else []
    return matches[0] if matches else None
