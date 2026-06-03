from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TARGETS = ["TOTAL", "NPLAN", "MOT", "COG"]
DEFAULT_LABEL_COL = "Tipo"
DEFAULT_DEMOGRAPHIC_COLS = ["Age", "Gender", "School year", "School", "Educational level"]
DEFAULT_COVARIATE_COLS = ["Age", "Gender", "School year"]
DEFAULT_SPEECHGRAPH_METRICS = [
    "nodes", "edges", "re", "pe", "l1", "l2", "l3", "lcc", "lsc", "atd",
    "density", "diameter", "asp", "cc",
]
DEFAULT_LEAKAGE_PATTERNS = [
    r"^\s*\d+\.\s*$",          # Barratt item columns such as 1., 2., ... 26.
    r"barratt",                    # Barratt raw/pre columns
    r"zscore",                     # z-scores derived from Barratt
]
DEFAULT_LEAKAGE_EXACT = {
    "TOTAL", "NPLAN", "MOT", "COG", "Tipo", "type", "high_low", "high_imp", "low_imp",
    "Barratt (pre)", "TOTAL_zscore", "COG_zscore", "MOT_zscore",
}


def parse_csv_list(text: str | Iterable[str] | None) -> list[str]:
    if text is None:
        return []
    if isinstance(text, str):
        return [x.strip() for x in text.split(",") if x.strip()]
    return [str(x).strip() for x in text if str(x).strip()]


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def resolve_columns(df: pd.DataFrame, requested: str | Iterable[str] | None) -> list[str]:
    """Resolve requested column names case-insensitively, preserving dataframe spelling."""
    wanted = parse_csv_list(requested)
    norm_to_col = {_norm(c): c for c in df.columns}
    out: list[str] = []
    for item in wanted:
        col = norm_to_col.get(_norm(item))
        if col is not None and col not in out:
            out.append(col)
    return out


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    norm_to_col = {_norm(c): c for c in df.columns}
    for cand in candidates:
        col = norm_to_col.get(_norm(cand))
        if col is not None:
            return col
    return None


def is_leakage_column(
    col: str,
    targets: Iterable[str] = DEFAULT_TARGETS,
    label_col: str = DEFAULT_LABEL_COL,
    extra_exclude: Iterable[str] | None = None,
) -> bool:
    exact = set(DEFAULT_LEAKAGE_EXACT)
    exact.update(str(t) for t in targets)
    exact.add(str(label_col))
    if extra_exclude:
        exact.update(str(x) for x in extra_exclude)
    if col in exact or _norm(col) in {_norm(x) for x in exact}:
        return True
    return any(re.search(pattern, str(col), flags=re.IGNORECASE) for pattern in DEFAULT_LEAKAGE_PATTERNS)


def detect_speechgraph_columns(df: pd.DataFrame) -> list[str]:
    """Detect wide SpeechGraph columns such as A1_W20_density."""
    metric_pattern = "|".join(re.escape(m) for m in DEFAULT_SPEECHGRAPH_METRICS)
    pat = re.compile(rf"^A\d+_W\d+_({metric_pattern})$", flags=re.IGNORECASE)
    return [c for c in df.columns if pat.match(str(c))]


def infer_cognitive_columns(
    df: pd.DataFrame,
    code_col: str = "code",
    targets: Iterable[str] = DEFAULT_TARGETS,
    label_col: str = DEFAULT_LABEL_COL,
    demographic_cols: Iterable[str] = DEFAULT_DEMOGRAPHIC_COLS,
    speechgraph_cols: Iterable[str] | None = None,
    extra_exclude: Iterable[str] | None = None,
) -> list[str]:
    """Infer non-Barratt cognitive/language metadata columns.

    The function intentionally excludes demographics, SpeechGraph variables, targets,
    high/low labels, Barratt item columns, and Barratt-derived z-scores.
    """
    speechgraph = set(speechgraph_cols or detect_speechgraph_columns(df))
    demographics = set(resolve_columns(df, demographic_cols))
    excludes = {code_col, *speechgraph, *demographics}
    if extra_exclude:
        excludes.update(extra_exclude)
    out: list[str] = []
    for col in df.columns:
        if col in excludes:
            continue
        if is_leakage_column(col, targets=targets, label_col=label_col, extra_exclude=extra_exclude):
            continue
        # Keep numeric cognitive/language columns and categorical task descriptors if any.
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique(dropna=True) <= 20:
            out.append(col)
    return out


@dataclass(frozen=True)
class FeatureBlocks:
    code_col: str
    targets: list[str]
    label_col: str | None
    demographics: list[str]
    cognitive: list[str]
    speechgraph: list[str]
    covariates: list[str]

    @property
    def metadata_features(self) -> list[str]:
        return list(dict.fromkeys([*self.demographics, *self.cognitive]))

    @property
    def multimodal_features(self) -> list[str]:
        return list(dict.fromkeys([*self.demographics, *self.cognitive, *self.speechgraph]))


def make_feature_blocks(
    df: pd.DataFrame,
    code_col: str = "code",
    targets_text: str = "TOTAL,NPLAN,MOT,COG",
    label_col: str = "Tipo",
    demographic_cols_text: str | None = None,
    cognitive_cols_text: str | None = None,
    covariate_cols_text: str | None = None,
) -> FeatureBlocks:
    targets = resolve_columns(df, targets_text)
    if not targets:
        targets = parse_csv_list(targets_text)
    label = first_existing_column(df, [label_col])
    demographics = resolve_columns(df, demographic_cols_text or DEFAULT_DEMOGRAPHIC_COLS)
    speechgraph = detect_speechgraph_columns(df)
    if cognitive_cols_text:
        cognitive = resolve_columns(df, cognitive_cols_text)
    else:
        cognitive = infer_cognitive_columns(
            df,
            code_col=code_col,
            targets=targets,
            label_col=label_col,
            demographic_cols=demographics,
            speechgraph_cols=speechgraph,
        )
    covariates = resolve_columns(df, covariate_cols_text or DEFAULT_COVARIATE_COLS)
    return FeatureBlocks(
        code_col=code_col,
        targets=targets,
        label_col=label,
        demographics=demographics,
        cognitive=cognitive,
        speechgraph=speechgraph,
        covariates=covariates,
    )


def numeric_and_categorical(df: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            # Try coercion: if mostly numeric strings, treat as numeric.
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() >= 0.80:
                numeric.append(col)
            else:
                categorical.append(col)
    return numeric, categorical


def safe_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df[list(columns)].apply(pd.to_numeric, errors="coerce") if columns else pd.DataFrame(index=df.index)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[finite])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q[order] = adjusted
    return q
