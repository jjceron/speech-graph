from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrConfig:
    method: str = "spearman"
    min_n: int = 20
    min_abs_r: float = 0.20


def correlation_table(
    df: pd.DataFrame,
    metrics: Iterable[str],
    targets: Iterable[str],
    config: CorrConfig | None = None,
) -> pd.DataFrame:
    cfg = config or CorrConfig()
    rows: list[dict] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        x_all = pd.to_numeric(df[metric], errors="coerce")
        for target in targets:
            if target not in df.columns or target == metric:
                continue
            y_all = pd.to_numeric(df[target], errors="coerce")
            mask = x_all.notna() & y_all.notna()
            n = int(mask.sum())
            if n < cfg.min_n:
                continue
            x = x_all[mask]
            y = y_all[mask]
            if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                continue
            if cfg.method == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)
            if pd.isna(r):
                continue
            rows.append({"metric": metric, "target": target, "r": float(r), "p": float(p), "n": n})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["metric", "target", "r", "p", "n", "abs_r"])
    out["abs_r"] = out["r"].abs()
    out = out.sort_values(["abs_r", "n", "metric", "target"], ascending=[False, False, True, True])
    return out


def filter_correlations(corr: pd.DataFrame, min_abs_r: float = 0.20, min_n: int = 20) -> pd.DataFrame:
    if corr.empty:
        return corr
    out = corr.copy()
    out["r"] = pd.to_numeric(out["r"], errors="coerce")
    out["abs_r"] = out["r"].abs()
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out = out[(out["abs_r"] >= min_abs_r) & (out["n"] >= min_n)].dropna(subset=["r", "n"])
    return out.sort_values(["abs_r", "n"], ascending=[False, False])


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return values * 0
    return (values - values.mean()) / sd
