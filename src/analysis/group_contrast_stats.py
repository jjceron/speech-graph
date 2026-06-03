"""Statistical utilities for SpeechGraph high/low group contrasts.

This module is intentionally independent of statsmodels so it can run in the
same lightweight environment used by the existing SpeechGraph pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ContrastConfig:
    label_col: str
    positive_label: str
    negative_label: str | None = None
    covariates: tuple[str, ...] = ("Age", "Gender", "School year")
    min_n: int = 80
    min_group_n: int = 20
    n_bootstrap: int = 1000
    random_state: int = 42


def normalize_name(value: object) -> str:
    """Normalize a column or label name for loose matching."""
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def resolve_column(columns: Sequence[str], requested: str, required: bool = True) -> str | None:
    """Resolve a requested column name using case/space-insensitive matching."""
    if requested in columns:
        return requested
    lookup = {normalize_name(c): c for c in columns}
    key = normalize_name(requested)
    if key in lookup:
        return lookup[key]
    if required:
        raise ValueError(f"Column not found: {requested!r}. Available columns include: {list(columns)[:20]}")
    return None


def resolve_columns(columns: Sequence[str], requested: Iterable[str], required: bool = False) -> list[str]:
    resolved: list[str] = []
    for col in requested:
        found = resolve_column(columns, col, required=required)
        if found is not None and found not in resolved:
            resolved.append(found)
    return resolved


def infer_code_column(columns: Sequence[str], preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates += ["code", "Code", "CODE", "Cod", "COD", "subject", "Subject", "id", "ID"]
    for candidate in candidates:
        found = resolve_column(columns, candidate, required=False)
        if found is not None:
            return found
    raise ValueError("Could not infer subject code column. Pass --metadata-code-col or --activity-code-col.")


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    adjusted = ranked * m / (np.arange(len(ranked)) + 1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    q[valid] = tmp
    return q


def _prepare_covariate_matrix(df: pd.DataFrame, covariates: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric covariate design matrix with one-hot encoded categoricals."""
    pieces: list[pd.DataFrame] = []
    names: list[str] = []
    for col in covariates:
        if col not in df.columns:
            continue
        s = df[col]
        # Try numeric first. If conversion generates many NaNs, treat as categorical.
        numeric = pd.to_numeric(s, errors="coerce")
        if pd.api.types.is_numeric_dtype(s) or numeric.notna().sum() >= max(3, int(0.9 * s.notna().sum())):
            z = numeric.astype(float)
            if z.notna().sum() == 0 or float(z.std(ddof=0)) == 0.0:
                continue
            z = (z - z.mean()) / z.std(ddof=0)
            pieces.append(pd.DataFrame({col: z}, index=df.index))
            names.append(col)
        else:
            dummies = pd.get_dummies(s.astype("category"), prefix=col, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                pieces.append(dummies)
                names.extend(dummies.columns.tolist())
    if not pieces:
        return pd.DataFrame(index=df.index), []
    X = pd.concat(pieces, axis=1)
    return X, names


def _ols_hc3(y: np.ndarray, X: np.ndarray) -> dict[str, object]:
    """Fit OLS with HC3 robust standard errors."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n <= p + 2:
        raise ValueError("Not enough observations for OLS model.")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    # Hat diagonal. Clip to avoid division by zero in degenerate designs.
    h = np.sum((X @ xtx_inv) * X, axis=1)
    h = np.clip(h, 0.0, 0.999999)
    meat = X.T @ (((resid / (1.0 - h)) ** 2)[:, None] * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / se
    df_resid = max(1, n - p)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df_resid)
    return {
        "beta": beta,
        "se": se,
        "t": t_values,
        "p": p_values,
        "df_resid": df_resid,
        "resid": resid,
    }


def _cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float, float]:
    """Return Cohen's d and Hedges' g for positive minus negative groups."""
    n1, n0 = len(x_pos), len(x_neg)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    sd1 = np.nanstd(x_pos, ddof=1)
    sd0 = np.nanstd(x_neg, ddof=1)
    pooled_var = ((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / max(1, n1 + n0 - 2)
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan, np.nan
    d = (np.nanmean(x_pos) - np.nanmean(x_neg)) / np.sqrt(pooled_var)
    correction = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0)) if (n1 + n0) > 2 else 1.0
    return float(d), float(d * correction)


def _fit_adjusted_standardized_effect(
    work: pd.DataFrame,
    metric_col: str,
    label_col: str,
    positive_label: str,
    covariates: Sequence[str],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit metric_z ~ positive_group + covariates and return label effect."""
    y_raw = pd.to_numeric(work[metric_col], errors="coerce").astype(float)
    y_sd = float(y_raw.std(ddof=0))
    if not np.isfinite(y_sd) or y_sd <= 0:
        raise ValueError("Metric has zero variance after filtering.")
    y = ((y_raw - y_raw.mean()) / y_sd).to_numpy(dtype=float)
    label = (work[label_col].astype(str) == str(positive_label)).astype(float)
    cov_df, cov_names = _prepare_covariate_matrix(work, covariates)
    X_df = pd.concat(
        [
            pd.DataFrame({"intercept": np.ones(len(work), dtype=float)}, index=work.index),
            pd.DataFrame({"group_positive": label.to_numpy(dtype=float)}, index=work.index),
            cov_df,
        ],
        axis=1,
    )
    # Drop any residual rows with non-finite values.
    finite = np.isfinite(y) & np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1)
    y = y[finite]
    X = X_df.to_numpy(dtype=float)[finite]
    labels = label.to_numpy(dtype=float)[finite]
    if len(np.unique(labels)) < 2:
        raise ValueError("Only one label level remains after filtering.")
    fit = _ols_hc3(y, X)
    result = {
        "adj_std_beta_high": float(fit["beta"][1]),
        "adj_std_se_high": float(fit["se"][1]),
        "adj_t_high": float(fit["t"][1]),
        "adj_p_high": float(fit["p"][1]),
        "df_resid": float(fit["df_resid"]),
    }
    return result, y, X, labels, X_df.columns.tolist()


def bootstrap_label_effect(
    y: np.ndarray,
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float, int]:
    """Bootstrap CI for the group coefficient from a prepared design matrix."""
    if n_bootstrap <= 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(random_state)
    n = len(y)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            fit = _ols_hc3(y[idx], X[idx])
            value = float(fit["beta"][1])
            if np.isfinite(value):
                estimates.append(value)
        except Exception:
            continue
    if not estimates:
        return np.nan, np.nan, np.nan, 0
    arr = np.asarray(estimates, dtype=float)
    return float(np.nanmean(arr)), float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)), int(len(arr))


def run_group_contrasts(
    data: pd.DataFrame,
    metrics: Sequence[str],
    config: ContrastConfig,
    activity_col: str = "activity_number",
    activity_name_col: str = "activity",
    window_col: str = "window_size",
) -> pd.DataFrame:
    """Run adjusted high/low group contrasts for each activity/window/metric."""
    label_col = resolve_column(data.columns, config.label_col)
    covariates = resolve_columns(data.columns, config.covariates, required=False)
    activity_col = resolve_column(data.columns, activity_col)
    window_col = resolve_column(data.columns, window_col)
    activity_name_col = resolve_column(data.columns, activity_name_col, required=False) or activity_col
    metric_cols = resolve_columns(data.columns, metrics, required=False)
    if not metric_cols:
        raise ValueError("No requested SpeechGraph metric columns were found in the input table.")

    records: list[dict[str, object]] = []
    grouped = data.groupby([activity_col, window_col], dropna=False, sort=True)
    for (activity_number, window_size), scheme_df in grouped:
        activity_name = scheme_df[activity_name_col].dropna().astype(str).iloc[0] if activity_name_col in scheme_df else str(activity_number)
        for metric in metric_cols:
            cols = [label_col, metric, activity_col, window_col] + covariates
            work = scheme_df[cols].copy()
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
            work = work.dropna(subset=[label_col, metric])
            # Drop rows with missing covariates used by the adjusted model.
            if covariates:
                work = work.dropna(subset=covariates)
            # Keep only requested labels.
            labels_as_str = work[label_col].astype(str)
            keep = labels_as_str == str(config.positive_label)
            if config.negative_label is not None:
                keep = keep | (labels_as_str == str(config.negative_label))
            else:
                keep = keep | (labels_as_str != str(config.positive_label))
            work = work.loc[keep].copy()
            labels_as_str = work[label_col].astype(str)
            n_pos = int((labels_as_str == str(config.positive_label)).sum())
            n_neg = int(len(work) - n_pos)
            n = int(len(work))
            base: dict[str, object] = {
                "activity_number": activity_number,
                "activity": activity_name,
                "window_size": window_size,
                "metric": metric,
                "n": n,
                "n_high": n_pos,
                "n_low": n_neg,
                "positive_label": config.positive_label,
                "negative_label": config.negative_label if config.negative_label is not None else "not_positive",
                "covariates": ",".join(covariates),
            }
            if n < config.min_n or n_pos < config.min_group_n or n_neg < config.min_group_n:
                base.update({"status": "skipped_min_n"})
                records.append(base)
                continue
            x_pos = work.loc[labels_as_str == str(config.positive_label), metric].to_numpy(dtype=float)
            x_neg = work.loc[labels_as_str != str(config.positive_label), metric].to_numpy(dtype=float)
            if np.nanstd(work[metric].to_numpy(dtype=float), ddof=0) <= 0:
                base.update({"status": "skipped_zero_variance"})
                records.append(base)
                continue
            cohen_d, hedges_g = _cohens_d(x_pos, x_neg)
            try:
                mw = stats.mannwhitneyu(x_pos, x_neg, alternative="two-sided")
                mw_p = float(mw.pvalue)
            except Exception:
                mw_p = np.nan
            try:
                adj, y, X, labels, design_cols = _fit_adjusted_standardized_effect(
                    work=work,
                    metric_col=metric,
                    label_col=label_col,
                    positive_label=config.positive_label,
                    covariates=covariates,
                )
                boot_mean, boot_low, boot_high, boot_n = bootstrap_label_effect(
                    y=y,
                    X=X,
                    labels=labels,
                    n_bootstrap=config.n_bootstrap,
                    random_state=config.random_state + len(records) * 17,
                )
                base.update(
                    {
                        "status": "ok",
                        "mean_high": float(np.nanmean(x_pos)),
                        "mean_low": float(np.nanmean(x_neg)),
                        "sd_high": float(np.nanstd(x_pos, ddof=1)),
                        "sd_low": float(np.nanstd(x_neg, ddof=1)),
                        "diff_high_minus_low": float(np.nanmean(x_pos) - np.nanmean(x_neg)),
                        "cohen_d_high_minus_low": cohen_d,
                        "hedges_g_high_minus_low": hedges_g,
                        "mannwhitney_p": mw_p,
                        **adj,
                        "bootstrap_mean_adj_std_beta": boot_mean,
                        "bootstrap_ci_low": boot_low,
                        "bootstrap_ci_high": boot_high,
                        "bootstrap_success_n": boot_n,
                        "design_columns": ",".join(design_cols),
                    }
                )
            except Exception as exc:
                base.update({"status": f"skipped_model_error: {exc}"})
            records.append(base)

    out = pd.DataFrame.from_records(records)
    if "adj_p_high" in out.columns:
        ok = out["status"].eq("ok") & out["adj_p_high"].notna()
        out.loc[ok, "adj_q_fdr_global"] = bh_fdr(out.loc[ok, "adj_p_high"].to_numpy(dtype=float))
        out.loc[ok, "mw_q_fdr_global"] = bh_fdr(out.loc[ok, "mannwhitney_p"].to_numpy(dtype=float))
        # FDR within metric and within activity-window are helpful diagnostics.
        out["adj_q_fdr_by_metric"] = np.nan
        out["adj_q_fdr_by_scheme"] = np.nan
        for _, idx in out.loc[ok].groupby("metric").groups.items():
            out.loc[idx, "adj_q_fdr_by_metric"] = bh_fdr(out.loc[idx, "adj_p_high"].to_numpy(dtype=float))
        for _, idx in out.loc[ok].groupby(["activity_number", "window_size"]).groups.items():
            out.loc[idx, "adj_q_fdr_by_scheme"] = bh_fdr(out.loc[idx, "adj_p_high"].to_numpy(dtype=float))
        out.loc[ok, "neg_log10_adj_q"] = -np.log10(np.clip(out.loc[ok, "adj_q_fdr_global"].to_numpy(dtype=float), 1e-300, 1.0))
        out.loc[ok, "ci_excludes_zero"] = (
            (out.loc[ok, "bootstrap_ci_low"] > 0) | (out.loc[ok, "bootstrap_ci_high"] < 0)
        )
        out.loc[ok, "significant_global_fdr_05"] = out.loc[ok, "adj_q_fdr_global"] < 0.05
        out.loc[ok, "candidate_marker"] = out.loc[ok, "ci_excludes_zero"].fillna(False) & (out.loc[ok, "adj_q_fdr_global"] < 0.10)
    return out


def summarize_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary by activity/window."""
    ok = results[results.get("status", "").eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for (activity, window), g in ok.groupby(["activity_number", "window_size"], sort=True):
        top = g.reindex(g["adj_std_beta_high"].abs().sort_values(ascending=False).index).head(1)
        rows.append(
            {
                "activity_number": activity,
                "window_size": window,
                "n_metrics_tested": int(len(g)),
                "n_global_fdr_05": int((g["adj_q_fdr_global"] < 0.05).sum()),
                "n_global_fdr_10": int((g["adj_q_fdr_global"] < 0.10).sum()),
                "n_ci_excludes_zero": int(g["ci_excludes_zero"].fillna(False).sum()),
                "top_metric": str(top["metric"].iloc[0]) if len(top) else None,
                "top_abs_adj_beta": float(top["adj_std_beta_high"].abs().iloc[0]) if len(top) else np.nan,
                "top_adj_beta": float(top["adj_std_beta_high"].iloc[0]) if len(top) else np.nan,
                "top_q_global": float(top["adj_q_fdr_global"].iloc[0]) if len(top) else np.nan,
            }
        )
    return pd.DataFrame(rows)
