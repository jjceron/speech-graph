"""Composite SpeechGraph index statistics for 05_run.

This module implements a focused, confirmatory follow-up to the broad 04_run
metric-wise group-contrast analysis. It builds a small number of pre-specified
SpeechGraph composite indices and evaluates high_imp vs low_imp differences
while adjusting for covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


@dataclass(frozen=True)
class IndexSpec:
    """Definition of a composite SpeechGraph index.

    positive_features are averaged after transformation. Negative features are
    averaged after transformation and subtracted from the positive part.
    """

    name: str
    positive_features: Tuple[str, ...]
    negative_features: Tuple[str, ...] = ()
    hypothesis_direction: str = "high_gt_low"
    description: str = ""
    family: str = "primary"


@dataclass(frozen=True)
class ModelSpec:
    """Covariate model definition."""

    name: str
    covariates: Tuple[str, ...]


def default_index_specs(include_secondary_global: bool = False) -> List[IndexSpec]:
    """Return the pre-specified 05_run composite indices."""
    specs: List[IndexSpec] = [
        IndexSpec(
            name="A2_recurrence_index",
            positive_features=(
                "A2_W10_pe",
                "A2_W20_pe",
                "A2_W30_pe",
                "A2_W10_l2",
                "A2_W20_l2",
                "A2_W30_l2",
            ),
            description=(
                "Activity 2 recurrence index: parallel/repeated transitions "
                "and two-step cycles across W10/W20/W30."
            ),
            family="primary",
        ),
        IndexSpec(
            name="A1_W20_compactness_index",
            positive_features=("A1_W20_atd", "A1_W20_density"),
            negative_features=("A1_W20_nodes", "A1_W20_asp", "A1_W20_diameter"),
            description=(
                "Activity 1 W20 compactness index: higher degree/density and "
                "lower graph size/path extent."
            ),
            family="primary",
        ),
        IndexSpec(
            name="A4_compact_recurrence_index",
            positive_features=(
                "A4_W10_re",
                "A4_W20_re",
                "A4_W30_re",
                "A4_W10_density",
                "A4_W20_density",
                "A4_W30_density",
                "A4_W10_l2",
                "A4_W20_l2",
                "A4_W30_l2",
            ),
            negative_features=(
                "A4_W10_nodes",
                "A4_W20_nodes",
                "A4_W30_nodes",
                "A4_W10_asp",
                "A4_W20_asp",
                "A4_W30_asp",
                "A4_W10_diameter",
                "A4_W20_diameter",
                "A4_W30_diameter",
            ),
            description=(
                "Activity 4 compact-recurrence index: recurrence/density/cycles "
                "minus graph size/path extent across W10/W20/W30."
            ),
            family="primary",
        ),
    ]
    if include_secondary_global:
        pos = []
        neg = []
        for activity in (1, 2, 4, 6):
            for window in (10, 20, 30):
                for metric in ("re", "pe", "l2", "density", "atd"):
                    pos.append(f"A{activity}_W{window}_{metric}")
                for metric in ("nodes", "asp", "diameter"):
                    neg.append(f"A{activity}_W{window}_{metric}")
        specs.append(
            IndexSpec(
                name="global_compact_recurrence_index",
                positive_features=tuple(pos),
                negative_features=tuple(neg),
                description=(
                    "Secondary broad compact-recurrence index across the most "
                    "supported activities."
                ),
                family="secondary",
            )
        )
    return specs


def default_model_specs(
    primary_covariates: Sequence[str],
    sensitivity_age_covariates: Sequence[str],
    sensitivity_full_covariates: Sequence[str],
) -> List[ModelSpec]:
    return [
        ModelSpec("primary_school_year_gender", tuple(primary_covariates)),
        ModelSpec("sensitivity_age_gender", tuple(sensitivity_age_covariates)),
        ModelSpec("sensitivity_age_school_year_gender", tuple(sensitivity_full_covariates)),
    ]


def resolve_column(columns: Iterable[str], requested: str) -> str:
    """Resolve a column name exactly or case/space-insensitively."""
    cols = list(columns)
    if requested in cols:
        return requested
    normalized = {str(c).strip().lower(): c for c in cols}
    key = requested.strip().lower()
    if key in normalized:
        return normalized[key]
    compact = {str(c).replace(" ", "").strip().lower(): c for c in cols}
    ckey = requested.replace(" ", "").strip().lower()
    if ckey in compact:
        return compact[ckey]
    raise ValueError(
        f"Column not found: {requested!r}. Available columns include: {cols[:30]}"
    )


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[mask] = out
    return q


def rank_inverse_normal(series: pd.Series) -> pd.Series:
    """Rank-based inverse normal transform preserving NaNs."""
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = x.notna()
    n = int(mask.sum())
    if n < 2:
        return out
    if stats is None:
        # Fallback to a standard z-score when scipy is unavailable.
        vals = x[mask].astype(float)
        sd = vals.std(ddof=0)
        out.loc[mask] = 0.0 if sd == 0 else (vals - vals.mean()) / sd
        return out
    ranks = stats.rankdata(x[mask].to_numpy(dtype=float), method="average")
    probs = (ranks - 0.5) / n
    out.loc[mask] = stats.norm.ppf(probs)
    return out


def winsor_z(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize and z-score a numeric series."""
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = x.notna()
    if mask.sum() < 2:
        return out
    lo = x[mask].quantile(lower)
    hi = x[mask].quantile(upper)
    vals = x[mask].clip(lo, hi)
    sd = vals.std(ddof=0)
    out.loc[mask] = 0.0 if sd == 0 else (vals - vals.mean()) / sd
    return out


def standard_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = x.notna()
    if mask.sum() < 2:
        return out
    vals = x[mask]
    sd = vals.std(ddof=0)
    out.loc[mask] = 0.0 if sd == 0 else (vals - vals.mean()) / sd
    return out


def transform_feature(series: pd.Series, method: str) -> pd.Series:
    method = method.lower().strip()
    if method in {"rank", "rank_normal", "rank_inverse_normal", "rin"}:
        return rank_inverse_normal(series)
    if method in {"winsor_z", "winsorized_z"}:
        return winsor_z(series)
    if method in {"z", "zscore", "standard_z"}:
        return standard_z(series)
    raise ValueError(f"Unknown transform method: {method!r}")


def build_speechgraph_wide(
    activity_df: pd.DataFrame,
    metrics: Sequence[str],
    id_col: str = "code",
    activity_col: str = "activity_number",
    window_col: str = "window_size",
) -> pd.DataFrame:
    """Convert long activity-window SpeechGraph features to one row per subject."""
    id_col = resolve_column(activity_df.columns, id_col)
    activity_col = resolve_column(activity_df.columns, activity_col)
    window_col = resolve_column(activity_df.columns, window_col)
    rows = []
    for _, row in activity_df.iterrows():
        code = row[id_col]
        activity = int(row[activity_col]) if pd.notna(row[activity_col]) else row[activity_col]
        window = int(row[window_col]) if pd.notna(row[window_col]) else row[window_col]
        out = {id_col: code}
        for metric in metrics:
            if metric in activity_df.columns:
                out[f"A{activity}_W{window}_{metric}"] = row[metric]
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=[id_col])
    wide = pd.DataFrame(rows).groupby(id_col, as_index=False).first()
    return wide


def build_subject_table(
    activity_df: pd.DataFrame,
    metrics: Sequence[str],
    label_col: str,
    covariates: Sequence[str],
    id_col: str = "code",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build subject-level table with metadata/covariates and SpeechGraph wide features."""
    id_col = resolve_column(activity_df.columns, id_col)
    label_col_res = resolve_column(activity_df.columns, label_col)
    cov_resolved = [resolve_column(activity_df.columns, c) for c in covariates]
    meta_cols = [id_col, label_col_res] + cov_resolved
    meta = activity_df[meta_cols].drop_duplicates(id_col).copy()
    wide = build_speechgraph_wide(activity_df, metrics=metrics, id_col=id_col)
    subject = meta.merge(wide, on=id_col, how="left")
    manifest = {
        "id_col": id_col,
        "label_col": label_col_res,
        "covariates": cov_resolved,
        "metrics": list(metrics),
        "n_subjects": int(subject[id_col].nunique()),
        "n_rows": int(len(subject)),
        "n_speechgraph_features": int(len([c for c in subject.columns if c.startswith("A") and "_W" in c])),
    }
    return subject, manifest


def compute_indices(
    subject_df: pd.DataFrame,
    specs: Sequence[IndexSpec],
    transform: str = "rank_normal",
    min_component_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute composite index values for each subject."""
    out = subject_df.copy()
    component_rows = []
    transformed_cache: Dict[str, pd.Series] = {}

    all_features = sorted({f for s in specs for f in (s.positive_features + s.negative_features)})
    missing_features = [f for f in all_features if f not in subject_df.columns]
    if missing_features:
        warnings.warn(f"Missing SpeechGraph features for index construction: {missing_features[:20]}")

    for f in all_features:
        if f in subject_df.columns:
            transformed_cache[f] = transform_feature(subject_df[f], transform)

    for spec in specs:
        pos_available = [f for f in spec.positive_features if f in transformed_cache]
        neg_available = [f for f in spec.negative_features if f in transformed_cache]
        expected_n = len(spec.positive_features) + len(spec.negative_features)
        available_n = len(pos_available) + len(neg_available)
        min_n = max(1, int(math.ceil(expected_n * min_component_fraction)))

        pos_mat = pd.DataFrame({f: transformed_cache[f] for f in pos_available}) if pos_available else pd.DataFrame(index=subject_df.index)
        neg_mat = pd.DataFrame({f: transformed_cache[f] for f in neg_available}) if neg_available else pd.DataFrame(index=subject_df.index)
        component_count = pos_mat.notna().sum(axis=1) + neg_mat.notna().sum(axis=1)
        pos_mean = pos_mat.mean(axis=1, skipna=True) if pos_available else pd.Series(0.0, index=subject_df.index)
        neg_mean = neg_mat.mean(axis=1, skipna=True) if neg_available else pd.Series(0.0, index=subject_df.index)
        values = pos_mean - neg_mean
        values = values.where(component_count >= min_n, np.nan)
        # Standardize final index for coefficient comparability.
        values = standard_z(values)
        out[spec.name] = values
        out[f"{spec.name}_component_count"] = component_count

        component_rows.append(
            {
                "index": spec.name,
                "family": spec.family,
                "description": spec.description,
                "positive_features": ",".join(spec.positive_features),
                "negative_features": ",".join(spec.negative_features),
                "expected_components": expected_n,
                "available_components": available_n,
                "minimum_required_components": min_n,
                "n_nonmissing_index": int(values.notna().sum()),
                "transform": transform,
                "min_component_fraction": min_component_fraction,
                "hypothesis_direction": spec.hypothesis_direction,
            }
        )
    return out, pd.DataFrame(component_rows)


def _prepare_design(
    df: pd.DataFrame,
    outcome_col: str,
    label_col: str,
    positive_label: str,
    negative_label: str,
    covariates: Sequence[str],
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    cols = [outcome_col, label_col] + list(covariates)
    d = df[cols].copy()
    d = d[d[label_col].isin([positive_label, negative_label])]
    d = d.dropna(subset=[outcome_col, label_col])
    y = pd.to_numeric(d[outcome_col], errors="coerce")
    d = d.loc[y.notna()].copy()
    y = y.loc[d.index].astype(float)
    group = (d[label_col] == positive_label).astype(float)
    design = pd.DataFrame({"intercept": 1.0, "group_positive": group}, index=d.index)
    for cov in covariates:
        s = d[cov]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() >= max(3, int(0.8 * len(s))):
            vals = num.astype(float)
            # Mean-impute numeric covariates within the analysis set.
            vals = vals.fillna(vals.mean())
            sd = vals.std(ddof=0)
            design[cov] = 0.0 if sd == 0 else (vals - vals.mean()) / sd
        else:
            cat = s.astype("category").cat.add_categories("__MISSING__").fillna("__MISSING__")
            dummies = pd.get_dummies(cat, prefix=cov, drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
    return y, design.astype(float), group, d


def _ols_fit(y: np.ndarray, X: np.ndarray, beta_index: int = 1) -> Dict[str, float | np.ndarray]:
    n, p = X.shape
    if n <= p + 1:
        raise ValueError("Insufficient observations for OLS model.")
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    df = n - p
    rss = float(np.sum(resid**2))
    sigma2 = rss / df if df > 0 else np.nan
    cov_classic = xtx_inv * sigma2
    se_classic = np.sqrt(np.maximum(np.diag(cov_classic), 0))

    # HC3 robust covariance.
    h = np.sum((X @ xtx_inv) * X, axis=1)
    denom = np.maximum(1.0 - h, 1e-8)
    scaled = (resid / denom) ** 2
    meat = X.T @ (X * scaled[:, None])
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.maximum(np.diag(cov_hc3), 0))

    b = float(beta[beta_index])
    se = float(se_hc3[beta_index])
    t = b / se if se > 0 else np.nan
    if stats is not None and np.isfinite(t):
        p_two = float(2 * stats.t.sf(abs(t), df))
        p_greater = float(stats.t.sf(t, df))
        p_less = float(stats.t.cdf(t, df))
    else:
        p_two = np.nan
        p_greater = np.nan
        p_less = np.nan
    return {
        "beta": b,
        "se_hc3": se,
        "t_hc3": t,
        "p_two_sided": p_two,
        "p_one_sided_greater": p_greater,
        "p_one_sided_less": p_less,
        "df_resid": float(df),
        "n": float(n),
        "p": float(p),
        "beta_all": beta,
        "fitted": fitted,
        "resid": resid,
    }


def _fit_reduced(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Drop group column at index 1, keep intercept/covariates.
    Xr = np.delete(X, 1, axis=1)
    beta = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fitted = Xr @ beta
    resid = y - fitted
    return fitted, resid


def _permutation_p(
    y: np.ndarray,
    X: np.ndarray,
    observed_beta: float,
    n_permutations: int,
    rng: np.random.Generator,
    alternative: str = "greater",
) -> Tuple[float, float, float]:
    if n_permutations <= 0:
        return np.nan, np.nan, np.nan
    fitted_red, resid_red = _fit_reduced(y, X)
    betas = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = fitted_red + rng.permutation(resid_red)
        try:
            fit = _ols_fit(y_perm, X)
            betas[i] = fit["beta"]  # type: ignore[index]
        except Exception:
            betas[i] = np.nan
    valid = np.isfinite(betas)
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan
    b = betas[valid]
    p_greater = float((np.sum(b >= observed_beta) + 1) / (len(b) + 1))
    p_less = float((np.sum(b <= observed_beta) + 1) / (len(b) + 1))
    p_two = float((np.sum(np.abs(b) >= abs(observed_beta)) + 1) / (len(b) + 1))
    return p_greater, p_less, p_two


def _bootstrap_ci(
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, int]:
    if n_bootstrap <= 0:
        return np.nan, np.nan, np.nan, 0
    n = len(y)
    betas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            fit = _ols_fit(y[idx], X[idx, :])
            b = fit["beta"]  # type: ignore[index]
            if np.isfinite(b):
                betas.append(float(b))
        except Exception:
            continue
    if not betas:
        return np.nan, np.nan, np.nan, 0
    arr = np.asarray(betas)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), int(len(arr))


def hedges_g(y: pd.Series, group: pd.Series) -> Tuple[float, float, float, float, float]:
    yp = y[group == 1]
    yn = y[group == 0]
    n1, n0 = len(yp), len(yn)
    m1, m0 = float(yp.mean()), float(yn.mean())
    s1, s0 = float(yp.std(ddof=1)), float(yn.std(ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max(n1 + n0 - 2, 1))
    d = (m1 - m0) / pooled if pooled > 0 else np.nan
    correction = 1 - 3 / (4 * (n1 + n0) - 9) if n1 + n0 > 2 else 1
    return float(d * correction), m1, m0, s1, s0


def run_index_contrasts(
    df: pd.DataFrame,
    index_specs: Sequence[IndexSpec],
    model_specs: Sequence[ModelSpec],
    label_col: str,
    positive_label: str,
    negative_label: str,
    primary_model_name: str,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    random_state: int = 42,
    alternative: str = "greater",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit covariate-adjusted group models for all composite indices."""
    rng = np.random.default_rng(random_state)
    label_col = resolve_column(df.columns, label_col)
    rows = []
    for spec in index_specs:
        if spec.name not in df.columns:
            raise ValueError(f"Index column not found: {spec.name}")
        for model in model_specs:
            covs = [resolve_column(df.columns, c) for c in model.covariates]
            y, design, group, analysis_data = _prepare_design(
                df,
                outcome_col=spec.name,
                label_col=label_col,
                positive_label=positive_label,
                negative_label=negative_label,
                covariates=covs,
            )
            X = design.to_numpy(dtype=float)
            yv = y.to_numpy(dtype=float)
            fit = _ols_fit(yv, X)
            beta = float(fit["beta"])
            p_greater, p_less, p_two_perm = _permutation_p(
                yv, X, beta, n_permutations=n_permutations, rng=rng, alternative=alternative
            )
            boot_mean, boot_lo, boot_hi, boot_n = _bootstrap_ci(
                yv, X, n_bootstrap=n_bootstrap, rng=rng
            )
            g, mean_high, mean_low, sd_high, sd_low = hedges_g(y, group)
            ci_excludes_zero = bool(np.isfinite(boot_lo) and np.isfinite(boot_hi) and (boot_lo > 0 or boot_hi < 0))
            rows.append(
                {
                    "index": spec.name,
                    "family": spec.family,
                    "description": spec.description,
                    "model": model.name,
                    "covariates": ",".join(covs),
                    "n": int(len(y)),
                    "n_high": int(group.sum()),
                    "n_low": int((group == 0).sum()),
                    "positive_label": positive_label,
                    "negative_label": negative_label,
                    "mean_high": mean_high,
                    "mean_low": mean_low,
                    "sd_high": sd_high,
                    "sd_low": sd_low,
                    "diff_high_minus_low": mean_high - mean_low,
                    "hedges_g_high_minus_low": g,
                    "adj_beta_high": beta,
                    "adj_se_hc3": float(fit["se_hc3"]),
                    "adj_t_hc3": float(fit["t_hc3"]),
                    "adj_p_two_sided_hc3": float(fit["p_two_sided"]),
                    "adj_p_one_sided_greater_hc3": float(fit["p_one_sided_greater"]),
                    "adj_p_one_sided_less_hc3": float(fit["p_one_sided_less"]),
                    "perm_p_one_sided_greater": p_greater,
                    "perm_p_one_sided_less": p_less,
                    "perm_p_two_sided": p_two_perm,
                    "bootstrap_mean_beta": boot_mean,
                    "bootstrap_ci_low": boot_lo,
                    "bootstrap_ci_high": boot_hi,
                    "bootstrap_success_n": boot_n,
                    "df_resid": float(fit["df_resid"]),
                    "design_columns": ",".join(design.columns),
                    "ci_excludes_zero": ci_excludes_zero,
                    "hypothesis_direction": spec.hypothesis_direction,
                }
            )
    results = pd.DataFrame(rows)

    # FDR correction primarily across primary family indices in the primary model.
    results["primary_family_primary_model"] = (
        (results["family"] == "primary") & (results["model"] == primary_model_name)
    )
    primary_mask = results["primary_family_primary_model"]
    results["perm_q_primary_family"] = np.nan
    results.loc[primary_mask, "perm_q_primary_family"] = fdr_bh(
        results.loc[primary_mask, "perm_p_one_sided_greater"].to_numpy(dtype=float)
    )
    results["hc3_q_primary_family"] = np.nan
    results.loc[primary_mask, "hc3_q_primary_family"] = fdr_bh(
        results.loc[primary_mask, "adj_p_one_sided_greater_hc3"].to_numpy(dtype=float)
    )

    # Sensitivity summary: same direction and CI status across all requested models.
    sens_rows = []
    for spec in index_specs:
        sub = results[results["index"] == spec.name].copy()
        primary = sub[sub["model"] == primary_model_name]
        primary_beta = float(primary["adj_beta_high"].iloc[0]) if not primary.empty else np.nan
        all_positive = bool((sub["adj_beta_high"] > 0).all()) if len(sub) else False
        all_ci_positive = bool((sub["bootstrap_ci_low"] > 0).all()) if len(sub) else False
        sens_rows.append(
            {
                "index": spec.name,
                "family": spec.family,
                "primary_model_beta": primary_beta,
                "all_models_positive_beta": all_positive,
                "all_models_ci_above_zero": all_ci_positive,
                "n_models": int(len(sub)),
                "models": ",".join(sub["model"].tolist()),
            }
        )
    return results, pd.DataFrame(sens_rows)
