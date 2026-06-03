from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from src.analysis.profile_preprocessing import FeatureBlocks, benjamini_hochberg, numeric_and_categorical


def _design_from_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.ones((len(df), 1), dtype=float)
    numeric, categorical = numeric_and_categorical(df, columns)
    mats = []
    if numeric:
        x = df[numeric].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True).fillna(0.0))
        sd = x.std(skipna=True).replace(0, 1.0)
        x = (x - x.mean(skipna=True)) / sd
        mats.append(x.to_numpy(dtype=float))
    for col in categorical:
        series = df[col].astype("object")
        mode = series.dropna().mode()
        fill = mode.iloc[0] if len(mode) else "missing"
        dummies = pd.get_dummies(series.fillna(fill).astype(str), prefix=col, drop_first=True)
        if not dummies.empty:
            mats.append(dummies.to_numpy(dtype=float))
    if not mats:
        return np.ones((len(df), 1), dtype=float)
    return np.column_stack([np.ones(len(df)), *mats])


def _profile_design(labels: pd.Series) -> np.ndarray:
    dummies = pd.get_dummies(labels.astype(str), prefix="profile", drop_first=True)
    if dummies.empty:
        return np.ones((len(labels), 1), dtype=float)
    return np.column_stack([np.ones(len(labels)), dummies.to_numpy(dtype=float)])


def _combine_design(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.column_stack([x1, x2[:, 1:]]) if x2.shape[1] > 1 else x1


def _ols_stats(y: np.ndarray, x: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    n = len(y)
    if n <= x.shape[1] + 1:
        return {"n": n, "p": x.shape[1], "rss": np.nan, "r2": np.nan, "adj_r2": np.nan, "df_resid": np.nan}
    beta = np.linalg.pinv(x) @ y
    pred = x @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    p = x.shape[1]
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1) if np.isfinite(r2) else np.nan
    return {"n": n, "p": p, "rss": rss, "r2": float(r2), "adj_r2": float(adj), "df_resid": int(n - p)}


def _f_test_nested(y: np.ndarray, x_reduced: np.ndarray, x_full: np.ndarray) -> dict:
    red = _ols_stats(y, x_reduced)
    full = _ols_stats(y, x_full)
    if not np.isfinite(red["rss"]) or not np.isfinite(full["rss"]):
        return {"f": np.nan, "p": np.nan, "df_num": np.nan, "df_den": np.nan, **{f"reduced_{k}": v for k, v in red.items()}, **{f"full_{k}": v for k, v in full.items()}}
    df_num = max(full["p"] - red["p"], 1)
    df_den = full["df_resid"]
    if df_den <= 0:
        f_stat, p_val = np.nan, np.nan
    else:
        numerator = max(red["rss"] - full["rss"], 0.0) / df_num
        denominator = full["rss"] / df_den
        f_stat = numerator / denominator if denominator > 0 else np.nan
        p_val = float(stats.f.sf(f_stat, df_num, df_den)) if np.isfinite(f_stat) else np.nan
    return {
        "f": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "p": p_val,
        "df_num": df_num,
        "df_den": df_den,
        **{f"reduced_{k}": v for k, v in red.items()},
        **{f"full_{k}": v for k, v in full.items()},
    }


def _eta_squared_from_groups(groups: list[np.ndarray]) -> tuple[float, float, float]:
    clean = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 0]
    if len(clean) < 2:
        return np.nan, np.nan, np.nan
    all_y = np.concatenate(clean)
    grand = np.mean(all_y)
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in clean)
    ss_total = np.sum((all_y - grand) ** 2)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    f_value, p_value = stats.f_oneway(*clean)
    return float(eta2), float(f_value), float(p_value)


def validate_one_profile_set(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: FeatureBlocks,
    profile_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = features.merge(assignments[[blocks.code_col, "profile", "profile_probability_max"]], on=blocks.code_col, how="inner")
    barratt_rows = []
    for target in blocks.targets:
        if target not in data.columns:
            continue
        y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
        groups = [pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float) for _, sub in data.groupby("profile")]
        eta2, f_value, anova_p = _eta_squared_from_groups(groups)
        try:
            kruskal = stats.kruskal(*[g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 0])
            kw_h, kw_p = float(kruskal.statistic), float(kruskal.pvalue)
        except Exception:
            kw_h, kw_p = np.nan, np.nan
        cov_design = _design_from_columns(data, blocks.covariates)
        prof_design = _profile_design(data["profile"])
        full_design = _combine_design(cov_design, prof_design)
        adjusted = _f_test_nested(y, cov_design, full_design)
        barratt_rows.append({
            "profile_set": profile_set,
            "target": target,
            "n": int(np.isfinite(y).sum()),
            "profiles": int(data["profile"].nunique()),
            "eta_squared": eta2,
            "anova_f": f_value,
            "anova_p": anova_p,
            "kruskal_h": kw_h,
            "kruskal_p": kw_p,
            "adjusted_profile_f": adjusted["f"],
            "adjusted_profile_p": adjusted["p"],
            "covariate_r2": adjusted["reduced_r2"],
            "covariate_adj_r2": adjusted["reduced_adj_r2"],
            "covariate_plus_profile_r2": adjusted["full_r2"],
            "covariate_plus_profile_adj_r2": adjusted["full_adj_r2"],
            "delta_adj_r2_over_covariates": adjusted["full_adj_r2"] - adjusted["reduced_adj_r2"] if np.isfinite(adjusted["full_adj_r2"]) and np.isfinite(adjusted["reduced_adj_r2"]) else np.nan,
        })
    barratt = pd.DataFrame(barratt_rows)
    if not barratt.empty:
        barratt["anova_q_fdr"] = benjamini_hochberg(barratt["anova_p"])
        barratt["kruskal_q_fdr"] = benjamini_hochberg(barratt["kruskal_p"])
        barratt["adjusted_profile_q_fdr"] = benjamini_hochberg(barratt["adjusted_profile_p"])

    enrichment_rows = []
    if blocks.label_col and blocks.label_col in data.columns:
        table = pd.crosstab(data["profile"], data[blocks.label_col])
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(table)
            n = table.to_numpy().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 and min(table.shape) > 1 else np.nan
        else:
            chi2, p, dof, cramers_v = np.nan, np.nan, np.nan, np.nan
        for profile, row in table.iterrows():
            total = int(row.sum())
            for label, count in row.items():
                enrichment_rows.append({
                    "profile_set": profile_set,
                    "profile": profile,
                    "label_col": blocks.label_col,
                    "label": label,
                    "count": int(count),
                    "profile_n": total,
                    "profile_proportion": float(count / total) if total else np.nan,
                    "chi2": float(chi2) if np.isfinite(chi2) else np.nan,
                    "chi2_p": float(p) if np.isfinite(p) else np.nan,
                    "chi2_dof": int(dof) if np.isfinite(dof) else np.nan,
                    "cramers_v": float(cramers_v) if np.isfinite(cramers_v) else np.nan,
                })
    enrichment = pd.DataFrame(enrichment_rows)
    return barratt, enrichment


def _comparison_rows_for_target(
    data: pd.DataFrame,
    blocks: FeatureBlocks,
    baseline_col: str,
    comparison_col: str,
    comparison_name: str,
    target: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=float)
    cov_design = _design_from_columns(data, blocks.covariates)
    baseline_design = _combine_design(cov_design, _profile_design(data[baseline_col]))
    comparison_design = _combine_design(cov_design, _profile_design(data[comparison_col]))
    cov_stats = _ols_stats(y, cov_design)
    base_stats = _ols_stats(y, baseline_design)
    comp_stats = _ols_stats(y, comparison_design)
    observed = comp_stats["adj_r2"] - base_stats["adj_r2"] if np.isfinite(comp_stats["adj_r2"]) and np.isfinite(base_stats["adj_r2"]) else np.nan
    null = []
    valid_mask = np.isfinite(y)
    for _ in range(n_permutations):
        yp = y.copy()
        yp[valid_mask] = rng.permutation(yp[valid_mask])
        bs = _ols_stats(yp, baseline_design)
        cs = _ols_stats(yp, comparison_design)
        if np.isfinite(bs["adj_r2"]) and np.isfinite(cs["adj_r2"]):
            null.append(cs["adj_r2"] - bs["adj_r2"])
    null = np.asarray(null, dtype=float)
    p_perm = (1.0 + np.sum(null >= observed)) / (1.0 + len(null)) if np.isfinite(observed) and len(null) else np.nan
    return {
        "comparison": comparison_name,
        "target": target,
        "n": int(np.isfinite(y).sum()),
        "covariate_adj_r2": cov_stats["adj_r2"],
        "baseline_profile_adj_r2": base_stats["adj_r2"],
        "comparison_profile_adj_r2": comp_stats["adj_r2"],
        "baseline_delta_adj_r2_over_covariates": base_stats["adj_r2"] - cov_stats["adj_r2"] if np.isfinite(base_stats["adj_r2"]) and np.isfinite(cov_stats["adj_r2"]) else np.nan,
        "comparison_delta_adj_r2_over_covariates": comp_stats["adj_r2"] - cov_stats["adj_r2"] if np.isfinite(comp_stats["adj_r2"]) and np.isfinite(cov_stats["adj_r2"]) else np.nan,
        "delta_adj_r2_comparison_minus_baseline": observed,
        "permutation_p_one_sided": p_perm,
        "n_permutations": int(len(null)),
    }


def compare_profile_sets_vs_metadata(
    features: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    blocks: FeatureBlocks,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if "metadata" not in assignments:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    base = assignments["metadata"][[blocks.code_col, "profile"]].rename(columns={"profile": "profile_metadata"})
    data = features.merge(base, on=blocks.code_col, how="inner")
    comparison_sets = [name for name in assignments if name != "metadata"]
    for name in comparison_sets:
        comp = assignments[name][[blocks.code_col, "profile"]].rename(columns={"profile": f"profile_{name}"})
        data = data.merge(comp, on=blocks.code_col, how="inner")
    rows = []
    for name in comparison_sets:
        col = f"profile_{name}"
        ari = adjusted_rand_score(data["profile_metadata"], data[col])
        same_label_prop = float((data["profile_metadata"] == data[col]).mean())
        for target in blocks.targets:
            if target not in data.columns:
                continue
            row = _comparison_rows_for_target(
                data=data,
                blocks=blocks,
                baseline_col="profile_metadata",
                comparison_col=col,
                comparison_name=f"{name}_vs_metadata",
                target=target,
                n_permutations=n_permutations,
                rng=rng,
            )
            row["profile_ari_vs_metadata"] = float(ari)
            row["same_label_proportion_vs_metadata"] = same_label_prop
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["permutation_q_fdr"] = benjamini_hochberg(out["permutation_p_one_sided"])
    return out


def run_profile_validation(
    subject_features_csv: str | Path,
    output_dir: str | Path,
    blocks: FeatureBlocks,
    assignment_paths: dict[str, str | Path],
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict[str, Path]:
    subject_features_csv = Path(subject_features_csv)
    output_dir = Path(output_dir)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(subject_features_csv)
    assignments = {name: pd.read_csv(path) for name, path in assignment_paths.items()}

    barratt_all, enrichment_all = [], []
    for name, assign in assignments.items():
        barratt, enrichment = validate_one_profile_set(features, assign, blocks, name)
        barratt_all.append(barratt)
        enrichment_all.append(enrichment)
    barratt_by_profile = pd.concat(barratt_all, ignore_index=True) if barratt_all else pd.DataFrame()
    high_low = pd.concat(enrichment_all, ignore_index=True) if enrichment_all else pd.DataFrame()

    comparisons = compare_profile_sets_vs_metadata(
        features=features,
        assignments=assignments,
        blocks=blocks,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    paths = {
        "barratt_by_profile": validation_dir / "barratt_by_profile.csv",
        "high_low_enrichment": validation_dir / "high_low_enrichment_by_profile.csv",
        "profile_set_comparisons": validation_dir / "profile_set_comparisons_vs_metadata.csv",
        "incremental_validity": validation_dir / "incremental_validity_metadata_vs_multimodal.csv",
        "manifest": validation_dir / "profile_validation_manifest.json",
    }
    barratt_by_profile.to_csv(paths["barratt_by_profile"], index=False)
    high_low.to_csv(paths["high_low_enrichment"], index=False)
    comparisons.to_csv(paths["profile_set_comparisons"], index=False)

    # Backward-compatible file. Prefer multimodal_balanced when available.
    if not comparisons.empty:
        preferred = "multimodal_balanced_vs_metadata" if (comparisons["comparison"] == "multimodal_balanced_vs_metadata").any() else "multimodal_vs_metadata"
        inc = comparisons[comparisons["comparison"] == preferred].copy()
        if not inc.empty:
            inc = inc.rename(columns={
                "baseline_profile_adj_r2": "metadata_profile_adj_r2",
                "comparison_profile_adj_r2": "multimodal_profile_adj_r2",
                "baseline_delta_adj_r2_over_covariates": "metadata_delta_adj_r2_over_covariates",
                "comparison_delta_adj_r2_over_covariates": "multimodal_delta_adj_r2_over_covariates",
                "delta_adj_r2_comparison_minus_baseline": "delta_adj_r2_multimodal_minus_metadata",
            })
        else:
            inc = pd.DataFrame()
    else:
        inc = pd.DataFrame()
    inc.to_csv(paths["incremental_validity"], index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject_features_csv": str(subject_features_csv),
        "assignment_paths": {k: str(v) for k, v in assignment_paths.items()},
        "targets": blocks.targets,
        "label_col": blocks.label_col,
        "covariates": blocks.covariates,
        "n_permutations": n_permutations,
        "comparison_file": str(paths["profile_set_comparisons"]),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profile validation saved: {validation_dir}")
    return paths
