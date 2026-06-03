from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm
except Exception as exc:
    raise RuntimeError("scipy is required for rank-normal transformation") from exc


def split_features(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def rank_normal(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    n = int(mask.sum())
    if n < 3:
        return out
    ranks = s[mask].rank(method="average")
    out.loc[mask] = norm.ppf((ranks - 0.5) / n)
    return out


def make_design(df: pd.DataFrame, covariates: list[str], label_col: str, positive_label: str, include_group: bool = True) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["intercept"] = 1.0
    if include_group:
        X["group_positive"] = (df[label_col] == positive_label).astype(float)

    for cov in covariates:
        if cov not in df.columns:
            continue
        col = df[cov]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().sum() >= max(10, len(df) * 0.6):
            X[cov] = num.fillna(num.mean()).astype(float)
        else:
            dummies = pd.get_dummies(col.fillna("__MISSING__").astype(str), prefix=cov, drop_first=True)
            X = pd.concat([X, dummies.astype(float)], axis=1)
    return X


def ols_group_beta(df: pd.DataFrame, y_col: str, covariates: list[str], label_col: str, positive_label: str, negative_label: str) -> float:
    d = df[[label_col, y_col] + [c for c in covariates if c in df.columns]].copy()
    d = d[d[label_col].isin([positive_label, negative_label])]
    d[y_col] = rank_normal(d[y_col])
    d = d.dropna(subset=[y_col])
    if len(d) < 20:
        return np.nan
    X = make_design(d, covariates, label_col, positive_label, include_group=True)
    y = d[y_col].astype(float).to_numpy()
    beta = np.linalg.lstsq(X.to_numpy(dtype=float), y, rcond=None)[0]
    return float(beta[list(X.columns).index("group_positive")])


def save_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, filename: Path, colorbar_label: str) -> None:
    vmax = np.nanmax(np.abs(matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(9, max(7, len(row_labels) * 0.32)))
    im = plt.imshow(matrix, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=35, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation figures for 05_run composite SpeechGraph indices.")
    parser.add_argument("--run-dir", default="outputs/05_run")
    parser.add_argument("--label-col", default="Tipo")
    parser.add_argument("--positive-label", default="high_imp")
    parser.add_argument("--negative-label", default="low_imp")
    parser.add_argument("--covariates", default="School year,Gender")
    args = parser.parse_args()

    base = Path(args.run_dir)
    figdir = base / "figures" / "presentation"
    figdir.mkdir(parents=True, exist_ok=True)
    covariates = [x.strip() for x in args.covariates.split(",") if x.strip()]

    idx_path = base / "tables" / "composite_indices_subject_level.csv"
    wide_path = base / "tables" / "subject_level_speechgraph_wide_for_indices.csv"
    comp_path = base / "tables" / "composite_index_components.csv"
    res_path = base / "tables" / "composite_index_contrast_results.csv"

    for p in [idx_path, wide_path, comp_path, res_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    idx = pd.read_csv(idx_path)
    wide = pd.read_csv(wide_path)
    comp = pd.read_csv(comp_path)
    res = pd.read_csv(res_path)

    primary = res[(res["model"] == "primary_school_year_gender") & (res["primary_family_primary_model"] == True)].copy()
    index_order = primary.sort_values("adj_beta_high", ascending=False)["index"].tolist()

    # 1) Adjusted group means
    rows = []
    for index_name in index_order:
        d = idx[[args.label_col, index_name] + [c for c in covariates if c in idx.columns]].copy()
        d = d[d[args.label_col].isin([args.positive_label, args.negative_label])].dropna(subset=[index_name])
        X_cov = make_design(d, covariates, args.label_col, args.positive_label, include_group=False)
        y = d[index_name].astype(float).to_numpy()
        beta = np.linalg.lstsq(X_cov.to_numpy(dtype=float), y, rcond=None)[0]
        resid = y - X_cov.to_numpy(dtype=float).dot(beta)
        d["adjusted_index_value"] = resid + np.nanmean(y)
        for label in [args.positive_label, args.negative_label]:
            vals = d.loc[d[args.label_col] == label, "adjusted_index_value"].dropna().to_numpy()
            rows.append({
                "index": index_name,
                "label": label,
                "mean": float(vals.mean()),
                "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                "n": len(vals),
            })
    adj_means = pd.DataFrame(rows)
    adj_means.to_csv(figdir / "adjusted_group_means_by_index.csv", index=False)

    plt.figure(figsize=(9, 5))
    x = np.arange(len(index_order))
    offsets = {args.positive_label: -0.12, args.negative_label: 0.12}
    for label in [args.positive_label, args.negative_label]:
        sub = adj_means[adj_means["label"] == label].set_index("index").loc[index_order].reset_index()
        plt.errorbar(x + offsets[label], sub["mean"], yerr=1.96 * sub["se"], fmt="o", capsize=4, label=label)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, index_order, rotation=25, ha="right")
    plt.ylabel("Adjusted residualized index mean")
    plt.title("Adjusted group means by composite SpeechGraph index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "adjusted_group_means_by_index.png", dpi=220)
    plt.close()

    # 2) Component-level adjusted effects
    component_records = []
    for _, row in comp.iterrows():
        index_name = row["index"]
        for feat in split_features(row.get("positive_features")):
            component_records.append({"index": index_name, "feature": feat, "weight": 1.0})
        for feat in split_features(row.get("negative_features")):
            component_records.append({"index": index_name, "feature": feat, "weight": -1.0})

    component_df = pd.DataFrame(component_records)
    effects = []
    for feat in sorted(component_df["feature"].unique()):
        beta = ols_group_beta(wide, feat, covariates, args.label_col, args.positive_label, args.negative_label) if feat in wide.columns else np.nan
        effects.append({"feature": feat, "raw_adjusted_beta_high_minus_low": beta})
    effects = pd.DataFrame(effects)

    component_df = component_df.merge(effects, on="feature", how="left")
    component_df["oriented_adjusted_beta"] = component_df["weight"] * component_df["raw_adjusted_beta_high_minus_low"]
    component_df.to_csv(figdir / "component_level_adjusted_effects.csv", index=False)

    features = component_df["feature"].drop_duplicates().tolist()
    indices = comp["index"].tolist()
    raw_mat = np.full((len(features), len(indices)), np.nan)
    oriented_mat = np.full((len(features), len(indices)), np.nan)

    for i, feat in enumerate(features):
        for j, ind in enumerate(indices):
            sub = component_df[(component_df["feature"] == feat) & (component_df["index"] == ind)]
            if len(sub):
                raw_mat[i, j] = sub["raw_adjusted_beta_high_minus_low"].iloc[0]
                oriented_mat[i, j] = sub["oriented_adjusted_beta"].iloc[0]

    save_heatmap(raw_mat, features, indices, "Component-level adjusted high-minus-low effects", figdir / "component_level_raw_effects_by_index.png", "Adjusted beta")
    save_heatmap(oriented_mat, features, indices, "Component-level effects oriented by index definition", figdir / "component_level_oriented_effects_by_index.png", "Oriented adjusted beta")

    # 3) Correlation among indices
    corr = idx[index_order].corr()
    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(corr.to_numpy(), vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson correlation")
    plt.xticks(np.arange(len(index_order)), index_order, rotation=35, ha="right")
    plt.yticks(np.arange(len(index_order)), index_order)
    plt.title("Correlation among composite SpeechGraph indices")
    for i in range(len(index_order)):
        for j in range(len(index_order)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figdir / "composite_index_correlation_matrix.png", dpi=220)
    plt.close()

    print("Saved presentation figures to:", figdir)
    for p in sorted(figdir.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
