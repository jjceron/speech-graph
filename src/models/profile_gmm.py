from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analysis.profile_preprocessing import FeatureBlocks, numeric_and_categorical, safe_numeric_frame


@dataclass
class NumericBlockTransformer:
    columns: list[str]
    use_pca: bool = False
    variance_threshold: float = 0.80
    max_components: int = 10
    prefix: str = "PC"

    def fit(self, df: pd.DataFrame):
        self.columns_ = [c for c in self.columns if c in df.columns]
        self.medians_ = None
        self.scaler_ = None
        self.pca_ = None
        self.feature_names_ = []
        if not self.columns_:
            return self
        x = safe_numeric_frame(df, self.columns_)
        self.medians_ = x.median(numeric_only=True).fillna(0.0)
        x = x.fillna(self.medians_)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        if self.use_pca and xs.shape[1] > 1 and xs.shape[0] > 2:
            full = PCA(random_state=0)
            full.fit(xs)
            cumulative = np.cumsum(full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
            n_comp = max(1, min(n_comp, self.max_components, xs.shape[1], xs.shape[0] - 1))
            self.pca_ = PCA(n_components=n_comp, random_state=0)
            self.pca_.fit(xs)
            self.feature_names_ = [f"{self.prefix}{i+1}" for i in range(n_comp)]
        else:
            self.feature_names_ = list(self.columns_)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "columns_", None):
            return np.empty((len(df), 0), dtype=float)
        x = safe_numeric_frame(df, self.columns_).fillna(self.medians_)
        xs = self.scaler_.transform(x)
        if self.pca_ is not None:
            return self.pca_.transform(xs)
        return xs

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def manifest(self) -> dict:
        return {
            "columns": getattr(self, "columns_", []),
            "use_pca": self.use_pca,
            "n_output_features": len(getattr(self, "feature_names_", [])),
            "feature_names": getattr(self, "feature_names_", []),
            "explained_variance_ratio": self.pca_.explained_variance_ratio_.tolist() if getattr(self, "pca_", None) is not None else None,
        }


@dataclass
class CategoricalBlockTransformer:
    columns: list[str]
    prefix: str = "cat"

    def fit(self, df: pd.DataFrame):
        self.columns_ = [c for c in self.columns if c in df.columns]
        self.modes_ = {}
        self.categories_ = {}
        self.feature_names_ = []
        for col in self.columns_:
            series = df[col].astype("object")
            mode = series.dropna().mode()
            fill = mode.iloc[0] if len(mode) else "missing"
            self.modes_[col] = fill
            values = series.fillna(fill).astype(str)
            cats = sorted(values.unique().tolist())
            self.categories_[col] = cats
            self.feature_names_.extend([f"{col}={cat}" for cat in cats])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "columns_", None):
            return np.empty((len(df), 0), dtype=float)
        matrices = []
        for col in self.columns_:
            fill = self.modes_[col]
            cats = self.categories_[col]
            values = df[col].astype("object").fillna(fill).astype(str)
            mat = np.zeros((len(df), len(cats)), dtype=float)
            cat_to_idx = {cat: i for i, cat in enumerate(cats)}
            for i, val in enumerate(values):
                j = cat_to_idx.get(val)
                if j is not None:
                    mat[i, j] = 1.0
            matrices.append(mat)
        return np.hstack(matrices) if matrices else np.empty((len(df), 0), dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def manifest(self) -> dict:
        return {
            "columns": getattr(self, "columns_", []),
            "n_output_features": len(getattr(self, "feature_names_", [])),
            "feature_names": getattr(self, "feature_names_", []),
            "categories": getattr(self, "categories_", {}),
        }


class ProfileBlockEncoder:
    """Block-aware encoder for profile discovery.

    Supported profile sets:
    - metadata: demographic + cognitive/language metadata.
    - speechgraph: SpeechGraph components only.
    - multimodal: metadata + SpeechGraph, globally standardized.
    - multimodal_balanced: metadata + SpeechGraph with equal block weights.

    The v2 encoder applies final standardization to encoded dimensions. This avoids
    the GMM being driven by arbitrary scale differences between PCA scores, raw
    numeric variables, and one-hot indicators.
    """

    VALID_PROFILE_SETS = {"metadata", "speechgraph", "multimodal", "multimodal_balanced"}

    def __init__(
        self,
        blocks: FeatureBlocks,
        profile_set: str,
        variance_threshold: float = 0.80,
        max_cognitive_components: int = 6,
        max_speechgraph_components: int = 12,
        final_standardize: bool = True,
    ):
        if profile_set not in self.VALID_PROFILE_SETS:
            raise ValueError(f"profile_set must be one of {sorted(self.VALID_PROFILE_SETS)}")
        self.blocks = blocks
        self.profile_set = profile_set
        self.variance_threshold = variance_threshold
        self.max_cognitive_components = max_cognitive_components
        self.max_speechgraph_components = max_speechgraph_components
        self.final_standardize = final_standardize

    def fit(self, df: pd.DataFrame):
        demographic_numeric, demographic_categorical = numeric_and_categorical(df, self.blocks.demographics)
        cognitive_numeric, cognitive_categorical = numeric_and_categorical(df, self.blocks.cognitive)
        speechgraph_numeric = [c for c in self.blocks.speechgraph if c in df.columns]

        self.demographic_numeric_ = NumericBlockTransformer(demographic_numeric, use_pca=False, prefix="demographic")
        self.demographic_categorical_ = CategoricalBlockTransformer(demographic_categorical)
        self.cognitive_numeric_ = NumericBlockTransformer(
            cognitive_numeric,
            use_pca=True,
            variance_threshold=self.variance_threshold,
            max_components=self.max_cognitive_components,
            prefix="cognitive_PC",
        )
        self.cognitive_categorical_ = CategoricalBlockTransformer(cognitive_categorical)
        self.speechgraph_numeric_ = NumericBlockTransformer(
            speechgraph_numeric,
            use_pca=True,
            variance_threshold=self.variance_threshold,
            max_components=self.max_speechgraph_components,
            prefix="speechgraph_PC",
        )

        self.demographic_numeric_.fit(df)
        self.demographic_categorical_.fit(df)
        self.cognitive_numeric_.fit(df)
        self.cognitive_categorical_.fit(df)
        self.speechgraph_numeric_.fit(df)
        self.feature_names_ = self._feature_names()

        # Fit final scaler after constructing the selected encoded matrix.
        raw = self._raw_transform(df)
        self.final_scaler_ = None
        if self.final_standardize and raw.shape[1] > 0:
            self.final_scaler_ = StandardScaler()
            self.final_scaler_.fit(raw)
        return self

    def _metadata_feature_names(self) -> list[str]:
        names = []
        names.extend(self.demographic_numeric_.feature_names_)
        names.extend(self.demographic_categorical_.feature_names_)
        names.extend(self.cognitive_numeric_.feature_names_)
        names.extend(self.cognitive_categorical_.feature_names_)
        return names

    def _feature_names(self) -> list[str]:
        meta_names = self._metadata_feature_names()
        sg_names = self.speechgraph_numeric_.feature_names_
        if self.profile_set == "metadata":
            return meta_names
        if self.profile_set == "speechgraph":
            return sg_names
        return [*meta_names, *sg_names]

    @staticmethod
    def _standardize_block(mat: np.ndarray) -> np.ndarray:
        if mat.shape[1] == 0:
            return mat
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        z = (mat - mean) / sd
        z[~np.isfinite(z)] = 0.0
        return z

    @classmethod
    def _balance_blocks(cls, blocks: list[np.ndarray]) -> list[np.ndarray]:
        non_empty = [b for b in blocks if b.shape[1] > 0]
        if not non_empty:
            return blocks
        balanced = []
        for b in blocks:
            if b.shape[1] == 0:
                balanced.append(b)
            else:
                # Equalize total block contribution: after column-standardization,
                # divide by sqrt(number of dimensions in the block).
                balanced.append(cls._standardize_block(b) / np.sqrt(b.shape[1]))
        return balanced

    def _block_matrices(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        demographic = np.hstack([
            self.demographic_numeric_.transform(df),
            self.demographic_categorical_.transform(df),
        ])
        cognitive = np.hstack([
            self.cognitive_numeric_.transform(df),
            self.cognitive_categorical_.transform(df),
        ])
        speechgraph = self.speechgraph_numeric_.transform(df)
        metadata = np.hstack([m for m in [demographic, cognitive] if m.shape[1] > 0])
        return {
            "demographic": demographic,
            "cognitive": cognitive,
            "metadata": metadata,
            "speechgraph": speechgraph,
        }

    def _raw_transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = self._block_matrices(df)
        if self.profile_set == "metadata":
            mats = [blocks["metadata"]]
        elif self.profile_set == "speechgraph":
            mats = [blocks["speechgraph"]]
        elif self.profile_set == "multimodal":
            mats = [blocks["metadata"], blocks["speechgraph"]]
        elif self.profile_set == "multimodal_balanced":
            mats = self._balance_blocks([blocks["metadata"], blocks["speechgraph"]])
        else:
            raise ValueError(f"Unknown profile_set: {self.profile_set}")
        out = np.hstack([m for m in mats if m.shape[1] > 0]) if mats else np.empty((len(df), 0), dtype=float)
        if out.shape[1] == 0:
            raise ValueError("No usable profile features after preprocessing.")
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = self._raw_transform(df)
        if getattr(self, "final_scaler_", None) is not None:
            out = self.final_scaler_.transform(out)
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def manifest(self) -> dict:
        return {
            "profile_set": self.profile_set,
            "variance_threshold": self.variance_threshold,
            "final_standardize": self.final_standardize,
            "feature_names": self.feature_names_,
            "demographic_numeric": self.demographic_numeric_.manifest(),
            "demographic_categorical": self.demographic_categorical_.manifest(),
            "cognitive_numeric": self.cognitive_numeric_.manifest(),
            "cognitive_categorical": self.cognitive_categorical_.manifest(),
            "speechgraph_numeric": self.speechgraph_numeric_.manifest(),
        }


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(counts.index)}
    return np.array([mapping[x] for x in labels], dtype=int)


def _fit_one_gmm(x: np.ndarray, k: int, random_state: int, covariance_type: str = "diag") -> GaussianMixture:
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=10,
        reg_covar=1e-5,
        random_state=random_state,
    )
    model.fit(x)
    return model


def _model_selection_table(x: np.ndarray, k_values: list[int], random_state: int, min_profile_size: int, covariance_type: str) -> pd.DataFrame:
    rows = []
    for k in k_values:
        if k >= len(x):
            continue
        try:
            model = _fit_one_gmm(x, k, random_state=random_state + k, covariance_type=covariance_type)
            labels = _relabel_by_size(model.predict(x))
            counts = pd.Series(labels).value_counts().sort_index()
            sil = np.nan
            if len(np.unique(labels)) > 1 and min(counts) > 1:
                sil = float(silhouette_score(x, labels))
            rows.append({
                "k": int(k),
                "bic": float(model.bic(x)),
                "aic": float(model.aic(x)),
                "silhouette": sil,
                "min_profile_size": int(counts.min()),
                "max_profile_size": int(counts.max()),
                "valid_min_size": bool(counts.min() >= min_profile_size),
                "converged": bool(model.converged_),
            })
        except Exception as exc:
            rows.append({
                "k": int(k), "bic": np.nan, "aic": np.nan, "silhouette": np.nan,
                "min_profile_size": np.nan, "max_profile_size": np.nan,
                "valid_min_size": False, "converged": False, "error": str(exc),
            })
    return pd.DataFrame(rows)


def choose_k(selection: pd.DataFrame) -> int:
    valid = selection[selection["valid_min_size"].fillna(False) & selection["bic"].notna()].copy()
    if valid.empty:
        valid = selection[selection["bic"].notna()].copy()
    if valid.empty:
        raise ValueError("No valid GMM solution could be fitted.")
    return int(valid.sort_values("bic", ascending=True).iloc[0]["k"])


def bootstrap_stability(
    df: pd.DataFrame,
    blocks: FeatureBlocks,
    profile_set: str,
    reference_labels: np.ndarray,
    k: int,
    n_bootstrap: int,
    random_state: int,
    variance_threshold: float,
    max_cognitive_components: int,
    max_speechgraph_components: int,
    covariance_type: str,
    final_standardize: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    n = len(df)
    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx].reset_index(drop=True)
        try:
            enc = ProfileBlockEncoder(
                blocks,
                profile_set=profile_set,
                variance_threshold=variance_threshold,
                max_cognitive_components=max_cognitive_components,
                max_speechgraph_components=max_speechgraph_components,
                final_standardize=final_standardize,
            )
            x_sample = enc.fit_transform(sample)
            gmm = _fit_one_gmm(x_sample, k, random_state=random_state + 1000 + b, covariance_type=covariance_type)
            x_full = enc.transform(df)
            labels = _relabel_by_size(gmm.predict(x_full))
            ari = float(adjusted_rand_score(reference_labels, labels))
            rows.append({"bootstrap": b, "ari_vs_reference": ari, "status": "ok"})
        except Exception as exc:
            rows.append({"bootstrap": b, "ari_vs_reference": np.nan, "status": "failed", "error": str(exc)})
    return pd.DataFrame(rows)


def feature_contrasts(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    blocks: FeatureBlocks,
    profile_col: str = "profile",
    top_n_per_profile: int = 25,
) -> pd.DataFrame:
    feature_cols = blocks.multimodal_features
    available = [c for c in feature_cols if c in df.columns]
    numeric = [c for c in available if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
    if not numeric:
        return pd.DataFrame()
    merged = df[[blocks.code_col, *numeric]].merge(assignments[[blocks.code_col, profile_col]], on=blocks.code_col, how="inner")
    x = merged[numeric].apply(pd.to_numeric, errors="coerce")
    overall_mean = x.mean(skipna=True)
    overall_sd = x.std(skipna=True).replace(0, np.nan)
    rows = []
    for profile, sub in merged.groupby(profile_col, dropna=False):
        sx = sub[numeric].apply(pd.to_numeric, errors="coerce")
        means = sx.mean(skipna=True)
        z = (means - overall_mean) / overall_sd
        for feature, value in z.items():
            rows.append({
                "profile": profile,
                "feature": feature,
                "profile_mean": float(means[feature]) if pd.notna(means[feature]) else np.nan,
                "overall_mean": float(overall_mean[feature]) if pd.notna(overall_mean[feature]) else np.nan,
                "z_contrast": float(value) if pd.notna(value) else np.nan,
                "abs_z_contrast": float(abs(value)) if pd.notna(value) else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["profile", "abs_z_contrast"], ascending=[True, False])
    return out.groupby("profile", group_keys=False).head(top_n_per_profile).reset_index(drop=True)


def run_profile_gmm(
    subject_features_csv: str | Path,
    output_dir: str | Path,
    blocks: FeatureBlocks,
    profile_set: str,
    k_values: list[int] | None = None,
    min_profile_size: int = 20,
    n_bootstrap: int = 200,
    random_state: int = 42,
    variance_threshold: float = 0.80,
    max_cognitive_components: int = 6,
    max_speechgraph_components: int = 12,
    covariance_type: str = "diag",
    final_standardize: bool = True,
) -> dict[str, Path]:
    subject_features_csv = Path(subject_features_csv)
    output_dir = Path(output_dir)
    profile_dir = output_dir / "profiles"
    interp_dir = output_dir / "interpretation"
    profile_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(subject_features_csv)
    if blocks.code_col not in df.columns:
        raise ValueError(f"Code column '{blocks.code_col}' not found in subject features.")
    k_values = k_values or [2, 3, 4, 5]

    encoder = ProfileBlockEncoder(
        blocks,
        profile_set=profile_set,
        variance_threshold=variance_threshold,
        max_cognitive_components=max_cognitive_components,
        max_speechgraph_components=max_speechgraph_components,
        final_standardize=final_standardize,
    )
    x = encoder.fit_transform(df)
    reduced = pd.DataFrame(x, columns=encoder.feature_names_)
    reduced.insert(0, blocks.code_col, df[blocks.code_col].astype(str).values)

    selection = _model_selection_table(x, k_values, random_state, min_profile_size, covariance_type)
    selection.insert(0, "profile_set", profile_set)
    final_k = choose_k(selection)
    final_gmm = _fit_one_gmm(x, final_k, random_state=random_state + 777, covariance_type=covariance_type)
    old_labels = final_gmm.predict(x)
    labels = _relabel_by_size(old_labels)
    probs_raw = final_gmm.predict_proba(x)
    counts = pd.Series(old_labels).value_counts().sort_values(ascending=False)
    order = list(counts.index)
    probs = probs_raw[:, order]

    assignments = pd.DataFrame({
        blocks.code_col: df[blocks.code_col].astype(str).values,
        "profile_set": profile_set,
        "profile_k": final_k,
        "profile": [f"P{int(v)+1}" for v in labels],
        "profile_numeric": labels + 1,
        "profile_probability_max": probs.max(axis=1),
    })
    for i in range(final_k):
        assignments[f"prob_P{i+1}"] = probs[:, i]

    if n_bootstrap > 0:
        stability = bootstrap_stability(
            df=df,
            blocks=blocks,
            profile_set=profile_set,
            reference_labels=labels,
            k=final_k,
            n_bootstrap=n_bootstrap,
            random_state=random_state + 2000,
            variance_threshold=variance_threshold,
            max_cognitive_components=max_cognitive_components,
            max_speechgraph_components=max_speechgraph_components,
            covariance_type=covariance_type,
            final_standardize=final_standardize,
        )
    else:
        stability = pd.DataFrame(columns=["bootstrap", "ari_vs_reference", "status"])
    stability_summary = pd.DataFrame([{
        "profile_set": profile_set,
        "k": final_k,
        "n_bootstrap": int(n_bootstrap),
        "mean_ari": float(stability["ari_vs_reference"].mean(skipna=True)) if stability["ari_vs_reference"].notna().any() else np.nan,
        "median_ari": float(stability["ari_vs_reference"].median(skipna=True)) if stability["ari_vs_reference"].notna().any() else np.nan,
        "std_ari": float(stability["ari_vs_reference"].std(skipna=True)) if stability["ari_vs_reference"].notna().any() else np.nan,
        "p025_ari": float(np.nanpercentile(stability["ari_vs_reference"], 2.5)) if stability["ari_vs_reference"].notna().any() else np.nan,
        "p975_ari": float(np.nanpercentile(stability["ari_vs_reference"], 97.5)) if stability["ari_vs_reference"].notna().any() else np.nan,
        "failed_bootstraps": int((stability["status"] != "ok").sum()) if "status" in stability else 0,
    }])

    contrasts = feature_contrasts(df, assignments, blocks, profile_col="profile")

    paths = {
        "assignments": profile_dir / f"profile_assignments_{profile_set}.csv",
        "reduced_matrix": profile_dir / f"profile_reduced_matrix_{profile_set}.csv",
        "selection": profile_dir / f"profile_model_selection_{profile_set}.csv",
        "stability": profile_dir / f"profile_stability_{profile_set}.csv",
        "stability_summary": profile_dir / f"profile_stability_summary_{profile_set}.csv",
        "encoder_manifest": profile_dir / f"profile_encoder_manifest_{profile_set}.json",
        "feature_contrasts": interp_dir / f"profile_feature_contrasts_{profile_set}.csv",
    }
    assignments.to_csv(paths["assignments"], index=False)
    reduced.to_csv(paths["reduced_matrix"], index=False)
    selection.to_csv(paths["selection"], index=False)
    stability.to_csv(paths["stability"], index=False)
    stability_summary.to_csv(paths["stability_summary"], index=False)
    contrasts.to_csv(paths["feature_contrasts"], index=False)
    paths["encoder_manifest"].write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "encoder": encoder.manifest()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{profile_set} profiles saved: {paths['assignments']} (k={final_k})")
    return paths
