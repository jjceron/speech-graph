from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _safe(text: object) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, fmt: str = ".4f") -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if not np.isfinite(width):
            continue
        y = patch.get_y() + patch.get_height() / 2
        ha = "left" if width >= 0 else "right"
        dx = 0.002 if width >= 0 else -0.002
        ax.text(width + dx, y, format(width, fmt), va="center", ha=ha, fontsize=8)


def plot_model_selection(run_dir: Path) -> int:
    count = 0
    files = list((run_dir / "profiles").glob("profile_model_selection_*.csv"))
    rows = []
    for p in files:
        df = _read(p)
        if not df.empty:
            rows.append(df)
    if not rows:
        return 0
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for profile_set, sub in data.groupby("profile_set"):
        ax.plot(sub["k"], sub["bic"], marker="o", label=str(profile_set))
    ax.set_xlabel("Number of profiles (k)")
    ax.set_ylabel("BIC")
    ax.set_title("GMM profile model selection")
    ax.legend()
    _save(fig, run_dir / "figures" / "profiles" / "profile_model_selection_bic.png")
    return count + 1


def plot_profile_maps(run_dir: Path) -> int:
    count = 0
    for reduced_path in (run_dir / "profiles").glob("profile_reduced_matrix_*.csv"):
        profile_set = reduced_path.stem.replace("profile_reduced_matrix_", "")
        assign_path = run_dir / "profiles" / f"profile_assignments_{profile_set}.csv"
        reduced = _read(reduced_path)
        assign = _read(assign_path)
        if reduced.empty or assign.empty:
            continue
        code_col = reduced.columns[0]
        data = reduced.merge(assign[[code_col, "profile"]], on=code_col, how="inner")
        components = [c for c in reduced.columns if c != code_col]
        if len(components) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        profiles = sorted(data["profile"].dropna().unique())
        for profile in profiles:
            sub = data[data["profile"] == profile]
            ax.scatter(sub[components[0]], sub[components[1]], label=str(profile), alpha=0.75)
        ax.set_xlabel(components[0])
        ax.set_ylabel(components[1])
        ax.set_title(f"Profile map: {profile_set}")
        ax.legend(title="Profile")
        _save(fig, run_dir / "figures" / "profiles" / f"profile_map_{_safe(profile_set)}.png")
        count += 1
    return count


def plot_barratt_by_profile(run_dir: Path, subject_features_csv: Path, targets: list[str]) -> int:
    features = _read(subject_features_csv)
    if features.empty:
        return 0
    count = 0
    for assign_path in (run_dir / "profiles").glob("profile_assignments_*.csv"):
        profile_set = assign_path.stem.replace("profile_assignments_", "")
        assign = _read(assign_path)
        if assign.empty:
            continue
        code_col = assign.columns[0]
        data = features.merge(assign[[code_col, "profile"]], on=code_col, how="inner")
        existing = [t for t in targets if t in data.columns]
        if not existing:
            continue
        for target in existing:
            groups = []
            labels = []
            for profile, sub in data.groupby("profile"):
                vals = pd.to_numeric(sub[target], errors="coerce").dropna().to_numpy()
                if len(vals):
                    groups.append(vals)
                    labels.append(str(profile))
            if len(groups) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(groups, labels=labels, showmeans=True)
            ax.set_xlabel("Profile")
            ax.set_ylabel(target)
            ax.set_title(f"{target} by profile: {profile_set}")
            _save(fig, run_dir / "figures" / "validation" / f"barratt_{_safe(target)}_by_profile_{_safe(profile_set)}.png")
            count += 1
    return count


def plot_high_low_enrichment(run_dir: Path) -> int:
    data = _read(run_dir / "validation" / "high_low_enrichment_by_profile.csv")
    if data.empty:
        return 0
    count = 0
    for profile_set, sub in data.groupby("profile_set"):
        piv = sub.pivot_table(index="profile", columns="label", values="profile_proportion", aggfunc="mean").fillna(0)
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        bottom = np.zeros(len(piv))
        x = np.arange(len(piv))
        for label in piv.columns:
            vals = piv[label].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=str(label))
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index.astype(str))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Within-profile proportion")
        ax.set_xlabel("Profile")
        ax.set_title(f"High/low enrichment by profile: {profile_set}")
        ax.legend(title="Label")
        _save(fig, run_dir / "figures" / "validation" / f"high_low_enrichment_{_safe(profile_set)}.png")
        count += 1
    return count


def plot_incremental_validity(run_dir: Path) -> int:
    data = _read(run_dir / "validation" / "incremental_validity_metadata_vs_multimodal.csv")
    if data.empty or "delta_adj_r2_multimodal_minus_metadata" not in data.columns:
        return 0
    data = data.sort_values("delta_adj_r2_multimodal_minus_metadata", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(data["target"].astype(str), data["delta_adj_r2_multimodal_minus_metadata"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Δ adjusted R²: multimodal profiles − metadata profiles")
    ax.set_ylabel("Barratt dimension")
    ax.set_title("Incremental validity of SpeechGraph-enhanced profiles")
    _annotate_bars(ax, fmt=".4f")
    _save(fig, run_dir / "figures" / "validation" / "metadata_vs_multimodal_delta_adj_r2.png")
    return 1


def plot_feature_heatmaps(run_dir: Path, top_n: int = 30) -> int:
    count = 0
    for path in (run_dir / "interpretation").glob("profile_feature_contrasts_*.csv"):
        profile_set = path.stem.replace("profile_feature_contrasts_", "")
        data = _read(path)
        if data.empty or "z_contrast" not in data.columns:
            continue
        top_features = (
            data.groupby("feature")["abs_z_contrast"].max().sort_values(ascending=False).head(top_n).index.tolist()
        )
        table = data[data["feature"].isin(top_features)].pivot_table(index="feature", columns="profile", values="z_contrast", aggfunc="mean")
        if table.empty:
            continue
        table = table.loc[top_features]
        values = table.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(table.columns)), max(6, 0.24 * len(table.index))))
        im = ax.imshow(values, aspect="auto")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns.astype(str))
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index.astype(str), fontsize=7)
        ax.set_xlabel("Profile")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top profile feature contrasts: {profile_set}")
        fig.colorbar(im, ax=ax, label="Profile z contrast")
        _save(fig, run_dir / "figures" / "interpretation" / f"profile_feature_heatmap_{_safe(profile_set)}.png")
        count += 1
    return count


def generate_profile_figures(run_dir: str | Path, subject_features_csv: str | Path | None = None, targets_text: str = "TOTAL,NPLAN,MOT,COG") -> int:
    run_dir = Path(run_dir)
    if subject_features_csv is None:
        subject_features_csv = run_dir / "features" / "subject_level_multimodal_features.csv"
    subject_features_csv = Path(subject_features_csv)
    targets = [x.strip() for x in targets_text.split(",") if x.strip()]
    count = 0
    count += plot_model_selection(run_dir)
    count += plot_profile_maps(run_dir)
    count += plot_barratt_by_profile(run_dir, subject_features_csv, targets)
    count += plot_high_low_enrichment(run_dir)
    count += plot_incremental_validity(run_dir)
    count += plot_feature_heatmaps(run_dir)
    print(f"Generated {count} profile figures in {run_dir / 'figures'}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for 03_run profile analysis.")
    parser.add_argument("--run-dir", default="outputs/03_run")
    parser.add_argument("--subject-features-csv", default=None)
    parser.add_argument("--targets", default="TOTAL,NPLAN,MOT,COG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_profile_figures(args.run_dir, args.subject_features_csv, args.targets)


if __name__ == "__main__":
    main()
