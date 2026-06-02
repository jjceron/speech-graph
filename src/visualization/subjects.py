from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    available_core_metrics,
    append_manifest,
    ensure_figures_dir,
    filter_analysis_level,
    pick_window,
    read_data_csv,
    safe_name,
)
from .plots_utils import save_horizontal_bars, save_scatter
from scipy import stats

from .stats_utils import zscore


def add_subject_profile_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_cols = [c for c in ["mean_z_lcc", "mean_z_lsc", "mean_z_edges"] if c in out.columns]
    if z_cols:
        z_frame = pd.DataFrame({c: pd.to_numeric(out[c], errors="coerce") for c in z_cols})
        out["connectivity_z_profile"] = z_frame.mean(axis=1)
        out["disorganization_proxy"] = -out["connectivity_z_profile"]
    else:
        # Fallback for no-random runs: larger LSC ratio/ASP/nodes and lower density usually mean more organized discourse.
        parts: list[pd.Series] = []
        for col, sign in [
            ("mean_lsc_ratio", 1),
            ("mean_lsc", 1),
            ("mean_nodes", 1),
            ("mean_asp", 1),
            ("mean_density", -1),
            ("mean_repeated_edges_ratio", -1),
        ]:
            if col in out.columns:
                parts.append(sign * zscore(out[col]))
        if parts:
            out["connectivity_z_profile"] = pd.concat(parts, axis=1).mean(axis=1)
            out["disorganization_proxy"] = -out["connectivity_z_profile"]
        else:
            out["connectivity_z_profile"] = pd.NA
            out["disorganization_proxy"] = pd.NA
    return out


def generate_subject_figures(
    run_dir: Path,
    window_size: int = 30,
    level: str = "file",
    top_n: int = 30,
    group_col: str = "Tipo",
    min_n: int = 20,
    min_abs_r: float = 0.20,
) -> list[dict]:
    df, data_path = read_data_csv(run_dir)
    if df is None or df.empty:
        print(f"[{run_dir.name}] no data CSV found")
        return []
    df = filter_analysis_level(df, level)
    df_w = pick_window(df, window_size)
    df_w = add_subject_profile_scores(df_w)
    out_dir = ensure_figures_dir(run_dir, "nlp_profile/subjects")
    rows: list[dict] = []

    keep_cols = [
        "code", "file", "window_size", "connectivity_z_profile", "disorganization_proxy",
        "mean_z_lcc", "mean_z_lsc", "mean_z_edges",
        "mean_lsc_ratio", "mean_lsc", "mean_nodes", "mean_density", "mean_asp",
        "TOTAL", "NPLAN", "MOT", "COG", "Barratt (pre)", "Age", "School year", "Tipo",
        "Naming task", "COHERENCIA NARRATIVA", "Reading comprehension task", "Verbal fluency tasks",
    ]
    label_cols = [c for c in df_w.columns if c.startswith("label_ratio_") and not any(x in c.upper() for x in ["STARTTIME", "ENDTIME"])]
    keep_cols = [c for c in keep_cols + label_cols if c in df_w.columns]
    profile = df_w[keep_cols].copy()
    profile_path = out_dir / f"subject_nlp_profile_w{window_size}.csv"
    profile.to_csv(profile_path, index=False)
    rows.append({"table": str(profile_path), "kind": "subject_profile"})

    values = pd.to_numeric(df_w["disorganization_proxy"], errors="coerce")
    rank = df_w.loc[values.notna()].copy()
    rank["disorganization_proxy"] = values[values.notna()]
    rank = rank.sort_values("disorganization_proxy", ascending=False).head(top_n)
    if not rank.empty and "code" in rank.columns:
        labels = []
        for _, row in rank.iterrows():
            extra = f" | {row[group_col]}" if group_col in rank.columns and pd.notna(row[group_col]) else ""
            labels.append(f"{row['code']}{extra}")
        path = save_horizontal_bars(
            labels,
            rank["disorganization_proxy"].tolist(),
            out_dir / f"subject_rank_disorganization_proxy_w{window_size}.png",
            f"{run_dir.name}: sujetos con mayor proxy de desorganización NLP (w{window_size})",
            xlabel="proxy = - promedio de conectividad z / perfil organizado",
        )
        if path:
            rows.append({"figure": str(path), "kind": "subject_rank_disorganization"})

    # Clinically useful scatterplots with the subject-level proxy, but only when
    # there is enough signal to justify a figure.
    proxy_corr_rows: list[dict] = []
    for target in ["TOTAL", "NPLAN", "MOT", "COG", "Barratt (pre)", "Age", "School year", "COHERENCIA NARRATIVA", "Verbal fluency tasks", "Naming task", "Reading comprehension task"]:
        if target not in df_w.columns:
            continue
        x = pd.to_numeric(df_w["disorganization_proxy"], errors="coerce")
        y = pd.to_numeric(df_w[target], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < min_n or x[mask].nunique() < 2 or y[mask].nunique() < 2:
            continue
        r, p = stats.spearmanr(x[mask], y[mask])
        if pd.isna(r) or abs(float(r)) < min_abs_r:
            continue
        proxy_corr_rows.append({"target": target, "r": float(r), "p": float(p), "n": int(mask.sum())})

    if proxy_corr_rows:
        proxy_corr = pd.DataFrame(proxy_corr_rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
        proxy_corr_path = out_dir / f"subject_disorganization_proxy_correlations_w{window_size}.csv"
        proxy_corr.to_csv(proxy_corr_path, index=False)
        rows.append({"table": str(proxy_corr_path), "kind": "subject_proxy_correlations"})
        for _, corr_row in proxy_corr.head(6).iterrows():
            target = str(corr_row["target"])
            path = save_scatter(
                df_w,
                "disorganization_proxy",
                target,
                out_dir / f"subject_disorganization_vs_{safe_name(target)}_w{window_size}.png",
                f"{run_dir.name}: proxy de desorganización NLP vs {target} (w{window_size})\nSpearman r={corr_row['r']:.3f}, n={int(corr_row['n'])}",
                group_col=group_col if group_col in df_w.columns else None,
                xlabel="proxy de desorganización NLP",
                ylabel=target,
            )
            if path:
                rows.append({"figure": str(path), "kind": "subject_proxy_scatter", "target": target})

    append_manifest(rows, run_dir, subdir="nlp_profile")
    print(f"[{run_dir.name}] subject figures/tables: {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create subject-level NLP profiles from existing outputs.")
    parser.add_argument("--run-dir", default="outputs/04_windows_random1000")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--level", default="file", choices=["file", "activity", "all"])
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--group-col", default="Tipo")
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-abs-r", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_subject_figures(
        Path(args.run_dir),
        window_size=args.window_size,
        level=args.level,
        top_n=args.top_n,
        group_col=args.group_col,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
    )


if __name__ == "__main__":
    main()
