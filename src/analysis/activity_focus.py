from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis import correlation_table
from src.preprocessing import canonical_activity

GRAPH_METRICS = [
    "mean_z_lcc",
    "mean_z_lsc",
    "mean_z_edges",
    "mean_z_density",
    "mean_z_asp",
    "mean_z_l2",
    "mean_z_l3",
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

TARGETS_CORE = [
    "Age",
    "School year",
    "TOTAL",
    "NPLAN",
    "MOT",
    "COG",
    "Barratt (pre)",
    "TOTAL_zscore",
    "COG_zscore",
    "MOT_zscore",
    "Naming task",
    "COHERENCIA NARRATIVA",
    "Conceptualization task",
    "Reading comprehension task",
    "Verbal fluency tasks",
    "FLUIDEZ VERBAL - PRUEBA SEMANTICA - FRUTAS",
    "FLUIDEZ VERBAL - PRUEBA SEMANTICA - ANIMALES",
    "FLUIDEZ VERBAL - PRUEBA FONETICA",
]

BARRATT_ITEMS = [f"{i}." for i in range(1, 27)]
ID_COLS = {"code", "file", "Cod", "level", "activity", "activity_number", "activity_index", "start_time", "end_time", "_merge", "_join_code"}


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        match = canonical_activity(part)
        if match.number is not None:
            out.append(match.number)
        else:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out


def _activity_number_series(df: pd.DataFrame) -> pd.Series:
    if "activity_number" in df.columns:
        numeric = pd.to_numeric(df["activity_number"], errors="coerce")
        if numeric.notna().any():
            return numeric.astype("Int64")
    if "activity" in df.columns:
        return df["activity"].map(lambda x: canonical_activity(x).number).astype("Int64")
    return pd.Series(pd.NA, index=df.index, dtype="Int64")


def _bh_fdr(p_values: Iterable[float]) -> list[float]:
    p = np.array([float(x) if pd.notna(x) else np.nan for x in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q.tolist()
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[mask])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q.tolist()


def _available_numeric(df: pd.DataFrame, requested: Iterable[str], min_n: int = 3) -> list[str]:
    cols: list[str] = []
    for col in requested:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= min_n:
            cols.append(col)
    return cols


def _label_cols(df: pd.DataFrame, min_nonzero: int) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if not col.startswith("label_ratio_"):
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if (values > 0).sum() >= min_nonzero:
            cols.append(col)
    preferred_order = [
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
    return [c for c in preferred_order if c in cols] + sorted([c for c in cols if c not in preferred_order])


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [m for m in GRAPH_METRICS if m in df.columns and pd.to_numeric(df[m], errors="coerce").notna().sum() >= 3]


def _valid_activity_rows(df: pd.DataFrame, window_size: int, require_metadata: bool = True) -> pd.Series:
    token_ok = pd.to_numeric(df.get("token_count", pd.Series(index=df.index, data=np.nan)), errors="coerce") >= window_size
    win_ok = pd.to_numeric(df.get("window_count", pd.Series(index=df.index, data=np.nan)), errors="coerce") >= 1
    if "_merge" in df.columns and require_metadata:
        meta_ok = df["_merge"].astype(str).eq("both")
    else:
        meta_ok = pd.Series(True, index=df.index)
    return token_ok.fillna(False) & win_ok.fillna(False) & meta_ok


def _qc_rows(df: pd.DataFrame, target_nums: set[int], secondary_nums: set[int], window_size: int) -> list[dict]:
    rows: list[dict] = []
    for act_num, sub in df.groupby("activity_number", dropna=False):
        act_int = int(act_num) if pd.notna(act_num) else None
        role = "target" if act_int in target_nums else "secondary" if act_int in secondary_nums else "other"
        token = pd.to_numeric(sub.get("token_count"), errors="coerce")
        wc = pd.to_numeric(sub.get("window_count"), errors="coerce")
        matched = int(sub["_merge"].astype(str).eq("both").sum()) if "_merge" in sub.columns else int(len(sub))
        valid = _valid_activity_rows(sub, window_size=window_size, require_metadata=False)
        rows.append({
            "section": "activity_qc",
            "activity": f"Actividad{act_int}" if act_int is not None else "NA",
            "activity_number": act_int,
            "activity_role": role,
            "n_rows": int(len(sub)),
            "n_unique_codes": int(sub["code"].nunique()) if "code" in sub.columns else int(len(sub)),
            "n_with_metadata": matched,
            "n_valid_window": int(valid.sum()),
            "token_min": float(token.min()) if token.notna().any() else np.nan,
            "token_median": float(token.median()) if token.notna().any() else np.nan,
            "token_mean": float(token.mean()) if token.notna().any() else np.nan,
            "window_count_median": float(wc.median()) if wc.notna().any() else np.nan,
            "window_size_required": window_size,
        })
    return rows


def _correlation_rows(
    df: pd.DataFrame,
    metrics: list[str],
    targets: list[str],
    section: str,
    method: str,
    min_n: int,
    min_abs_r: float,
) -> list[dict]:
    rows: list[dict] = []
    for act_num, sub in df.groupby("activity_number", dropna=False):
        act_int = int(act_num) if pd.notna(act_num) else None
        corr = correlation_table(sub, metrics, targets, method=method)
        if corr.empty:
            continue
        corr["abs_r"] = corr["r"].abs()
        corr = corr[(corr["n"] >= min_n) & (corr["abs_r"] >= min_abs_r)].copy()
        if corr.empty:
            continue
        corr["q_fdr_within_section"] = _bh_fdr(corr["p"])
        corr = corr.sort_values(["abs_r", "p"], ascending=[False, True])
        for r in corr.itertuples(index=False):
            rows.append({
                "section": section,
                "activity": f"Actividad{act_int}" if act_int is not None else "NA",
                "activity_number": act_int,
                "metric": r.metric,
                "target": r.target,
                "r": float(r.r),
                "abs_r": float(abs(r.r)),
                "p": float(r.p),
                "q_fdr_within_section": float(r.q_fdr_within_section) if pd.notna(r.q_fdr_within_section) else np.nan,
                "n": int(r.n),
            })
    return rows


def _group_difference_rows(
    df: pd.DataFrame,
    variables: list[str],
    group_col: str,
    min_group_n: int,
) -> list[dict]:
    if group_col not in df.columns:
        return []
    rows: list[dict] = []
    for act_num, sub in df.groupby("activity_number", dropna=False):
        act_int = int(act_num) if pd.notna(act_num) else None
        groups = [g for g in sub[group_col].dropna().unique()]
        if len(groups) != 2:
            continue
        g1, g2 = sorted(groups, key=lambda x: str(x))
        for var in variables:
            if var not in sub.columns:
                continue
            vals = pd.to_numeric(sub[var], errors="coerce")
            a = vals[sub[group_col].eq(g1)].dropna()
            b = vals[sub[group_col].eq(g2)].dropna()
            if len(a) < min_group_n or len(b) < min_group_n or a.nunique() < 2 and b.nunique() < 2:
                continue
            try:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                continue
            rows.append({
                "section": "group_difference",
                "activity": f"Actividad{act_int}" if act_int is not None else "NA",
                "activity_number": act_int,
                "metric": var,
                "target": group_col,
                "group_1": str(g1),
                "group_2": str(g2),
                "group_1_n": int(len(a)),
                "group_2_n": int(len(b)),
                "group_1_median": float(a.median()),
                "group_2_median": float(b.median()),
                "group_1_mean": float(a.mean()),
                "group_2_mean": float(b.mean()),
                "delta_median_g1_minus_g2": float(a.median() - b.median()),
                "p": float(p),
            })
    if rows:
        pvals = [r.get("p", np.nan) for r in rows]
        qvals = _bh_fdr(pvals)
        for r, q in zip(rows, qvals):
            r["q_fdr_within_section"] = q
    return rows


def _subject_proxy_rows(df: pd.DataFrame, group_col: str, top_n: int = 15) -> list[dict]:
    needed = [c for c in ["mean_density", "mean_repeated_edges_ratio", "mean_l2", "mean_l3", "mean_nodes", "mean_lsc"] if c in df.columns]
    if len(needed) < 3 or "code" not in df.columns:
        return []
    rows: list[dict] = []
    for act_num, sub in df.groupby("activity_number", dropna=False):
        act_int = int(act_num) if pd.notna(act_num) else None
        tmp = sub.copy()
        proxy = pd.Series(0.0, index=tmp.index)
        used = []
        for col, sign in [
            ("mean_density", 1),
            ("mean_repeated_edges_ratio", 1),
            ("mean_l2", 1),
            ("mean_l3", 1),
            ("mean_nodes", -1),
            ("mean_lsc", -1),
        ]:
            if col not in tmp.columns:
                continue
            x = pd.to_numeric(tmp[col], errors="coerce")
            if x.notna().sum() < 5 or x.std(ddof=0) == 0:
                continue
            z = (x - x.mean()) / x.std(ddof=0)
            proxy = proxy + sign * z.fillna(0)
            used.append(col)
        if not used:
            continue
        tmp["activity_disorganization_proxy"] = proxy / max(1, len(used))
        top = tmp.sort_values("activity_disorganization_proxy", ascending=False).head(top_n)
        for rank, r in enumerate(top.itertuples(index=False), start=1):
            rows.append({
                "section": "subject_activity_proxy_top",
                "activity": f"Actividad{act_int}" if act_int is not None else "NA",
                "activity_number": act_int,
                "rank": rank,
                "code": getattr(r, "code"),
                "file": getattr(r, "file", ""),
                "group": getattr(r, group_col, "") if group_col in tmp.columns else "",
                "activity_disorganization_proxy": float(getattr(r, "activity_disorganization_proxy")),
                "proxy_components": ";".join(used),
            })
    return rows


def _write_report(results: pd.DataFrame, output_md: Path, run_name: str, target_nums: list[int], secondary_nums: list[int], window_size: int, random_times: str | int | None) -> None:
    lines: list[str] = []
    lines.append(f"# Activity-focused speech graph analysis: {run_name}")
    lines.append("")
    lines.append("Este reporte resume el análisis por actividad/clase usando grafos dirigidos de palabras y ventana móvil de 30 palabras. Las transcripciones no se modifican: el código normaliza etiquetas del protocolo y remueve timestamps técnicos durante el análisis.")
    lines.append("")
    lines.append("## Parámetros")
    lines.append(f"- `window_size`: {window_size}")
    lines.append(f"- `random_times`: {random_times}")
    lines.append(f"- Actividades target: {', '.join('Actividad'+str(x) for x in target_nums)}")
    lines.append(f"- Actividades secundarias: {', '.join('Actividad'+str(x) for x in secondary_nums)}")
    lines.append("")

    qc = results[results["section"].eq("activity_qc")].copy()
    if not qc.empty:
        lines.append("## Control de calidad por actividad")
        show = qc.sort_values("activity_number")[["activity", "activity_role", "n_rows", "n_with_metadata", "n_valid_window", "token_median", "window_count_median"]]
        lines.append(show.to_markdown(index=False))
        lines.append("")

    for section, title in [("graph_correlation", "Correlaciones de métricas de grafos"), ("label_correlation", "Correlaciones de etiquetas discursivas")]:
        sub = results[results["section"].eq(section)].copy()
        lines.append(f"## {title}")
        if sub.empty:
            lines.append("No hubo asociaciones que pasaran los filtros actuales.")
        else:
            top = sub.sort_values(["abs_r", "p"], ascending=[False, True]).head(20)
            cols = ["activity", "metric", "target", "r", "p", "q_fdr_within_section", "n"]
            lines.append(top[cols].to_markdown(index=False))
        lines.append("")

    gd = results[results["section"].eq("group_difference")].copy()
    lines.append("## Diferencias entre grupos")
    if gd.empty:
        lines.append("No hubo diferencias entre grupos que pudieran calcularse con los filtros actuales.")
    else:
        top = gd.sort_values(["p"]).head(20)
        cols = ["activity", "metric", "target", "group_1", "group_2", "group_1_n", "group_2_n", "group_1_median", "group_2_median", "p", "q_fdr_within_section"]
        lines.append(top[cols].to_markdown(index=False))
    lines.append("")

    sp = results[results["section"].eq("subject_activity_proxy_top")].copy()
    lines.append("## Sujetos extremos por actividad")
    if sp.empty:
        lines.append("No se calculó proxy por actividad.")
    else:
        for activity, sub in sp.groupby("activity"):
            lines.append(f"### {activity}")
            cols = ["rank", "code", "group", "activity_disorganization_proxy"]
            lines.append(sub.sort_values("rank").head(10)[cols].to_markdown(index=False))
            lines.append("")

    lines.append("## Lectura metodológica")
    lines.append("- Actividad 2, 6 y 7 se consideran targets porque son más cercanas al perfil narrativo/discursivo que queremos estudiar.")
    lines.append("- Actividad 1, 4 y 5 sirven como análisis secundario para ver si una señal depende del tipo de tarea o aparece de forma transversal.")
    lines.append("- Sin baseline aleatorio, los resultados son exploratorios. Si una señal es consistente, conviene validarla con `--random-times 20` o `--random-times 100` en ventana 30.")
    lines.append("- Edad y curso deben considerarse covariables importantes porque ya mostraron asociación fuerte con las métricas globales.")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_activity_focus_outputs(
    input_csv: Path,
    output_dir: Path,
    window_size: int = 30,
    target_activities: str = "2,6,7",
    secondary_activities: str = "1,4,5",
    method: str = "spearman",
    min_n: int = 30,
    min_abs_r: float = 0.15,
    label_min_nonzero: int = 20,
    group_col: str = "Tipo",
    min_group_n: int = 10,
    include_barratt_items: bool = False,
) -> tuple[pd.DataFrame, Path, Path]:
    df = pd.read_csv(input_csv)
    out_dir = output_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "window_size" in df.columns:
        df = df[pd.to_numeric(df["window_size"], errors="coerce").eq(window_size)].copy()
    if "level" in df.columns:
        df = df[df["level"].astype(str).str.lower().eq("activity")].copy()

    df["activity_number"] = _activity_number_series(df)
    target_nums = parse_int_list(target_activities)
    secondary_nums = parse_int_list(secondary_activities)
    target_set = set(target_nums)
    secondary_set = set(secondary_nums)
    df["activity_role"] = df["activity_number"].map(lambda x: "target" if pd.notna(x) and int(x) in target_set else "secondary" if pd.notna(x) and int(x) in secondary_set else "other")

    # Graph metrics require enough linguistic material for a 30-word window.
    graph_valid = _valid_activity_rows(df, window_size=window_size, require_metadata=True)
    df_graph_valid = df[graph_valid].copy()

    # Label features such as SIN_RESPUESTA are meaningful precisely when a segment
    # has little or no lexical response, so label analyses only require metadata.
    if "_merge" in df.columns:
        meta_valid = df["_merge"].astype(str).eq("both")
    else:
        meta_valid = pd.Series(True, index=df.index)
    df_label_valid = df[meta_valid].copy()

    metrics = _metric_cols(df_graph_valid)
    targets = _available_numeric(df_label_valid, TARGETS_CORE + (BARRATT_ITEMS if include_barratt_items else []), min_n=min_n)
    labels = _label_cols(df_label_valid, min_nonzero=label_min_nonzero)

    rows: list[dict] = []
    rows.extend(_qc_rows(df, target_set, secondary_set, window_size))
    rows.extend(_correlation_rows(df_graph_valid, metrics, targets, "graph_correlation", method, min_n, min_abs_r))
    rows.extend(_correlation_rows(df_label_valid, labels, targets, "label_correlation", method, min_n, min_abs_r))
    rows.extend(_group_difference_rows(df_graph_valid, metrics, group_col=group_col, min_group_n=min_group_n))
    rows.extend(_group_difference_rows(df_label_valid, labels, group_col=group_col, min_group_n=min_group_n))
    rows.extend(_subject_proxy_rows(df_graph_valid[df_graph_valid["activity_number"].isin(target_nums + secondary_nums)], group_col=group_col, top_n=15))

    results = pd.DataFrame(rows)
    if not results.empty:
        # Add role to all rows.
        role_map = {n: "target" for n in target_nums} | {n: "secondary" for n in secondary_nums}
        if "activity_number" in results.columns:
            results["activity_role"] = results["activity_number"].map(lambda x: role_map.get(int(x), "other") if pd.notna(x) else "other")
    else:
        results = pd.DataFrame(columns=["section", "activity", "activity_number", "activity_role"])

    csv_path = out_dir / "activity_focus_results.csv"
    md_path = out_dir / "activity_focus_report.md"
    results.to_csv(csv_path, index=False)
    random_times = df.get("random_times", pd.Series([""])).dropna().astype(str).unique()
    random_text = ",".join(random_times[:3]) if len(random_times) else ""
    _write_report(results, md_path, output_dir.name, target_nums, secondary_nums, window_size, random_text)
    return results, csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create two-file activity-focused summary for speech graph results.")
    parser.add_argument("--input-csv", default="outputs/02_w30_by_activity/graph_metrics_all_windows_with_meta.csv")
    parser.add_argument("--output-dir", default="outputs/02_w30_by_activity")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--target-activities", default="2,6,7")
    parser.add_argument("--secondary-activities", default="1,4,5")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--label-min-nonzero", type=int, default=20)
    parser.add_argument("--group-col", default="Tipo")
    parser.add_argument("--min-group-n", type=int, default=10)
    parser.add_argument("--include-barratt-items", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, csv_path, md_path = generate_activity_focus_outputs(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
        target_activities=args.target_activities,
        secondary_activities=args.secondary_activities,
        method=args.method,
        min_n=args.min_n,
        min_abs_r=args.min_abs_r,
        label_min_nonzero=args.label_min_nonzero,
        group_col=args.group_col,
        min_group_n=args.min_group_n,
        include_barratt_items=args.include_barratt_items,
    )
    print(f"Activity focus CSV: {csv_path}")
    print(f"Activity focus report: {md_path}")


if __name__ == "__main__":
    main()
