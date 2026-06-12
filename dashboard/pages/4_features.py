import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import list_completed, get_targets, load_rfe_ranking, load_selected_features, load_best_report
from utils.plots import rfe_ranking_chart
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Features", page_icon="🔬", layout="wide")
render_sidebar()
st.title("🔬 Features Analysis")

completed = list_completed()
if not completed:
    st.warning("No completed experiments.")
    st.stop()

tab_rfe, tab_shap = st.tabs(["RFE", "SHAP Analysis"])

with tab_rfe:
    col_w, col_e, col_t = st.columns(3)
    with col_w:
        windows = sorted(set(w for w, _ in completed))
        window = st.selectbox("Window", windows, index=0, key="feat_w")
    with col_e:
        exps = [e for w, e in completed if w == window]
        experiment = st.selectbox("Experiment", exps, index=0, key="feat_e")
    with col_t:
        target = st.selectbox("Target", get_targets(), index=0, key="feat_t")

    st.subheader(f"W{window} — {experiment} — {target}")

    col1, col2 = st.columns(2)

    with col1:
        ranking = load_rfe_ranking(window, experiment, target)
        if ranking is not None:
            fig = rfe_ranking_chart(ranking)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RFE ranking data.")

    with col2:
        report = load_best_report(window, experiment, target)
        if report:
            feat = report.get("selected_features", [])
            bp = report.get("best_params", {})
            st.subheader("Selected Features")
            st.write(f"**{len(feat)}** features selected by RFE")
            st.write(f"Model: **{bp.get('regressor', '?')}**")
            st.dataframe(pd.DataFrame({"Feature": feat}), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"Feature Comparison Across Experiments — Target: {target}")

    all_rows = []
    for w, e in completed:
        report = load_best_report(w, e, target)
        if report:
            feat = report.get("selected_features", [])
            bp = report.get("best_params", {})
            vs = report.get("validation_summary", {})
            ts = report.get("test_summary", {})
            all_rows.append({
                "Window": f"W{w}",
                "Experiment": e,
                "Target": target,
                "Model": bp.get("regressor", "?"),
                "MAE val": vs.get("mae_mean_val"),
                "MAE test": ts.get("mae_mean_test"),
                "N Features": len(feat),
                "Features": ", ".join(feat),
            })

    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values("MAE val")
    st.dataframe(df_all, use_container_width=True, hide_index=True)

with tab_shap:
    col_sw, col_se, col_st = st.columns(3)
    with col_sw:
        shap_windows = sorted(set(w for w, _ in completed))
        sw = st.selectbox("Window", shap_windows, index=0, key="shap_w")
    with col_se:
        shap_exps = [e for w, e in completed if w == sw]
        se = st.selectbox("Experiment", shap_exps, index=0, key="shap_e")
    with col_st:
        stg = st.selectbox("Target", get_targets(), index=0, key="shap_t")

    from utils.loader import _exp_dir
    shap_dir = _exp_dir(sw, se) / stg
    shap_summary_path = shap_dir / "shap_summary.json"
    shap_values_path = shap_dir / "shap_values.csv"

    if not shap_summary_path.exists():
        st.info("SHAP values not computed yet. Run:  python src/analysis/compute_shap.py --all")
        st.stop()

    with open(shap_summary_path) as f:
        summary = json.load(f)

    st.subheader(f"SHAP Feature Importance — W{sw} {se} — {stg}")
    st.caption(f"Model: **{summary['regressor']}** | Subjects: {summary['n_subjects']} | Features: {summary['n_features']}")

    mas = summary["mean_abs_shap"]
    if not mas or all(v == 0.0 for v in mas.values()):
        st.warning("All SHAP values are zero — the model collapsed to constant prediction (degenerate).")
        st.stop()

    mas_sorted = sorted(mas.items(), key=lambda x: x[1], reverse=True)
    feat_names = [x[0] for x in mas_sorted]
    feat_vals = [x[1] for x in mas_sorted]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feat_vals[::-1],
        y=feat_names[::-1],
        orientation="h",
        marker_color="#2ca02c",
    ))
    fig.update_layout(
        title="Mean |SHAP| per feature",
        xaxis_title="mean |SHAP|",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, len(feat_names) * 30),
    )
    st.plotly_chart(fig, use_container_width=True)

    shap_df = pd.read_csv(shap_values_path)
    shap_val_cols = [c for c in shap_df.columns if c not in ("subject", "y_true", "y_pred")]
    display_df = shap_df[["subject"] + shap_val_cols + ["y_true", "y_pred"]].copy()
    for col in shap_val_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    display_df["y_true"] = display_df["y_true"].map(lambda x: f"{x:.1f}")
    display_df["y_pred"] = display_df["y_pred"].map(lambda x: f"{x:.4f}")

    st.subheader("Per-subject SHAP values")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
