import json
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import list_completed, get_windows, get_experiments, get_targets, load_rfe_ranking, load_selected_features, load_best_report
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
        window = st.selectbox("Window", get_windows(), index=0, key="feat_w")
    with col_e:
        experiment = st.selectbox("Experiment", [e for e in get_experiments() if get_targets(window=window, experiment=e)], index=0, key="feat_e")
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
            st.info(f"**Model:** {bp.get('regressor', '?')}, **{len(feat)}** features selected by RFE")
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
        sw = st.selectbox("Window", get_windows(), index=0, key="shap_w")
    with col_se:
        se = st.selectbox("Experiment", [e for e in get_experiments() if get_targets(window=sw, experiment=e)], index=0, key="shap_e")
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

    report_shap = load_best_report(sw, se, stg)
    caption_parts = [
        f"Model: **{summary['regressor']}**",
        f"Subjects: {summary['n_subjects']}",
        f"Features: {summary['n_features']}",
    ]
    if report_shap:
        vs = report_shap.get("validation_summary", {})
        ts = report_shap.get("test_summary", {})
        mae_v = vs.get("mae_mean_val")
        mae_t = ts.get("mae_mean_test")
        r2_v = vs.get("r2_mean_val")
        r2_t = ts.get("r2_mean_test")
        if mae_v is not None:
            caption_parts.append(f"MAE val: {mae_v:.4f}")
        if mae_t is not None:
            caption_parts.append(f"MAE test: {mae_t:.4f}")
        if r2_v is not None:
            caption_parts.append(f"R² val: {r2_v:.4f}")
        if r2_t is not None:
            caption_parts.append(f"R² test: {r2_t:.4f}")
    st.caption(" | ".join(caption_parts))

    shap_df = pd.read_csv(shap_values_path)
    shap_val_cols = [c for c in shap_df.columns if c not in ("subject", "y_true", "y_pred")]

    feat_data = []
    for col in shap_val_cols:
        vals = shap_df[col].values
        abs_vals = np.abs(vals)
        mean_v = float(vals.mean())
        mean_abs = float(abs_vals.mean())
        lo_s = float(np.percentile(vals, 25))
        hi_s = float(np.percentile(vals, 75))
        lo_a = float(np.percentile(abs_vals, 25))
        hi_a = float(np.percentile(abs_vals, 75))
        feat_data.append((col, mean_v, mean_abs, lo_s, hi_s, lo_a, hi_a))

    feat_data.sort(key=lambda x: x[2], reverse=True)

    all_zero = all(x[2] < 1e-12 for x in feat_data)
    if all_zero:
        st.warning("All SHAP values are zero — the model collapsed to constant prediction (degenerate).")
        st.stop()

    feat_names = [x[0] for x in feat_data]

    # --- Chart 1: Ranking by Magnitude ---
    means_abs = [x[2] for x in feat_data]
    err_lo_a = [x[2] - x[5] for x in feat_data]
    err_hi_a = [x[6] - x[2] for x in feat_data]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=means_abs[::-1],
        y=feat_names[::-1],
        orientation="h",
        marker_color="#2ca02c",
        error_x=dict(
            type="data", symmetric=False,
            array=err_hi_a[::-1], arrayminus=err_lo_a[::-1],
            visible=True, thickness=1, width=3,
        ),
    ))
    fig1.update_layout(
        title="SHAP Feature Importance — Ranking by Magnitude",
        xaxis_title="Mean |SHAP|",
        template="plotly_white",
        height=max(450, len(feat_names) * 40),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: Direction ---
    means_s = [x[1] for x in feat_data]
    colors = ["#d62728" if m < 0 else "#2ca02c" for m in means_s]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=means_s[::-1],
        y=feat_names[::-1],
        orientation="h",
        marker_color=colors[::-1],
    ))
    fig2.update_layout(
        title="Mean Signed SHAP Contribution — Direction",
        xaxis_title="Mean SHAP contribution (→ higher prediction)",
        template="plotly_white",
        height=max(450, len(feat_names) * 40),
        shapes=[{
            "type": "line", "x0": 0, "y0": -0.5,
            "x1": 0, "y1": len(feat_names) - 0.5,
            "line": {"color": "gray", "width": 1, "dash": "dot"},
        }],
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("⬇ Sorted by mean |SHAP| descending — highest importance first")

    # --- Beeswarm ---
    feat_values_path = shap_dir / "shap_feature_values.csv"
    if feat_values_path.exists():
        feat_vals_df = pd.read_csv(feat_values_path)
        bf = go.Figure()
        nf = len(feat_names)
        rng = np.random.RandomState(42)
        for i, feat in enumerate(feat_names):
            sv = shap_df[feat].values
            fv = feat_vals_df[feat].values.astype(float)
            fmin, fmax = fv.min(), fv.max()
            fv_norm = (fv - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(fv)
            yj = rng.uniform(-0.2, 0.2, len(sv))
            bf.add_trace(go.Scatter(
                x=sv,
                y=[nf - 1 - i + y for y in yj],
                mode="markers",
                marker=dict(
                    size=4, color=fv_norm,
                    colorscale="RdYlBu_r",
                    cmin=0, cmax=1,
                    showscale=i == 0,
                    colorbar=dict(
                        title="Feature value",
                        tickvals=[0, 1], ticktext=["Low", "High"],
                    ) if i == 0 else None,
                ),
                name=feat,
                customdata=fv,
                hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{customdata:.4f}}<extra></extra>",
            ))
        bf.update_layout(
            title="SHAP value distribution by subject",
            xaxis_title="SHAP value (→ higher prediction)",
            yaxis=dict(tickvals=list(range(nf)), ticktext=feat_names[::-1]),
            template="plotly_white",
            height=max(450, nf * 60),
            showlegend=False,
            shapes=[{
                "type": "line", "x0": 0, "y0": -0.5,
                "x1": 0, "y1": nf - 0.5,
                "line": {"color": "gray", "width": 1, "dash": "dot"},
            }],
        )
        st.plotly_chart(bf, use_container_width=True)

    display_df = shap_df[["subject"] + shap_val_cols + ["y_true", "y_pred"]].copy()
    display_df["subject"] = display_df["subject"].apply(lambda s: s.split("-")[-1])
    for col in shap_val_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    display_df["y_true"] = display_df["y_true"].map(lambda x: f"{x:.1f}")
    display_df["y_pred"] = display_df["y_pred"].map(lambda x: f"{x:.4f}")

    st.subheader("Per-subject SHAP values")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
