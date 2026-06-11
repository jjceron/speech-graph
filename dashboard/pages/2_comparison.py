import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import (
    list_completed,
    load_best_report,
    load_all_reports,
    load_predictions,
    get_task,
    get_targets,
    get_experiments,
    get_windows,
)
from utils.plots import (
    bar_r2_comparison,
    metric_comparison_chart,
    EXPERIMENT_LABELS,
)
from utils.sidebar import render_sidebar

st.set_page_config(page_title="Cross-Experiment", page_icon="📊", layout="wide")
render_sidebar()
st.title(f"📊 Cross-Experiment Comparison — Task {get_task()}")

completed = list_completed()

tab_all, tab_single, tab_scenario = st.tabs(["All Scenarios", "By Target", "Scenario Explorer"])

with tab_all:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_targets = get_targets()
        selected_targets = st.multiselect(
            "Targets", options=all_targets, default=all_targets, key="all_targets"
        )
    with col_f2:
        all_exps = get_experiments()
        selected_exps = st.multiselect(
            "Experiments",
            options=all_exps,
            default=all_exps,
            format_func=lambda x: EXPERIMENT_LABELS.get(x, x),
            key="all_exps",
        )

    reports = load_all_reports()
    all_rows = []
    for (w, e, t), r in reports.items():
        if t not in selected_targets or e not in selected_exps:
            continue
        vs = r.get("validation_summary", {})
        ts = r.get("test_summary", {})
        all_rows.append({
            "label": f"W{w} {e}  {t}",
            "target": t,
            "window": w,
            "experiment": e,
            "r2_mean": ts.get("r2_mean_test", 0),
            "r2_lower": ts.get("r2_ci_lower_test", 0),
            "r2_upper": ts.get("r2_ci_upper_test", 0),
            "r2_val_mean": vs.get("r2_mean_val"),
            "r2_val_lower": vs.get("r2_ci_lower_val"),
            "r2_val_upper": vs.get("r2_ci_upper_val"),
            "mae_mean": ts.get("mae_mean_test"),
            "mae_lower": ts.get("mae_ci_lower_test"),
            "mae_upper": ts.get("mae_ci_upper_test"),
            "mae_val_mean": vs.get("mae_mean_val"),
            "mae_val_lower": vs.get("mae_ci_lower_val"),
            "mae_val_upper": vs.get("mae_ci_upper_val"),
        })
    if not all_rows:
        st.warning("No data for the selected filters.")
        st.stop()

    df = pd.DataFrame(all_rows)

    fig_r2_test = bar_r2_comparison(df, "r2", "R² test [IC 95%]")
    st.plotly_chart(fig_r2_test, use_container_width=True)
    fig_r2_val = bar_r2_comparison(df, "r2", "R² validation [IC 95%]", suffix="_val")
    st.plotly_chart(fig_r2_val, use_container_width=True)
    fig_mae_test = bar_r2_comparison(df, "mae", "MAE test [IC 95%]")
    st.plotly_chart(fig_mae_test, use_container_width=True)
    fig_mae_val = bar_r2_comparison(df, "mae", "MAE validation [IC 95%]", suffix="_val")
    st.plotly_chart(fig_mae_val, use_container_width=True)

with tab_single:
    avail_targets = get_targets()
    avail_windows = sorted(set(w for w, _ in completed))

    col_metric, col_tgts, col_wins = st.columns(3)
    with col_metric:
        bt_metric = st.selectbox("Metric", ["R²", "MAE"], key="bt_metric")
    with col_tgts:
        sel_targets = st.multiselect("Targets", avail_targets, default=avail_targets, key="bt_targets")
    with col_wins:
        sel_windows = st.multiselect("Windows", avail_windows, default=avail_windows, key="bt_wins")

    rows = []
    preds_for_dist = None
    for w, e in completed:
        if w not in sel_windows:
            continue
        for t in sel_targets:
            r = load_best_report(w, e, t)
            if not r:
                continue
            ts = r.get("test_summary", {})
            vs = r.get("validation_summary", {})
            rows.append({
                "label": f"W{w} — {e}  {t}",
                "window": w,
                "experiment": e,
                "target": t,
                "r2_mean": ts.get("r2_mean_test", 0),
                "r2_lower": ts.get("r2_ci_lower_test", 0),
                "r2_upper": ts.get("r2_ci_upper_test", 0),
                "r2_val_mean": vs.get("r2_mean_val"),
                "r2_val_lower": vs.get("r2_ci_lower_val"),
                "r2_val_upper": vs.get("r2_ci_upper_val"),
                "mae_mean": ts.get("mae_mean_test", 0),
                "mae_lower": ts.get("mae_ci_lower_test", 0),
                "mae_upper": ts.get("mae_ci_upper_test", 0),
                "mae_val_mean": vs.get("mae_mean_val"),
                "mae_val_lower": vs.get("mae_ci_lower_val"),
                "mae_val_upper": vs.get("mae_ci_upper_val"),
                "rho_mean": ts.get("rho_mean_test", 0) or 0,
                "rho_lower": ts.get("rho_ci_lower_test", 0) or 0,
                "rho_upper": ts.get("rho_ci_upper_test", 0) or 0,
                "pct_below": ts.get("r2_below_zero_test", 0) * 100,
            })
            if preds_for_dist is None:
                preds_for_dist = load_predictions(w, e, t)

    if not rows:
        st.warning("No data for the selected filters.")
        st.stop()

    df = pd.DataFrame(rows)
    metric_key = "r2" if bt_metric == "R²" else "mae"

    tgt_str = ", ".join(sel_targets) if len(sel_targets) <= 3 else f"{len(sel_targets)} targets"

    # --- Metric Val | Metric Test ---
    col1, col2 = st.columns(2)
    with col1:
        fig_val = bar_r2_comparison(df, metric_key, f"{bt_metric} validation — {tgt_str}", suffix="_val")
        st.plotly_chart(fig_val, use_container_width=True)
    with col2:
        fig_test = bar_r2_comparison(df, metric_key, f"{bt_metric} test — {tgt_str}")
        st.plotly_chart(fig_test, use_container_width=True)

    # --- Distribution: val + test overlapped (full-width) ---
    if preds_for_dist is not None:
        val_vals = preds_for_dist[preds_for_dist["set"] == "VAL"]["y_true"].dropna().values
        test_vals = preds_for_dist[preds_for_dist["set"] == "TEST"]["y_true"].dropna().values
        if len(val_vals) > 0 or len(test_vals) > 0:
            fig_dist = go.Figure()
            if len(val_vals) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=val_vals, nbinsx=40, name="Validation",
                    marker_color="#1f77b4", opacity=0.5,
                ))
            if len(test_vals) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=test_vals, nbinsx=40, name="Test",
                    marker_color="#d62728", opacity=0.7,
                ))
            fig_dist.update_layout(
                barmode="overlay",
                title=f"Target variable distribution — {tgt_str}",
                xaxis_title="y_true",
                yaxis_title="Splits",
                template="plotly_white",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # --- Spearman ρ | % R²<0 ---
    col3, col4 = st.columns(2)
    with col3:
        fig_rho = bar_r2_comparison(df, "rho", f"Spearman ρ test — {tgt_str}")
        st.plotly_chart(fig_rho, use_container_width=True)
    with col4:
        fig_below = go.Figure()
        fig_below.add_trace(
            go.Bar(
                x=df["label"],
                y=df["pct_below"],
                marker_color="#d62728",
                text=df["pct_below"].round(1).astype(str) + "%",
                textposition="outside",
            )
        )
        fig_below.update_layout(
            title=f"% R² < 0 — {tgt_str}",
            xaxis_title="Window — Experiment",
            yaxis_title="% Splits",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_below, use_container_width=True)

with tab_scenario:
    if not completed:
        st.warning("No completed experiments for this task.")
        st.stop()

    col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
    with col_s1:
        avail_targets = get_targets()
        scenario_target = st.selectbox("Target", avail_targets, index=0, key="scenario_t")
    with col_s2:
        avail_exps = sorted(set(e for _, e in completed))
        selected_exps = st.multiselect(
            "Experiments",
            options=avail_exps,
            default=avail_exps,
            format_func=lambda x: EXPERIMENT_LABELS.get(x, x),
        )
    with col_s3:
        avail_wins = sorted(set(w for w, _ in completed), key=int)
        selected_wins = st.multiselect(
            "Windows",
            options=avail_wins,
            default=avail_wins,
        )

    scenario_rows = []
    for w, e in completed:
        if e not in selected_exps or w not in selected_wins:
            continue
        r = load_best_report(w, e, scenario_target)
        if not r:
            continue
        vs = r.get("validation_summary", {})
        ts = r.get("test_summary", {})
        bp = r.get("best_params", {})
        feat = r.get("selected_features", [])
        scenario_rows.append({
            "label": f"W{w} {e}",
            "window": w,
            "experiment": e,
            "target": scenario_target,
            "r2_val_mean": vs.get("r2_mean_val"),
            "r2_val_lower": vs.get("r2_ci_lower_val"),
            "r2_val_upper": vs.get("r2_ci_upper_val"),
            "r2_test_mean": ts.get("r2_mean_test"),
            "r2_test_lower": ts.get("r2_ci_lower_test"),
            "r2_test_upper": ts.get("r2_ci_upper_test"),
            "mae_val_mean": vs.get("mae_mean_val"),
            "mae_val_lower": vs.get("mae_ci_lower_val"),
            "mae_val_upper": vs.get("mae_ci_upper_val"),
            "mae_test_mean": ts.get("mae_mean_test"),
            "mae_test_lower": ts.get("mae_ci_lower_test"),
            "mae_test_upper": ts.get("mae_ci_upper_test"),
            "model": bp.get("regressor", "?"),
            "n_features": len(feat),
            "pct_below": ts.get("r2_below_zero_test", 0) * 100,
        })

    if not scenario_rows:
        st.warning("No data for the selected filters.")
        st.stop()

    sdf = pd.DataFrame(scenario_rows)

    st.subheader("Scenario Comparison Table")
    display_cols = {
        "label": "Scenario",
        "r2_test_mean": "R² test",
        "r2_test_lower": "R² lower",
        "r2_test_upper": "R² upper",
        "mae_test_mean": "MAE test",
        "mae_test_lower": "MAE lower",
        "mae_test_upper": "MAE upper",
        "r2_val_mean": "R² val",
        "mae_val_mean": "MAE val",
        "model": "Model",
        "n_features": "N Feat",
        "pct_below": "% R²<0",
    }
    display_df = sdf[list(display_cols.keys())].rename(columns=display_cols)
    display_df["R² test"] = display_df["R² test"].round(4)
    display_df["MAE test"] = display_df["MAE test"].round(3)
    display_df["R² val"] = display_df["R² val"].round(4)
    display_df["MAE val"] = display_df["MAE val"].round(3)
    display_df["% R²<0"] = display_df["% R²<0"].round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Metrics Overview — Validation & Test")

    row1_left, row1_right = st.columns(2)
    with row1_left:
        fig_r2_val = metric_comparison_chart(
            sdf, "r2_val", "R²", f"R² validation — {scenario_target}",
            show_hline_zero=True,
        )
        st.plotly_chart(fig_r2_val, use_container_width=True)
    with row1_right:
        fig_r2_test = metric_comparison_chart(
            sdf, "r2_test", "R²", f"R² test — {scenario_target}",
            show_hline_zero=True,
        )
        st.plotly_chart(fig_r2_test, use_container_width=True)

    row2_left, row2_right = st.columns(2)
    with row2_left:
        fig_mae_val = metric_comparison_chart(
            sdf, "mae_val", "MAE", f"MAE validation — {scenario_target}",
        )
        st.plotly_chart(fig_mae_val, use_container_width=True)
    with row2_right:
        fig_mae_test = metric_comparison_chart(
            sdf, "mae_test", "MAE", f"MAE test — {scenario_target}",
        )
        st.plotly_chart(fig_mae_test, use_container_width=True)
