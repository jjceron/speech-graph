import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.loader import (
    list_completed,
    load_best_report,
    load_all_reports,
    get_task,
    get_targets,
    get_experiments,
    get_windows,
)
from utils.plots import (
    bar_r2_comparison,
    forest_plot,
    metric_comparison_chart,
    TARGET_COLORS,
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
)

st.set_page_config(page_title="Cross-Experiment", page_icon="📊", layout="wide")
st.title(f"📊 Cross-Experiment Comparison — Task {get_task()}")

completed = list_completed()

tab_all, tab_single, tab_scenario = st.tabs(["All Scenarios", "By Target", "Scenario Explorer"])

with tab_all:
    reports = load_all_reports()
    all_data = []
    for (w, e, t), r in reports.items():
        ts = r.get("test_summary", {})
        r2_mean = ts.get("r2_mean_test", 0)
        if r2_mean is None:
            continue
        all_data.append({
            "label": f"W{w} {e:<10} {t}",
            "r2_mean": r2_mean,
            "r2_lower": ts.get("r2_ci_lower_test", 0),
            "r2_upper": ts.get("r2_ci_upper_test", 0),
            "color": TARGET_COLORS.get(t, "#333"),
        })
    if all_data:
        all_data.sort(key=lambda x: x["r2_mean"])
        fig = forest_plot(all_data)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Each dot shows R² test with 95% CI. "
        "Dots to the right of the red line (R²=0) indicate scenarios that outperform the mean baseline. "
        "If the entire CI is > 0, the scenario is statistically significant at α=0.05."
    )

with tab_single:
    avail_targets = get_targets()
    target = st.selectbox("Target", avail_targets, index=0)

    rows = []
    for w, e in completed:
        r = load_best_report(w, e, target)
        if r:
            ts = r.get("test_summary", {})
            rows.append(
                {
                    "label": f"W{w} — {e}",
                    "window": w,
                    "experiment": e,
                    "target": target,
                    "r2_mean": ts.get("r2_mean_test", 0),
                    "r2_lower": ts.get("r2_ci_lower_test", 0),
                    "r2_upper": ts.get("r2_ci_upper_test", 0),
                    "mae_mean": ts.get("mae_mean_test", 0),
                    "mae_lower": ts.get("mae_ci_lower_test", 0),
                    "mae_upper": ts.get("mae_ci_upper_test", 0),
                    "rho_mean": ts.get("rho_mean_test", 0) or 0,
                    "rho_lower": ts.get("rho_ci_lower_test", 0) or 0,
                    "rho_upper": ts.get("rho_ci_upper_test", 0) or 0,
                    "pct_below": ts.get("r2_below_zero_test", 0) * 100,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("No data for this target.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = bar_r2_comparison(df, "r2", f"R² test — {target}")
        st.plotly_chart(fig_r2, use_container_width=True)
    with col2:
        fig_mae = bar_r2_comparison(df, "mae", f"MAE test — {target}")
        st.plotly_chart(fig_mae, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_rho = bar_r2_comparison(df, "rho", f"Spearman ρ test — {target}")
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
            title=f"% R² < 0 — {target}",
            xaxis_title="Window — Experiment",
            yaxis_title="% Splits",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_below, use_container_width=True)

    if target == "COG_V1":
        st.markdown("""
        **Insight:** COG_V1 is the only target with positive R² — only in **raw** and **rawzscore** experiments.
        Z-scores alone lose the signal entirely. W20 raw shows the best performance (R² ≈ 0.041, MAE ≈ 0.970).
        """)
    elif target == "MOT":
        st.markdown("""
        **Insight:** MOT shows weak, inconsistent signal. W10 zscores (R² ≈ 0.023) and W20 zscores (R² ≈ 0.031)
        are the best performers. Raw features turn negative at W20.
        """)

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
