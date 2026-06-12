import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
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
    compute_subject_metrics,
    plot_target_vs_predicted,
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

    fig_mae_test = bar_r2_comparison(df, "mae", "MAE test [IC 95%]")
    st.plotly_chart(fig_mae_test, use_container_width=True)
    fig_mae_val = bar_r2_comparison(df, "mae", "MAE validation [IC 95%]", suffix="_val")
    st.plotly_chart(fig_mae_val, use_container_width=True)
    fig_r2_test = bar_r2_comparison(df, "r2", "R² test [IC 95%]")
    st.plotly_chart(fig_r2_test, use_container_width=True)
    fig_r2_val = bar_r2_comparison(df, "r2", "R² validation [IC 95%]", suffix="_val")
    st.plotly_chart(fig_r2_val, use_container_width=True)

with tab_single:
    avail_windows = sorted(set(w for w, _ in completed))
    avail_experiments = sorted(set(e for _, e in completed))
    avail_targets = get_targets()

    col_metric, col_win, col_exp, col_tgt = st.columns(4)
    with col_metric:
        bt_metric = st.selectbox("Metric", ["MAE", "R²"], index=0, key="bt_metric")
    with col_win:
        bt_window = st.selectbox("Window", avail_windows, key="bt_window")
    with col_exp:
        bt_exp = st.selectbox(
            "Experiment", avail_experiments,
            format_func=lambda x: EXPERIMENT_LABELS.get(x, x),
            key="bt_exp",
        )
    with col_tgt:
        bt_target = st.selectbox(
            "Target", get_targets(window=bt_window, experiment=bt_exp),
            key="bt_target",
        )

    preds = load_predictions(bt_window, bt_exp, bt_target)
    if preds is None or len(preds) == 0:
        st.warning("No predictions data available for the selected combination.")
        st.stop()

    scenario_label = f"W{bt_window} {bt_exp}  {bt_target}"

    # --- Target vs Predicted — Test | Validation (side by side) ---
    col_test, col_val = st.columns(2)
    with col_test:
        sd_test = compute_subject_metrics(preds, set_name="TEST")
        if len(sd_test) > 0:
            fig_test = plot_target_vs_predicted(
                sd_test, set_name="Test", scenario_label=scenario_label,
            )
            st.plotly_chart(fig_test, use_container_width=True)
        else:
            st.info("No TEST data.")
    with col_val:
        sd_val = compute_subject_metrics(preds, set_name="VALIDATION")
        if len(sd_val) > 0:
            fig_val = plot_target_vs_predicted(
                sd_val, set_name="Validation", scenario_label=scenario_label,
            )
            st.plotly_chart(fig_val, use_container_width=True)
        else:
            st.info("No VALIDATION data.")

    # --- Distribution: y_true across subjects + boxplot of individual values ---
    sub_yt = preds["y_true"]
    subj_means = sub_yt.groupby(preds["subject"]).mean()
    if len(sub_yt) > 0:
        col_hist, col_box = st.columns([3, 1])
        with col_hist:
            subj_list = sorted(subj_means.index)
            _init_counts = Counter(s.split("-")[-1] for s in subj_list)
            _display_map = {}
            for s in subj_list:
                ini = s.split("-")[-1]
                if _init_counts[ini] > 1:
                    _display_map[s] = f"{ini} ({s.split('-')[2]})"
                else:
                    _display_map[s] = ini

            default_subj = subj_list[0]
            highlight_subj = st.session_state.get("comp_highlight_subj", default_subj)
            subj_val = float(subj_means[highlight_subj])

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=subj_means.values, nbinsx=40,
                marker_color="#1f77b4", opacity=0.7, name="Subjects",
            ))

            overall_mean = float(subj_means.mean())
            fig_dist.add_vline(
                x=overall_mean, line_dash="dash", line_color="red", line_width=2,
                annotation_text=f"Mean = {overall_mean:.2f}",
                annotation_position="top right",
            )

            fig_dist.add_vline(
                x=subj_val, line_dash="dash", line_color="green", line_width=2,
                annotation_text=highlight_subj.split("-")[-1][:8],
                annotation_position="top left",
            )

            fig_dist.update_layout(
                title="Target Variable Distribution by Subject",
                xaxis_title=f"{bt_target} mean per subject",
                yaxis_title="Number of subjects",
                template="plotly_white",
                height=450, bargap=0.05,
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.selectbox(
                "Highlight subject",
                options=subj_list,
                format_func=lambda s: _display_map.get(s, s.split("-")[-1]),
                key="comp_highlight_subj",
            )
        with col_box:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=sub_yt.values,
                name=bt_target,
                boxmean="sd",
                marker_color="#1f77b4",
            ))
            fig_box.update_layout(
                title=f"{bt_target} variance",
                yaxis_title="y_true",
                template="plotly_white",
                height=450,
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

with tab_scenario:
    if not completed:
        st.warning("No completed experiments for this task.")
        st.stop()

    col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 1])
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
    with col_s4:
        se_metric = st.selectbox("Metric", ["MAE", "R²"], index=0, key="se_metric")

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
    is_mae = se_metric == "MAE"
    if is_mae:
        sdf = sdf.sort_values("mae_val_mean")
    else:
        sdf = sdf.sort_values("r2_val_mean", ascending=False)

    st.subheader(f"Scenario Comparison Table — {se_metric}")
    if is_mae:
        display_cols = {
            "label": "Scenario",
            "mae_test_mean": "MAE test",
            "mae_test_lower": "MAE lower",
            "mae_test_upper": "MAE upper",
            "mae_val_mean": "MAE val",
            "mae_val_lower": "MAE val lower",
            "mae_val_upper": "MAE val upper",
            "model": "Model",
            "n_features": "N Feat",
            "pct_below": "% R²<0",
        }
    else:
        display_cols = {
            "label": "Scenario",
            "r2_test_mean": "R² test",
            "r2_test_lower": "R² lower",
            "r2_test_upper": "R² upper",
            "r2_val_mean": "R² val",
            "r2_val_lower": "R² val lower",
            "r2_val_upper": "R² val upper",
            "model": "Model",
            "n_features": "N Feat",
            "pct_below": "% R²<0",
        }
    display_df = sdf[list(display_cols.keys())].rename(columns=display_cols)
    if is_mae:
        for c in ["MAE test", "MAE lower", "MAE upper", "MAE val", "MAE val lower", "MAE val upper"]:
            display_df[c] = display_df[c].round(3)
    else:
        for c in ["R² test", "R² lower", "R² upper", "R² val", "R² val lower", "R² val upper"]:
            display_df[c] = display_df[c].round(4)
    display_df["% R²<0"] = display_df["% R²<0"].round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"Metrics Overview — {se_metric} — Validation & Test")

    col_left, col_right = st.columns(2)
    suffix = ""
    metric_key_val = f"mae_val" if is_mae else "r2_val"
    metric_key_test = f"mae_test" if is_mae else "r2_test"
    metric_label = "MAE" if is_mae else "R²"
    with col_left:
        fig_val = metric_comparison_chart(
            sdf, metric_key_val, metric_label,
            f"{metric_label} validation — {scenario_target}",
            show_hline_zero=not is_mae,
        )
        st.plotly_chart(fig_val, use_container_width=True)
    with col_right:
        fig_test = metric_comparison_chart(
            sdf, metric_key_test, metric_label,
            f"{metric_label} test — {scenario_target}",
            show_hline_zero=not is_mae,
        )
        st.plotly_chart(fig_test, use_container_width=True)
