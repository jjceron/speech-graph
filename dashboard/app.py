import streamlit as st
from utils.loader import list_completed, load_all_reports, load_best_report, get_task, get_windows, get_experiments, get_targets, EXPECTED_WINDOWS, EXPECTED_EXPERIMENTS
from utils.plots import TARGET_COLORS
from utils.sidebar import render_sidebar
import pandas as pd

st.set_page_config(
    page_title="SpeechGraph — Regression Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

completed = list_completed()

st.title("📊 SpeechGraph Regression Dashboard")
wins = get_windows()
win_range = f"W{wins[0]}–W{wins[-1]}" if wins else "—"
st.markdown(f"#### Task {get_task()} — Optuna Regression Results ({win_range})")

reports = load_all_reports()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments complete", f"{len(completed)}/{len(EXPECTED_WINDOWS) * len(EXPECTED_EXPERIMENTS)}")

best_r2 = -999
best_label = ""
best_mae = 999
best_mae_label = ""
for (w, e, t), r in reports.items():
    ts = r.get("test_summary", {})
    r2 = ts.get("r2_mean_test", -999)
    mae = ts.get("mae_mean_test", 999)
    if r2 is not None and r2 > best_r2:
        best_r2 = r2
        best_label = f"W{w} {e} {t}"
    if mae is not None and mae < best_mae:
        best_mae = mae
        best_mae_label = f"W{w} {e} {t}"
col2.metric("Best R²", f"{best_r2:.4f}", delta=best_label)
col3.metric("Best MAE", f"{best_mae:.3f}", delta=best_mae_label)

r2_vals = [r["test_summary"].get("r2_mean_test", -999) for r in reports.values()]
r2_pos = sum(1 for v in r2_vals if v > 0)
col4.metric("R² > 0", f"{r2_pos}/{len(r2_vals)}")

st.markdown("---")

col_view, col_set, col_metric = st.columns(3)
with col_view:
    view_mode = st.radio("View", ["Top 5", "All Results"], horizontal=True)
with col_set:
    set_mode = st.radio("Set", ["val", "test"], horizontal=True, key="set_mode")
with col_metric:
    sort_metric = st.selectbox("Sort by", ["MAE", "R²"], index=0)

suffix = set_mode

rows = []
for (w, e, t), r in reports.items():
    s = r.get("{}_summary".format("validation" if suffix == "val" else "test"), {})
    bp = r.get("best_params", {})
    feat = r.get("selected_features", [])
    rows.append({
        "sort_key": s.get("r2_mean_{}".format(suffix), 0),
        "sort_key_mae": s.get("mae_mean_{}".format(suffix), 999),
        "Window": f"W{w}",
        "Experiment": e,
        "Target": t,
        "R² [IC 95%]": f"{s.get(f'r2_mean_{suffix}', 0):.4f} [{s.get(f'r2_ci_lower_{suffix}', 0):.4f}, {s.get(f'r2_ci_upper_{suffix}', 0):.4f}]",
        "MAE [IC 95%]": f"{s.get(f'mae_mean_{suffix}', 0):.3f} [{s.get(f'mae_ci_lower_{suffix}', 0):.3f}, {s.get(f'mae_ci_upper_{suffix}', 0):.3f}]",
        "D2MAE [IC 95%]": f"{s.get(f'd2mae_mean_{suffix}', 0):.4f} [{s.get(f'd2mae_ci_lower_{suffix}', 0):.4f}, {s.get(f'd2mae_ci_upper_{suffix}', 0):.4f}]",
        "% R²<0": f"{s.get(f'r2_below_zero_{suffix}', 0) * 100:.1f}%",
        "Model": bp.get("regressor", "?"),
        "Features": ", ".join(feat) if feat else "-",
    })

if rows:
    df = pd.DataFrame(rows)
    if sort_metric == "R²":
        df = df.sort_values("sort_key", ascending=False).drop(columns=["sort_key", "sort_key_mae"])
    else:
        df = df.sort_values("sort_key_mae", ascending=True).drop(columns=["sort_key", "sort_key_mae"])
    if view_mode == "Top 5":
        df = df.head(5)
    st.subheader(f"{'Top 5' if view_mode == 'Top 5' else 'All'} Results — by {sort_metric}")
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Data from `outputs/regression_optuna/task{get_task()}/` — Last updated: see git log")
