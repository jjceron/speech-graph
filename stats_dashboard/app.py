from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats_dashboard.utils.correlation import (
    IMPULSIVITY_MEASURES,
    apply_conditional_filtering,
    apply_corrections,
    build_correlation_table,
    flatten_df_for_display,
    get_graph_features_for_network,
    resolve_column_name,
    style_table_html,
    export_to_excel,
)
from stats_dashboard.utils.loader import (
    ALL_VARS,
    GRAPH_FEATURES,
    METADATA_VARS,
    METRICS_DIR,
    NETWORK_TYPE_OPTIONS,
    Z_GRAPH_FEATURES,
    build_file_list,
    get_graph_feature_columns,
    is_random_graph,
    load_and_prepare_metadata,
    load_metric_file,
    strip_z_prefix,
)
from stats_dashboard.utils.plots import (
    compute_stats_text,
    histogram_simple,
    join_plot_loess,
)

st.set_page_config(
    page_title="SpeechGraph — Statistical Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 SpeechGraph — Statistical Analysis Dashboard")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.file_infos = []
    st.session_state.metadata_df = None
    st.session_state.task_label = "Task2"
    st.session_state.network_type = NETWORK_TYPE_OPTIONS[0]

tab_setup, tab_descriptive, tab_join, tab_corr = st.tabs(
    ["⚙️ Setup", "📈 Descriptive Statistics", "🔬 Join Plots", "📋 Correlations"]
)

with tab_setup:
    st.header("Selection of Task and Network Type")

    col1, col2 = st.columns(2)
    with col1:
        task_label = st.selectbox(
            "Select Task",
            ["Task2", "Task6-A", "Task6-B", "Task7"],
            index=["Task2", "Task6-A", "Task6-B", "Task7"].index(
                st.session_state.task_label
            ),
        )
    with col2:
        network_type = st.selectbox(
            "Select Network Type",
            NETWORK_TYPE_OPTIONS,
            index=NETWORK_TYPE_OPTIONS.index(st.session_state.network_type)
            if st.session_state.network_type in NETWORK_TYPE_OPTIONS
            else 0,
        )

    if st.button("🔍 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading files and metadata..."):
            file_infos = build_file_list(task_label, network_type)
            metadata_df = load_and_prepare_metadata()

            if not file_infos:
                st.error(
                    f"No files found for {task_label} / {network_type} in "
                    f"{METRICS_DIR / ('Task' + task_label.split('Task')[1][0])}"
                )
            else:
                for finfo in file_infos:
                    finfo["df"] = load_metric_file(finfo["path"])

                st.session_state.file_infos = file_infos
                st.session_state.metadata_df = metadata_df
                st.session_state.task_label = task_label
                st.session_state.network_type = network_type
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(file_infos)} files successfully!")

    if st.session_state.data_loaded:
        st.subheader("Loaded files")
        for finfo in st.session_state.file_infos:
            st.code(f"{finfo['label']}: {finfo['path'].name}")

        meta = st.session_state.metadata_df
        st.subheader("Metadata preview (first 5 rows)")
        preview_cols = ["Cod"] + [c for c in METADATA_VARS if c in meta.columns]
        st.dataframe(meta[preview_cols].head(), use_container_width=True, hide_index=True)

        st.caption(
            f"Metadata shape: {meta.shape} | "
            f"MOT_V4 = items 8+13+16+21+23 | COG_V1 = items 3+6"
        )

if not st.session_state.data_loaded:
    st.info("👈 Go to the **Setup** tab and click **Load Data** to begin.")
    st.stop()

task_label = st.session_state.task_label
network_type = st.session_state.network_type
file_infos = st.session_state.file_infos
metadata_df = st.session_state.metadata_df

# ──────────────────────────────────────────────────
# TAB 2: Descriptive Statistics
# ──────────────────────────────────────────────────
with tab_descriptive:
    st.header("Descriptive Statistics")

    graph_feat_options = list(GRAPH_FEATURES)
    if is_random_graph(network_type):
        graph_feat_options = [f"z_{f}" for f in graph_feat_options]

    all_var_options = METADATA_VARS + graph_feat_options
    selected_var = st.selectbox("Select variable", all_var_options)

    is_meta_var = selected_var in METADATA_VARS

    if is_meta_var:
        df_plot = metadata_df
        col_name = selected_var
        fig = histogram_simple(df_plot, col_name)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(compute_stats_text(df_plot[col_name].dropna()))
    else:
        actual_col = selected_var
        clean_col = strip_z_prefix(actual_col)

        n_cols = 2
        n_rows = (len(file_infos) + n_cols - 1) // n_cols

        for row_idx in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                file_idx = row_idx * n_cols + col_idx
                if file_idx >= len(file_infos):
                    break
                finfo = file_infos[file_idx]
                df_metric = finfo["df"]

                if actual_col not in df_metric.columns:
                    with cols[col_idx]:
                        st.warning(f"Column '{actual_col}' not in {finfo['label']}")
                    continue

                fig = histogram_simple(df_metric, actual_col)
                with cols[col_idx]:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"**{finfo['label']}** — "
                        + compute_stats_text(df_metric[actual_col].dropna())
                    )

# ──────────────────────────────────────────────────
# TAB 3: Join Plots
# ──────────────────────────────────────────────────
with tab_join:
    st.header("Join Plots: Graph Features vs Metadata")

    if is_random_graph(network_type):
        gf_options = Z_GRAPH_FEATURES
    else:
        gf_options = GRAPH_FEATURES

    col_gf, col_md = st.columns(2)
    with col_gf:
        selected_gf = st.selectbox("Graph feature (Y axis)", gf_options)
    with col_md:
        selected_md = st.selectbox("Metadata variable (X axis)", METADATA_VARS)

    n_cols = 2
    n_rows = (len(file_infos) + n_cols - 1) // n_cols

    for row_idx in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            file_idx = row_idx * n_cols + col_idx
            if file_idx >= len(file_infos):
                break
            finfo = file_infos[file_idx]
            df_metric = finfo["df"]

            if selected_gf not in df_metric.columns or selected_md not in metadata_df.columns:
                with cols[col_idx]:
                    st.warning(f"Columns not available for {finfo['label']}")
                continue

            merged = df_metric.merge(
                metadata_df[["Cod", selected_md]],
                left_on="file",
                right_on="Cod",
                how="inner",
            )

            if len(merged) < 5:
                with cols[col_idx]:
                    st.warning(f"Too few merged observations for {finfo['label']}")
                continue

            fig = join_plot_loess(
                merged,
                x_col=selected_md,
                y_col=selected_gf,
                title=finfo["label"],
            )
            with cols[col_idx]:
                st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────
# TAB 4: Correlations
# ──────────────────────────────────────────────────
with tab_corr:
    st.header("Correlation Analysis: Simple and Partial")

    file_labels = [f["label"] for f in file_infos]

    graph_feat_cols = get_graph_features_for_network(network_type, task_label)

    zero_var_features = set()
    for gf in graph_feat_cols:
        for finfo in file_infos:
            actual_col = resolve_column_name(gf, finfo["df"], network_type)
            if actual_col is not None and actual_col in finfo["df"].columns:
                if finfo["df"][actual_col].var() < 1e-12:
                    zero_var_features.add(gf)

    with st.spinner("Computing correlations..."):
        corr_df = build_correlation_table(
            file_infos, metadata_df, network_type, task_label
        )

        if corr_df.empty:
            st.warning("No correlations could be computed with current selection.")
            st.stop()

        corr_df, correction_info = apply_corrections(corr_df, file_labels)
        corr_df = apply_conditional_filtering(corr_df, file_labels)

    display_df = flatten_df_for_display(corr_df, file_labels)

    first_alpha = None
    for cinfo in correction_info.values():
        first_alpha = cinfo["bonferroni_alpha"]
        break
    if first_alpha is not None:
        st.markdown(
            f"Bonferroni α corregido: **{first_alpha:.6f}** &nbsp;|&nbsp; **FDR q = 0.05**"
        )

    if not corr_df.empty:
        excel_bytes = export_to_excel(
            display_df, corr_df, file_labels,
            f"{task_label} - {network_type}.xlsx",
            correction_info,
            zero_var_features=zero_var_features,
        )
        st.download_button(
            label="📥 Print table",
            data=excel_bytes,
            file_name=f"{task_label} - {network_type}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    html_table = style_table_html(
        display_df, corr_df, file_labels, correction_info,
        zero_var_features=zero_var_features,
    )
    st.markdown(html_table, unsafe_allow_html=True)

    with st.expander("📖 Notes"):
        st.markdown("""
- **rho values in green** indicate |rho| ≥ 0.1 (partial correlations).
- **p-values in gray** indicate p > 0.05.
- **Bold p-values** pass Bonferroni correction.
- **Asterisks (*)** mark p-values that pass FDR correction (q=0.05).
- Empty cells: |rho| < 0.1 or constant variable.
- Correlations computed as Spearman (simple and partial, controlling for School year).
- Corrections applied per group: (Impulsivity measure × File).
- **Filas en gris itálico**: variables con varianza cero; no se calcularon correlaciones
  y no se incluyeron en la corrección Bonferroni/FDR.
""")
