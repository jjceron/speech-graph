# Plan: Add subject highlight to Target Distribution in Comparison page

## File to modify
`dashboard/pages/2_comparison.py`

## Change
Replace the `with col_hist:` block (lines 143-156) to add:
1. Red dashed vline for overall mean of `subj_means`
2. Selectbox "Highlight subject on histogram" (same pattern as Subjects page, key=`comp_highlight_subj`)
3. Green dashed vline for selected subject

`col_box` with the boxplot (variance chart) stays untouched.

## New code
```python
with col_hist:
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

    subj_list = sorted(subj_means.index)
    highlight_subj = st.selectbox(
        "Highlight subject on histogram",
        [f"{s} (μ={subj_means[s]:.2f})" for s in subj_list],
        key="comp_highlight_subj",
    )
    if highlight_subj:
        subj_id = highlight_subj.split(" (")[0]
        subj_val = float(subj_means[subj_id])
        fig_dist.add_vline(
            x=subj_val, line_dash="dash", line_color="green", line_width=2,
            annotation_text=subj_id.split("-")[-1][:8],
            annotation_position="top left",
        )

    fig_dist.update_layout(
        title=f"Target distribution — {scenario_label}",
        xaxis_title=f"{bt_target} mean per subject",
        yaxis_title="Number of subjects",
        template="plotly_white",
        height=350, bargap=0.05,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
```

## No changes to
- `col_box` (boxplot / variance chart)
- Any other tabs or pages
