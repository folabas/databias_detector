"""Streamlit component: rich visualizations for group representation and outcomes.

- Group representation (counts) via bar and pie charts
- Outcome positive rates per group (bar)
- Optional false positive rate per group if predictions provided (bar)
- Numeric correlation heatmap
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def _coerce_binary(series: pd.Series) -> pd.Series | None:
    """Try to coerce a series to binary 0/1.
    - If numeric with <=2 unique values: return as int
    - If string with <=2 unique values: map consistently to 0/1
    - If numeric probabilities/scores: threshold at 0.5
    Returns coerced series or None if cannot coerce.
    """
    s = series.dropna()
    if s.empty:
        return None
    uniq = s.unique()
    nunique = len(uniq)
    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        # Binary numeric
        if nunique <= 2:
            return series.astype(float).round().astype(int)
        # Probabilities/scores
        if series.min() >= 0 and series.max() <= 1 and nunique > 2:
            return (series >= 0.5).astype(int)
        # Cannot coerce reliably
        return None
    # Strings / categorical
    if nunique <= 2:
        # Deterministic mapping: sort values and map first to 0, second to 1
        vals = sorted([str(v) for v in uniq])
        mapping = {vals[0]: 0, vals[1]: 1} if len(vals) == 2 else {vals[0]: 1}
        return series.astype(str).map(mapping).astype(int)
    return None


def render_group_visualizations(df: pd.DataFrame, sensitive_feature: str, target_col: str, predictions_col: str | None):
    st.subheader("Group Representation")
    if sensitive_feature not in df.columns:
        st.info("Sensitive feature not found in dataset.")
        return

    # Group counts
    counts = df[sensitive_feature].value_counts(dropna=False).reset_index()
    counts.columns = [sensitive_feature, "count"]
    try:
        fig_bar = px.bar(counts, x=sensitive_feature, y="count", title="Group Counts", text="count")
        fig_bar.update_layout(xaxis_title=sensitive_feature, yaxis_title="Count")
        st.plotly_chart(fig_bar, config={"responsive": True, "displayModeBar": True}, use_container_width=True)
        fig_pie = px.pie(counts, names=sensitive_feature, values="count", title="Group Share (Pie)")
        st.plotly_chart(fig_pie, config={"responsive": True, "displayModeBar": True}, use_container_width=True)
    except Exception:
        st.dataframe(counts)

    st.subheader("Outcome Rates by Group")
    if target_col not in df.columns:
        st.info("Target column not found in dataset.")
    else:
        target_bin = _coerce_binary(df[target_col])
        if target_bin is None:
            st.warning("Target is not binary; cannot compute outcome rates. Choose a binary column or preprocess.")
        else:
            # Compute positive rates by group
            tmp = pd.DataFrame({sensitive_feature: df[sensitive_feature], "y": target_bin})
            grp = tmp.dropna(subset=[sensitive_feature, "y"]).groupby(sensitive_feature)["y"].mean().reset_index()
            grp.columns = [sensitive_feature, "positive_rate"]
            try:
                fig_rate = px.bar(grp, x=sensitive_feature, y="positive_rate", title="Positive Outcome Rate by Group")
                fig_rate.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_rate, config={"responsive": True, "displayModeBar": True}, use_container_width=True)
            except Exception:
                st.dataframe(grp)

    # Optional FPR chart if predictions provided
    if predictions_col and predictions_col in df.columns and target_col in df.columns:
        st.subheader("False Positive Rate by Group (if predictions provided)")
        pred_bin = _coerce_binary(df[predictions_col])
        target_bin = _coerce_binary(df[target_col])
        if pred_bin is None or target_bin is None:
            st.info("Predictions/target could not be coerced to binary; skipping FPR chart.")
        else:
            tmp = pd.DataFrame({
                sensitive_feature: df[sensitive_feature],
                "y": target_bin,
                "pred": pred_bin,
            }).dropna(subset=[sensitive_feature, "y", "pred"])
            # FPR per group: among negatives (y==0), fraction with pred==1
            def fpr(g):
                neg = g[g["y"] == 0]
                return (neg["pred"] == 1).mean() if len(neg) > 0 else np.nan
            fpr_by_group = tmp.groupby(sensitive_feature).apply(fpr).reset_index(name="FPR")
            try:
                fig_fpr = px.bar(fpr_by_group, x=sensitive_feature, y="FPR", title="False Positive Rate by Group")
                fig_fpr.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_fpr, config={"responsive": True, "displayModeBar": True}, use_container_width=True)
            except Exception:
                st.dataframe(fpr_by_group)

    # Correlation heatmap for numeric columns
    st.subheader("Correlation Heatmap (numeric features)")
    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)
        try:
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig_corr, config={"responsive": True, "displayModeBar": True}, use_container_width=True)
        except Exception:
            st.dataframe(corr)
    else:
        st.info("Not enough numeric columns to compute correlations.")