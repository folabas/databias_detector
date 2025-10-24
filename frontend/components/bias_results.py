"""Streamlit component: bias metrics and fairness score display.

Shows aggregated fairness score with emoji feedback and metrics table.
"""
import streamlit as st
import pandas as pd

def render_bias_results(fairness_score, metrics: dict, component_scores: dict, df: pd.DataFrame, target_col: str, predictions_col: str | None):
    if fairness_score is not None:
        if fairness_score >= 80:
            status_emoji = "🟢"
        elif fairness_score >= 60:
            status_emoji = "🟡"
        else:
            status_emoji = "🔴"
        st.success(f"{status_emoji} Fairness Score: {fairness_score:.2f} / 100")
    else:
        nuniq_target = df[target_col].dropna().nunique() if target_col in df.columns else None
        nuniq_pred = (
            df[predictions_col].dropna().nunique() if predictions_col and predictions_col in df.columns else None
        )
        binary_candidates = [c for c in df.columns if df[c].dropna().nunique() <= 2]
        suggestions = []
        if nuniq_target is not None:
            if nuniq_target > 2:
                suggestions.append(
                    f"- Target '{target_col}' has {nuniq_target} unique values; choose a binary column (two unique values)."
                )
            else:
                suggestions.append(
                    f"- Target '{target_col}' looks binary; if metrics are still unavailable, check predictions selection or data types."
                )
        if nuniq_pred is not None:
            if nuniq_pred > 2:
                suggestions.append(
                    f"- Predictions has {nuniq_pred} unique values; threshold to 0/1 (e.g., score >= 0.5 → 1, else 0)."
                )
        if "income" in df.columns:
            suggestions.append("- Consider using 'income' as the target if available.")
        if binary_candidates:
            show_candidates = ", ".join(binary_candidates[:5])
            suggestions.append(f"- Binary-like columns detected: {show_candidates}")

        st.warning(
            "Fairness Score unavailable. Metrics require a binary target and (optionally) binary predictions."
        )
        st.info(
            "\n".join(suggestions)
            if suggestions
            else "Select a binary target (two unique values) or binarize a numeric/label column (e.g., value >= threshold → 1, else 0)."
        )

    st.subheader("Metric Breakdown")
    st.write({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})