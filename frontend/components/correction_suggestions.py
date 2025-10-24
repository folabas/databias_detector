"""Streamlit component: bias correction suggestions.

Renders a compact list of actionable steps (resampling, reweighting, thresholding, calibration, data collection).
"""
import streamlit as st
from typing import List

def render_correction_suggestions(suggestions: List[str] | None):
    st.subheader("Bias Correction Suggestions")
    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.info("No suggestions generated. If fairness is acceptable, no action may be required.")