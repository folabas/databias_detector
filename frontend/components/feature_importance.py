"""Streamlit component: feature importance visualization.

Uses Plotly bar chart when SHAP or correlation importances are available.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def render_feature_importance(explainability: dict, title: str):
    if explainability and explainability.get("feature_importances"):
        imp = explainability["feature_importances"]
        imp_df = pd.DataFrame(imp)
        try:
            fig_imp = px.bar(imp_df.head(20), x="feature", y="importance", title=title)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.dataframe(imp_df.head(20))
    elif explainability:
        st.info(explainability.get("reason", "Explainability not available."))