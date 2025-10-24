"""Streamlit component: dataset preview and summary.

Renders head, column list, and a compact overview block.
"""
import streamlit as st
import pandas as pd

def render_dataset_preview(df: pd.DataFrame, dataset_analysis: dict):
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write("Columns:", list(df.columns))
    if dataset_analysis:
        with st.expander("🧠 Intelligent Dataset Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**📊 Dataset Overview**")
                st.write(f"• Total rows: {dataset_analysis.get('total_rows', 'N/A')}")
                st.write(f"• Total columns: {dataset_analysis.get('total_columns', 'N/A')}")
            with col2:
                st.write("**🎯 Recommended Columns**")
                sens = dataset_analysis.get("sensitive_features", []) or []
                targs = dataset_analysis.get("target_candidates", []) or []
                if sens:
                    st.write(f"• Sensitive features: {', '.join(sens[:3])}")
                if targs:
                    st.write(f"• Target candidates: {', '.join(targs[:3])}")
            with col3:
                st.write("**✅ Binary Columns**")
                bin_cols = dataset_analysis.get("binary_columns", []) or []
                if bin_cols:
                    st.write(f"• Binary columns: {', '.join(bin_cols[:3])}")
                else:
                    st.write("• No binary columns detected")