import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import os

# Resolve backend URL without requiring Streamlit secrets
try:
    BACKEND_URL = st.secrets["backend_url"]
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Upload limits configured via .streamlit/config.toml

st.set_page_config(page_title="DataBias Detector", page_icon="🧮", layout="wide")
st.title("🧮 DataBias Detector")
st.write("Upload your dataset, select a sensitive feature, and analyze fairness.")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])


# All detection logic is now handled by the intelligent backend analysis


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.write("Columns:", list(df.columns))

    # Get comprehensive dataset analysis from intelligent backend
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=20)
        resp_json = resp.json()
        
        # Extract intelligent analysis results
        detected_sensitive = resp_json.get("detected_sensitive", [])
        binary_columns = resp_json.get("binary_columns", [])
        target_candidates = resp_json.get("target_candidates", [])
        dataset_analysis = resp_json.get("dataset_analysis", {})
        
        # Show intelligent analysis summary
        if dataset_analysis:
            with st.expander("🧠 Intelligent Dataset Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**📊 Dataset Overview**")
                    st.write(f"• Total rows: {dataset_analysis.get('total_rows', 'N/A')}")
                    st.write(f"• Total columns: {dataset_analysis.get('total_columns', 'N/A')}")
                
                with col2:
                    st.write("**🎯 Recommended Columns**")
                    if detected_sensitive:
                        st.write(f"• Sensitive features: {', '.join(detected_sensitive[:3])}")
                    if target_candidates:
                        st.write(f"• Target candidates: {', '.join(target_candidates[:3])}")
                
                with col3:
                    st.write("**✅ Binary Columns**")
                    if binary_columns:
                        st.write(f"• Binary columns: {', '.join(binary_columns[:3])}")
                    else:
                        st.write("• No binary columns detected")
        
    except Exception as e:
        st.warning(f"Could not get intelligent analysis from backend: {e}")
        # Fallback to basic detection
        detected_sensitive = []
        binary_columns = []
        target_candidates = []
        dataset_analysis = {}
    
    # Prefer binary target candidates
    binary_target_candidates = [col for col in target_candidates if col in binary_columns]

    # UI selections with tooltips
    st.markdown("### Column Selection")
    
    # Sensitive feature selection with tooltip
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        sensitive_default = (
            detected_sensitive[0] if detected_sensitive else ("gender" if "gender" in df.columns else None)
        )
        sensitive_feature = st.selectbox(
            "Sensitive feature",
            options=list(df.columns),
            index=(list(df.columns).index(sensitive_default) if sensitive_default in df.columns else 0),
            help="The demographic attribute to analyze for bias (e.g., gender, race, age group). This should be a categorical column representing different groups you want to compare for fairness."
        )
    
    # Target/Outcome selection with intelligent filtering
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        # Use binary columns if available, otherwise show all with warning
        if binary_columns:
            target_options = binary_columns
            # Default to best binary target candidate
            if binary_target_candidates:
                target_default = binary_target_candidates[0]
            else:
                target_default = binary_columns[0]
        else:
            target_options = list(df.columns)
            target_default = target_candidates[0] if target_candidates else df.columns[0]
        
        target_col = st.selectbox(
            "Target/Outcome column (binary)" + (" ⚠️ No binary columns detected" if not binary_columns else ""),
            options=target_options,
            index=(target_options.index(target_default) if target_default in target_options else 0),
            help="The outcome you want to measure bias in. Must be binary (exactly 2 unique values like Yes/No, 0/1, True/False). This represents what you're trying to predict or the decision being made."
        )
    
    # Predictions selection with tooltip
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        predictions_col = st.selectbox(
            "Predictions column (optional)",
            options=["<none>"] + list(df.columns),
            index=0,
            help="Optional: The model's predictions or scores. If provided, should be binary (0/1) or probability scores. If not provided, the target column will be used as a proxy for predictions in the analysis."
        )

    # Show column information
    if binary_columns:
        st.info(f"✅ **Binary columns detected:** {', '.join(binary_columns)}")
    else:
        st.warning("⚠️ **No binary columns detected.** Fairness metrics require binary targets. Consider preprocessing your data to create binary columns (e.g., age >= 30 → 1, else 0).")
    
    # Show selected column stats
    with st.expander("📊 Selected Column Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Sensitive Feature: {sensitive_feature}**")
            st.write(f"Unique values: {df[sensitive_feature].nunique()}")
            st.write(f"Sample values: {list(df[sensitive_feature].dropna().unique()[:3])}")
        
        with col2:
            st.write(f"**Target: {target_col}**")
            st.write(f"Unique values: {df[target_col].nunique()}")
            st.write(f"Sample values: {list(df[target_col].dropna().unique()[:3])}")
            if df[target_col].nunique() > 2:
                st.warning("⚠️ Not binary - metrics may be unavailable")
        
        with col3:
            if predictions_col and predictions_col != "<none>":
                st.write(f"**Predictions: {predictions_col}**")
                st.write(f"Unique values: {df[predictions_col].nunique()}")
                st.write(f"Sample values: {list(df[predictions_col].dropna().unique()[:3])}")
            else:
                st.write("**Predictions: None selected**")
                st.write("Target will be used as proxy")

    if st.button("Analyze Bias", type="primary"):
        st.info("Analyzing…")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        data = {
            "sensitive_feature": sensitive_feature,
            "target": target_col,
        }
        if predictions_col and predictions_col != "<none>":
            data["predictions_col"] = predictions_col

        try:
            res = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=60)
            result = res.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        if result.get("error"):
            st.error(result["error"])
        else:
            fairness_score = result.get("fairness_score")
            metrics = result.get("metrics", {})
            comp_scores = result.get("component_scores", {})

            if fairness_score is not None:
                st.success(f"Fairness Score: {fairness_score:.2f} / 100")
            else:
                # Provide friendlier guidance when metrics are unavailable
                nuniq_target = (
                    df[target_col].dropna().nunique() if target_col in df.columns else None
                )
                nuniq_pred = (
                    df[predictions_col].dropna().nunique()
                    if predictions_col and predictions_col != "<none>" and predictions_col in df.columns
                    else None
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
                            f"- Predictions '{predictions_col}' has {nuniq_pred} unique values; threshold to 0/1 (e.g., score >= 0.5 → 1, else 0)."
                        )
                    else:
                        suggestions.append(
                            f"- Predictions '{predictions_col}' looks binary; ensure it is correctly selected."
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
            st.write({k: round(v, 4) if v is not None else None for k, v in metrics.items()})

            # Visualizations
            st.subheader("Representation by Sensitive Feature")
            if sensitive_feature in df.columns:
                fig = px.pie(df, names=sensitive_feature, title=f"Distribution of {sensitive_feature}")
                st.plotly_chart(fig, use_container_width=True)

            # Outcome rate by group (bar chart)
            if sensitive_feature in df.columns and target_col in df.columns:
                try:
                    # Compute mean of binary-like target per group (works as rate if binary)
                    # If target isn't binary, this is the average numeric value per group.
                    group_rates = df.groupby(sensitive_feature)[target_col].mean(numeric_only=True)
                    bar_df = group_rates.reset_index().rename(columns={target_col: "rate"})
                    fig_bar = px.bar(bar_df, x=sensitive_feature, y="rate", title=f"Outcome rate by {sensitive_feature}")
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    pass

            st.subheader("Explanation")
            st.write(
                "Demographic Parity Difference and Equal Opportunity Difference measure how outcomes/predictions differ across groups. "
                "Values near 0 indicate fairness; larger absolute values indicate potential bias."
            )

            # Optional export
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export JSON"):
                    st.download_button(
                        label="Download results.json",
                        data=io.BytesIO(bytes(str(result), "utf-8")),
                        file_name="results.json",
                        mime="application/json",
                    )