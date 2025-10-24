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

# Multilingual UI support
TRANSLATIONS = {
    "en": {"title": "🧮 DataBias Detector", "upload": "Upload your dataset", "upload_csv": "Upload a CSV", "analyze": "Analyze Bias", "ai_expl": "AI Explanation", "feat_influence": "Feature Influence (Explainability)", "lang": "Language", "theme": "Dark mode"},
    "fr": {"title": "🧮 Détecteur de Biais", "upload": "Téléchargez votre ensemble de données", "upload_csv": "Téléchargez un CSV", "analyze": "Analyser le biais", "ai_expl": "Explication IA", "feat_influence": "Influence des caractéristiques (Explicabilité)", "lang": "Langue", "theme": "Mode sombre"},
    "es": {"title": "🧮 Detector de Sesgo", "upload": "Sube tu conjunto de datos", "upload_csv": "Sube un CSV", "analyze": "Analizar sesgo", "ai_expl": "Explicación IA", "feat_influence": "Influencia de características (Explicabilidad)", "lang": "Idioma", "theme": "Modo oscuro"},
}
lang = st.sidebar.selectbox(TRANSLATIONS["en"]["lang"], options=["en","fr","es"], index=0, help="Choose your interface language.")
T = TRANSLATIONS.get(lang, TRANSLATIONS["en"]) 

# Dark mode toggle (CSS-based)
st.sidebar.checkbox(T["theme"], key="dark_mode", help="Toggle a simple dark theme.")
if st.session_state.get("dark_mode"):
    st.markdown(
        """
        <style>
        .stApp { background-color: #111827; color: #e5e7eb; }
        .stMarkdown, .stText, .stDataFrame, .stSelectbox, .stButton { color: #e5e7eb !important; }
        .stButton>button { background-color: #374151; color: #e5e7eb; border: 1px solid #4b5563; }
        .stSelectbox>div>div>select { background-color: #1f2937; color: #e5e7eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title(T["title"])
st.write(T["upload"])

uploaded_file = st.file_uploader(T["upload_csv"], type=["csv"], help="CSV only; small files recommended.")


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

    if st.button(T["analyze"], type="primary"):
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

            # Emoji feedback beside fairness score
            if fairness_score is not None:
                if fairness_score >= 80:
                    status_emoji = "🟢"
                elif fairness_score >= 60:
                    status_emoji = "🟡"
                else:
                    status_emoji = "🔴"
                st.success(f"{status_emoji} Fairness Score: {fairness_score:.2f} / 100")
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
            # Include additional metrics in table view
            st.write({k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})

            # Call AI explanation endpoint
            try:
                explain_payload = {"metrics": metrics, "sensitive_feature": sensitive_feature}
                ex_res = requests.post(f"{BACKEND_URL}/explain", json=explain_payload, timeout=30)
                if ex_res.status_code == 200:
                    explanation_text = ex_res.json().get("explanation", "")
                    st.subheader(T["ai_expl"])
                    st.write(explanation_text)
                else:
                    st.info("AI explanation unavailable (configure HUGGINGFACE_API_TOKEN or OLLAMA_URL).")
            except Exception:
                st.info("AI explanation unavailable (configure HUGGINGFACE_API_TOKEN or OLLAMA_URL).")

            # Feature influence (Explainability)
            explainability = result.get("explainability")
            if explainability and explainability.get("feature_importances"):
                st.expander(T["feat_influence"], expanded=False).write("")
                imp = explainability["feature_importances"]
                imp_df = pd.DataFrame(imp)
                try:
                    fig_imp = px.bar(imp_df.head(20), x="feature", y="importance", title=T["feat_influence"]) 
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception:
                    st.dataframe(imp_df.head(20))
            elif explainability:
                st.info(explainability.get("reason", "Explainability not available."))

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