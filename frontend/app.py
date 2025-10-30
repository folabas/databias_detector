import requests
import pandas as pd
import streamlit as st
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
 
try:
    from frontend.components.dataset_preview import render_dataset_preview
    from frontend.components.bias_results import render_bias_results
    from frontend.components.ai_explanation import render_ai_explanation
    from frontend.components.feature_importance import render_feature_importance
    from frontend.components.correction_suggestions import render_correction_suggestions
    from frontend.components.group_visualizations import render_group_visualizations
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from components.dataset_preview import render_dataset_preview
    from components.bias_results import render_bias_results
    from components.ai_explanation import render_ai_explanation
    from components.feature_importance import render_feature_importance
    from components.correction_suggestions import render_correction_suggestions
    from components.group_visualizations import render_group_visualizations

 
try:
    BACKEND_URL = st.secrets["backend_url"]
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Configurable wait for delayed backend upload processing
def get_upload_wait_config():
    try:
        max_wait = int(st.secrets.get("upload_max_wait_seconds", 100))
    except Exception:
        max_wait = int(os.environ.get("UPLOAD_MAX_WAIT_SECONDS", "100"))
    try:
        req_timeout = int(st.secrets.get("upload_request_timeout_seconds", max_wait))
    except Exception:
        req_timeout = int(os.environ.get("UPLOAD_REQUEST_TIMEOUT_SECONDS", str(max_wait)))
    return max_wait, req_timeout


def upload_with_delay_handling(uploaded_file, backend_url: str):
    """
    Post the uploaded file to the backend /upload with a delay-tolerant UI.
    - Shows status message and countdown progress while waiting
    - Ends early if backend responds before max wait
    - Logs actual vs expected timing in session_state
    - Errors clearly if backend exceeds max wait
    """
    max_wait, req_timeout = get_upload_wait_config()

    status_ph = st.empty()
    progress_ph = st.progress(0)
    countdown_ph = st.empty()

    start_time = time.time()
    if "upload_timings" not in st.session_state:
        st.session_state["upload_timings"] = []

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            requests.post,
            f"{backend_url}/upload",
            files=files,
            timeout=req_timeout,
        )

        # Poll until done or until max_wait reached
        while True:
            elapsed = time.time() - start_time
            remaining = int(max(0, max_wait - elapsed))
            percentage = int(min(100, (elapsed / max_wait) * 100)) if max_wait > 0 else 100

            status_ph.info("Processing upload… the backend may be delayed.")
            progress_ph.progress(percentage)
            countdown_ph.write(f"Waiting: {remaining}s remaining")

            if future.done():
                try:
                    resp = future.result()
                    resp_json = resp.json()
                    total = time.time() - start_time
                    st.session_state["upload_timings"].append(
                        {
                            "expected_wait": max_wait,
                            "actual_elapsed": round(total, 2),
                            "completed": True,
                            "status_code": resp.status_code,
                        }
                    )
                    status_ph.success(f"Upload processed in {int(total)}s.")
                    progress_ph.progress(100)
                    countdown_ph.write(f"Completed in {int(total)}s (expected {max_wait}s)")
                    return resp_json
                except Exception as e:
                    total = time.time() - start_time
                    st.session_state["upload_timings"].append(
                        {
                            "expected_wait": max_wait,
                            "actual_elapsed": round(total, 2),
                            "completed": False,
                            "error": str(e),
                        }
                    )
                    status_ph.error(f"Upload processing failed: {e}")
                    return None

            if elapsed >= max_wait:
                total = time.time() - start_time
                st.session_state["upload_timings"].append(
                    {
                        "expected_wait": max_wait,
                        "actual_elapsed": round(total, 2),
                        "completed": False,
                        "error": "max_wait_exceeded",
                    }
                )
                status_ph.error(
                    "Upload processing exceeded the maximum wait time. Please try again or increase the wait limit."
                )
                return None

            time.sleep(0.5)


# Configurable wait for delayed backend analyze processing
def get_analyze_wait_config():
    try:
        max_wait = int(st.secrets.get("analyze_max_wait_seconds", 120))
    except Exception:
        max_wait = int(os.environ.get("ANALYZE_MAX_WAIT_SECONDS", "120"))
    try:
        req_timeout = int(st.secrets.get("analyze_request_timeout_seconds", max_wait))
    except Exception:
        req_timeout = int(os.environ.get("ANALYZE_REQUEST_TIMEOUT_SECONDS", str(max_wait)))
    return max_wait, req_timeout


def analyze_with_delay_handling(uploaded_file, data: dict, backend_url: str):
    """
    Post the file and payload to /analyze with delay-tolerant UI similar to upload.
    - Shows status message and countdown progress while waiting
    - Ends early if backend responds before max wait
    - Logs actual vs expected timing in session_state
    - Errors clearly if backend exceeds max wait
    """
    max_wait, req_timeout = get_analyze_wait_config()

    status_ph = st.empty()
    progress_ph = st.progress(0)
    countdown_ph = st.empty()

    start_time = time.time()
    if "analyze_timings" not in st.session_state:
        st.session_state["analyze_timings"] = []

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            requests.post,
            f"{backend_url}/analyze",
            files=files,
            data=data,
            timeout=req_timeout,
        )

        # Poll until done or until max_wait reached
        while True:
            elapsed = time.time() - start_time
            remaining = int(max(0, max_wait - elapsed))
            percentage = int(min(100, (elapsed / max_wait) * 100)) if max_wait > 0 else 100

            status_ph.info("Analyzing… the backend may be delayed.")
            progress_ph.progress(percentage)
            countdown_ph.write(f"Waiting: {remaining}s remaining")

            if future.done():
                try:
                    resp = future.result()
                    resp_json = resp.json()
                    total = time.time() - start_time
                    st.session_state["analyze_timings"].append(
                        {
                            "expected_wait": max_wait,
                            "actual_elapsed": round(total, 2),
                            "completed": True,
                            "status_code": resp.status_code,
                        }
                    )
                    status_ph.success(f"Analysis completed in {int(total)}s.")
                    progress_ph.progress(100)
                    countdown_ph.write(f"Completed in {int(total)}s (expected {max_wait}s)")
                    return resp_json
                except Exception as e:
                    total = time.time() - start_time
                    st.session_state["analyze_timings"].append(
                        {
                            "expected_wait": max_wait,
                            "actual_elapsed": round(total, 2),
                            "completed": False,
                            "error": str(e),
                        }
                    )
                    status_ph.error(f"Analysis failed: {e}")
                    return None

            if elapsed >= max_wait:
                total = time.time() - start_time
                st.session_state["analyze_timings"].append(
                    {
                        "expected_wait": max_wait,
                        "actual_elapsed": round(total, 2),
                        "completed": False,
                        "error": "max_wait_exceeded",
                    }
                )
                status_ph.error(
                    "Analysis exceeded the maximum wait time. Please try again or increase the wait limit."
                )
                return None

            time.sleep(0.5)

st.set_page_config(page_title="DataBias Detector", page_icon="🧮", layout="wide")

st.title("🧮 DataBias Detector")
st.write("Upload your dataset")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"], help="CSV only; small files recommended.")

 

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    resp_json = upload_with_delay_handling(uploaded_file, BACKEND_URL)
    if resp_json:
        detected_sensitive = resp_json.get("detected_sensitive", [])
        binary_columns = resp_json.get("binary_columns", [])
        target_candidates = resp_json.get("target_candidates", [])
        dataset_analysis = resp_json.get("dataset_analysis", {})
    else:
        st.warning("Could not get intelligent analysis from backend within the wait window.")
        detected_sensitive, binary_columns, target_candidates, dataset_analysis = [], [], [], {}

    st.subheader("Data Preview")
    st.dataframe(df, width='stretch')
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
    
    binary_target_candidates = [col for col in target_candidates if col in binary_columns]

    st.markdown("### Column Selection")
    
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
    
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        if binary_columns:
            target_options = binary_columns
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
    
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        predictions_col = st.selectbox(
            "Predictions column (optional)",
            options=["<none>"] + list(df.columns),
            index=0,
            help="Optional: The model's predictions or scores. If provided, should be binary (0/1) or probability scores. If not provided, the target column will be used as a proxy for predictions in the analysis."
        )

    if binary_columns:
        st.info(f"✅ **Binary columns detected:** {', '.join(binary_columns)}")
    else:
        st.warning("⚠️ **No binary columns detected.** Fairness metrics require binary targets. Consider preprocessing your data to create binary columns (e.g., age >= 30 → 1, else 0).")
    
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
        data = {"sensitive_feature": sensitive_feature, "target": target_col}
        if predictions_col and predictions_col != "<none>":
            data["predictions_col"] = predictions_col

        result = analyze_with_delay_handling(uploaded_file, data, BACKEND_URL)
        if result is None:
            st.error("Analyze processing exceeded the maximum wait or failed.")
            st.stop()
        if result.get("error"):
            st.error(result["error"])
        else:
            fairness_score = result.get("fairness_score")
            metrics = result.get("metrics", {})
            comp_scores = result.get("component_scores", {})
            render_bias_results(fairness_score, metrics, comp_scores, df, target_col, (predictions_col if predictions_col != "<none>" else None))
            render_ai_explanation(BACKEND_URL, metrics, sensitive_feature, "AI Explanation") 
            explainability = result.get("explainability")
            expander = st.expander("Feature Influence (Explainability)", expanded=False)
            method = None
            if isinstance(explainability, dict):
                reason = explainability.get("reason")
                if reason and "Correlation" in reason:
                    method = "Correlation-based"
                elif reason is None and explainability.get("feature_importances"):
                    method = "SHAP-based"
                elif reason:
                    method = reason
            if method:
                expander.write(f"Method used: {method}")
            else:
                expander.write("Method used: Explainability unavailable")
            render_feature_importance(explainability, "Feature Influence (Explainability)") 

            viz_expander = st.expander("Rich Visualizations", expanded=True)
            with viz_expander:
                render_group_visualizations(df, sensitive_feature, target_col, (predictions_col if predictions_col != "<none>" else None))

            render_correction_suggestions(result.get("suggestions", []))

            st.download_button(
                label="Download results.json",
                data=json.dumps(result, indent=2),
                file_name="results.json",
                mime="application/json",
            )