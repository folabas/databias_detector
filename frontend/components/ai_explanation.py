"""Streamlit component: AI explanation block.

Calls backend /explain and renders the returned text.
"""
import streamlit as st
import requests

def render_ai_explanation(backend_url: str, metrics: dict, sensitive_feature: str | None, label: str):
    try:
        ex_res = requests.post(f"{backend_url}/explain", json={"metrics": metrics, "sensitive_feature": sensitive_feature}, timeout=30)
        if ex_res.status_code == 200:
            explanation_text = ex_res.json().get("explanation", "")
            st.subheader(label)
            st.write(explanation_text)
        else:
            st.info("AI explanation unavailable (configure HUGGINGFACE_API_TOKEN or OLLAMA_BASE_URL).")
    except Exception:
        st.info("AI explanation unavailable (configure HUGGINGFACE_API_TOKEN or OLLAMA_BASE_URL).")