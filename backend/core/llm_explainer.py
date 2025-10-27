"""LLM-based explanations for fairness metrics.

Integrates with Hugging Face or Ollama when configured, with a templated fallback.
"""
from typing import Dict, Optional
import json
import requests
from .config import settings


def _template_explanation(metrics: Dict[str, float], sensitive_feature: Optional[str]) -> str:
    spd = metrics.get("statistical_parity_difference")
    diratio = metrics.get("disparate_impact_ratio")
    ped = metrics.get("predictive_equality_difference")

    def fmt(v: Optional[float]) -> str:
        try:
            if v is None:
                return "N/A"
            return f"{float(v):.3f}"
        except Exception:
            return "N/A"

    group_text = f" across groups in '{sensitive_feature}'" if sensitive_feature else " across groups"
    return (
        "This dataset shows fairness metrics" + group_text + ". "
        f"Statistical Parity Difference: {fmt(spd)}, "
        f"Disparate Impact Ratio: {fmt(diratio)}, "
        f"Predictive Equality Difference: {fmt(ped)}. "
        "Values near 0 (for differences) and near 1 (for ratios) indicate improved fairness."
    )


def generate_bias_explanation(metrics: Dict[str, float], sensitive_feature: Optional[str] = None) -> str:
    """Generate a plain-English explanation using configured LLM provider or fallback."""
    # Prefer Hugging Face
    if settings.HUGGINGFACE_API_TOKEN:
        try:
            # A lean, free-friendly endpoint; model can be switched via env if desired.
            model_url = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
            headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_TOKEN}"}
            prompt = (
                "Explain these fairness metrics in simple terms and give one actionable tip.\n"
                f"Metrics: {json.dumps(metrics)}\nSensitive feature: {sensitive_feature or 'N/A'}\n"
            )
            payload = {"inputs": prompt}
            resp = requests.post(model_url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                # HF inference responses can be diverse; handle text list or dict
                if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, list) and data and isinstance(data[0], str):
                    return data[0]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
        except Exception:
            pass
    
    if settings.OLLAMA_BASE_URL:
        try:
            prompt = (
                "Explain these fairness metrics in simple terms and give one actionable tip.\n"
                f"Metrics: {json.dumps(metrics)}\nSensitive feature: {sensitive_feature or 'N/A'}\n"
            )
            resp = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": "llama3", "prompt": prompt},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "response" in data:
                    return data["response"]
        except Exception:
            pass
    return _template_explanation(metrics, sensitive_feature)