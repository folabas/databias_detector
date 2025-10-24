"""Placeholder tests for /explain endpoint.

Ensures the endpoint responds with an explanation string.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_explain_endpoint_text():
    payload = {"metrics": {"statistical_parity_difference": 0.2, "disparate_impact_ratio": 0.85, "predictive_equality_difference": 0.1}, "sensitive_feature": "gender"}
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "explanation" in data