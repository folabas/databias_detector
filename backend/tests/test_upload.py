"""Placeholder tests for /upload endpoint.

Use TestClient to check response structure.
"""
import io
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_upload_endpoint_returns_keys():
    csv = io.BytesIO(b"gender,accepted\nM,1\nF,0\n")
    files = {"file": ("tiny.csv", csv.getvalue(), "text/csv")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    for k in ["detected_sensitive","binary_columns","target_candidates","dataset_analysis"]:
        assert k in data