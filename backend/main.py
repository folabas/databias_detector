from typing import Optional

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .bias_analysis import (
    analyze_bias, 
    detect_sensitive_features, 
    detect_binary_columns, 
    detect_target_candidates,
    analyze_dataset_structure,
    generate_bias_explanation,
    explain_feature_influence,
)

app = FastAPI(title="DataBias Detector API", version="0.1.0")

# Allow local dev from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ColumnsResponse(BaseModel):
    columns: list[str]
    detected_sensitive: list[str]
    binary_columns: list[str]
    target_candidates: list[str]
    dataset_analysis: dict


class AnalyzeResponse(BaseModel):
    sensitive_feature: Optional[str]
    target: Optional[str]
    metrics: dict
    fairness_score: Optional[float]
    component_scores: dict
    available_metrics: list[str]
    error: Optional[str] = None
    # Optional extras
    explainability: Optional[dict] = None


class ExplainRequest(BaseModel):
    metrics: dict
    sensitive_feature: str


class ExplainResponse(BaseModel):
    explanation: str


@app.get("/")
def root():
    return {"message": "DataBias Detector API running!"}


@app.post("/upload", response_model=ColumnsResponse)
async def upload_file(file: UploadFile = File(...)):
    # Stream CSV into DataFrame without loading entire bytes into memory
    file.file.seek(0)
    df = pd.read_csv(file.file)
    
    # Perform comprehensive dataset analysis
    columns = df.columns.tolist()
    detected_sensitive = detect_sensitive_features(df)
    binary_columns = detect_binary_columns(df)
    target_candidates = detect_target_candidates(df)
    dataset_analysis = analyze_dataset_structure(df)
    
    return {
        "columns": columns, 
        "detected_sensitive": detected_sensitive,
        "binary_columns": binary_columns,
        "target_candidates": target_candidates,
        "dataset_analysis": dataset_analysis
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_bias_endpoint(
    file: UploadFile = File(...),
    sensitive_feature: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
    predictions_col: Optional[str] = Form(None),
):
    # Stream CSV into DataFrame
    file.file.seek(0)
    df = pd.read_csv(file.file)

    result = analyze_bias(
        df=df,
        sensitive_feature=sensitive_feature,
        target_col=target,
        predictions_col=predictions_col,
    )

    # Add friendly errors
    if "error" in result:
        return AnalyzeResponse(
            sensitive_feature=sensitive_feature,
            target=target,
            metrics={},
            fairness_score=None,
            component_scores={},
            available_metrics=[],
            error=result["error"],
        )

    # Include explainability (feature influence) when possible
    explainability = None
    if target and target in df.columns:
        try:
            explainability = explain_feature_influence(df, target)
        except Exception:
            explainability = {"feature_importances": [], "explanation_available": False}

    return AnalyzeResponse(**result, explainability=explainability)

@app.post("/explain", response_model=ExplainResponse)
async def explain_bias(req: ExplainRequest):
    text = generate_bias_explanation(req.metrics, req.sensitive_feature)
    return ExplainResponse(explanation=text)