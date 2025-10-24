"""FastAPI entrypoint exposing routes only.

Routes delegate to modular core components for analysis, explainability, and utilities.
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .core.schema import ColumnsResponse, AnalyzeResponse, ExplainRequest, ExplainResponse
from .core.utils import detect_binary_columns, detect_target_candidates, analyze_dataset_structure
from .core.analysis import analyze_bias, suggest_bias_corrections
from .core.explainability import explain_feature_influence
from .core.llm_explainer import generate_bias_explanation

app = FastAPI(title="DataBias Detector API")

# CORS for local Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "DataBias Detector API", "routes": ["/upload", "/analyze", "/explain"]}

@app.post("/upload", response_model=ColumnsResponse)
async def upload(file: UploadFile = File(...)):
    csv_bytes = await file.read()
    import pandas as pd
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))

    # Intelligent analysis
    binary_cols = detect_binary_columns(df)
    targets = detect_target_candidates(df)
    dataset_info = analyze_dataset_structure(df)

    # Sensitive features heuristic (simple keywords)
    sensitive_keywords = ["gender", "sex", "race", "ethnicity", "age", "marital", "religion", "nationality"]
    detected_sensitive = [c for c in df.columns if any(k in c.lower() for k in sensitive_keywords)]

    return ColumnsResponse(
        detected_sensitive=detected_sensitive,
        binary_columns=binary_cols,
        target_candidates=targets,
        dataset_analysis=dataset_info,
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    sensitive_feature: str = Form(...),
    target: str = Form(...),
    predictions_col: Optional[str] = Form(None),
):
    csv_bytes = await file.read()
    result = analyze_bias(csv_bytes, sensitive_feature, target, predictions_col)

    if result.get("error"):
        return AnalyzeResponse(
            fairness_score=None,
            metrics={},
            component_scores={},
            error=result["error"],
        )

    # Include explainability and suggestions
    import pandas as pd
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    explainability = explain_feature_influence(df, target, predictions_col)
    suggestions = suggest_bias_corrections(df, sensitive_feature, target, predictions_col, result.get("metrics", {}))

    return AnalyzeResponse(
        fairness_score=result["fairness_score"],
        metrics=result["metrics"],
        component_scores=result["component_scores"],
        explainability=explainability,
        suggestions=suggestions,
    )

@app.post("/explain", response_model=ExplainResponse)
async def explain(payload: ExplainRequest):
    text = generate_bias_explanation(metrics=payload.metrics, sensitive_feature=payload.sensitive_feature)
    return ExplainResponse(explanation=text)