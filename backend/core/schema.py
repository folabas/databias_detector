"""Pydantic models used by API endpoints.

Defines request/response schemas for dataset analysis, bias metrics, and explanations.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class ColumnsResponse(BaseModel):
    detected_sensitive: List[str]
    binary_columns: List[str]
    target_candidates: List[str]
    dataset_analysis: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    fairness_score: Optional[float]
    metrics: Dict[str, float]
    component_scores: Dict[str, float]
    explainability: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None

class ExplainRequest(BaseModel):
    metrics: Dict[str, float]
    sensitive_feature: Optional[str] = None

class ExplainResponse(BaseModel):
    explanation: str