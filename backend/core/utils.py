"""Utility functions for dataset inspection and preprocessing.

Includes detection of binary columns, target candidates, and general dataset structure.
"""
from typing import List, Dict, Any
import pandas as pd


def detect_binary_columns(df: pd.DataFrame, max_unique: int = 2) -> List[str]:
    """Return columns with at most `max_unique` unique non-null values."""
    binary_cols: List[str] = []
    for col in df.columns:
        uniq = df[col].dropna().nunique()
        if uniq <= max_unique:
            binary_cols.append(col)
    return binary_cols


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """Suggest candidate target/outcome columns (favor binary columns)."""
    candidates = []
    for col in df.columns:
        uniq = df[col].dropna().nunique()
        if uniq <= 2:
            candidates.append(col)
    # heuristic: prioritize common outcome-like names
    preferred = [c for c in candidates if any(k in c.lower() for k in ["target", "label", "outcome", "result", "accepted", "approved", "hired", "churn"])]
    return preferred + [c for c in candidates if c not in preferred]


def analyze_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize dataset shape and simple stats used by the UI."""
    try:
        return {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "column_types": {c: str(df[c].dtype) for c in df.columns},
        }
    except Exception:
        return {"total_rows": None, "total_columns": None, "column_types": {}}