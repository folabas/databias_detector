from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
)


def detect_sensitive_features(df: pd.DataFrame) -> List[str]:
    """Detect likely sensitive feature columns using intelligent analysis.

    Strategy:
    1. Analyze all columns for categorical characteristics
    2. Prioritize low-cardinality categorical columns (2-10 unique values)
    3. Consider string/object columns that might represent groups
    4. Exclude obviously non-sensitive columns (IDs, dates, etc.)
    """
    detected = []
    
    for col in df.columns:
        # Skip obviously non-sensitive columns
        if _is_likely_non_sensitive(col, df[col]):
            continue
            
        # Check if column has categorical characteristics suitable for bias analysis
        if _is_potential_sensitive_feature(df[col]):
            detected.append(col)
    
    # Sort by suitability score (lower cardinality = better for bias analysis)
    detected.sort(key=lambda col: _get_sensitivity_score(df[col]))
    
    return detected


def _is_likely_non_sensitive(col_name: str, series: pd.Series) -> bool:
    """Check if a column is likely NOT a sensitive feature."""
    col_lower = col_name.lower()
    
    # Skip ID columns, timestamps, URLs, etc.
    non_sensitive_patterns = [
        'id', 'uuid', 'key', 'index', 'timestamp', 'date', 'time',
        'url', 'link', 'email', 'phone', 'address', 'zip', 'postal',
        'latitude', 'longitude', 'lat', 'lng', 'coordinate'
    ]
    
    for pattern in non_sensitive_patterns:
        if pattern in col_lower:
            return True
    
    # Skip high-cardinality numeric columns (likely continuous variables)
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique()
        total_count = len(series.dropna())
        if total_count > 0 and unique_count / total_count > 0.8:  # >80% unique values
            return True
    
    return False


def _is_potential_sensitive_feature(series: pd.Series) -> bool:
    """Check if a column could be a sensitive feature for bias analysis."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return False
    
    unique_count = series_clean.nunique()
    
    # Good candidates: 2-10 unique values (categorical groups)
    if 2 <= unique_count <= 10:
        return True
    
    # For string/object columns, be more lenient (up to 15 categories)
    if series.dtype == 'object' and 2 <= unique_count <= 15:
        return True
    
    return False


def _get_sensitivity_score(series: pd.Series) -> float:
    """Calculate a score for how suitable a column is as a sensitive feature.
    Lower score = better candidate."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return float('inf')
    
    unique_count = series_clean.nunique()
    
    # Prefer binary features (score = 1), then low cardinality
    if unique_count == 2:
        return 1.0
    elif unique_count <= 5:
        return 2.0 + (unique_count - 2) * 0.1
    elif unique_count <= 10:
        return 3.0 + (unique_count - 5) * 0.2
    else:
        return 10.0 + unique_count


def detect_binary_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are binary or can be easily converted to binary."""
    binary_cols = []
    
    for col in df.columns:
        if _is_binary_column(df[col]):
            binary_cols.append(col)
    
    # Sort by binary confidence score
    binary_cols.sort(key=lambda col: _get_binary_confidence_score(df[col]), reverse=True)
    
    return binary_cols


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """Detect columns that could serve as target/outcome variables."""
    candidates = []
    
    for col in df.columns:
        if _is_potential_target(col, df[col]):
            candidates.append(col)
    
    # Sort by target suitability (binary columns first, then by name patterns)
    candidates.sort(key=lambda col: _get_target_suitability_score(col, df[col]))
    
    return candidates


def _is_binary_column(series: pd.Series) -> bool:
    """Check if a column is binary or easily convertible to binary."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return False
    
    unique_count = series_clean.nunique()
    
    # Exactly 2 unique values = binary
    if unique_count == 2:
        return True
    
    # Single unique value can be treated as binary (all 0s or all 1s)
    if unique_count == 1:
        return True
    
    return False


def _is_potential_target(col_name: str, series: pd.Series) -> bool:
    """Check if a column could be a target/outcome variable."""
    col_lower = col_name.lower()
    
    # Common target column name patterns
    target_patterns = [
        'outcome', 'target', 'label', 'result', 'decision', 'approved', 'accepted',
        'hired', 'admitted', 'selected', 'success', 'fail', 'pass', 'win', 'loss',
        'positive', 'negative', 'class', 'category', 'prediction', 'income',
        'salary', 'wage', 'loan', 'credit', 'default', 'churn', 'fraud'
    ]
    
    # Check name patterns
    for pattern in target_patterns:
        if pattern in col_lower:
            return True
    
    # Check if it's binary (good for classification targets)
    if _is_binary_column(series):
        return True
    
    # Check if it's low-cardinality categorical (good for classification)
    if series.dtype == 'object':
        unique_count = series.nunique()
        if 2 <= unique_count <= 5:
            return True
    
    return False


def _get_binary_confidence_score(series: pd.Series) -> float:
    """Calculate confidence that a column is truly binary. Higher = more confident."""
    series_clean = series.dropna()
    if len(series_clean) == 0:
        return 0.0
    
    unique_count = series_clean.nunique()
    
    if unique_count == 2:
        # Check if values look like binary (0/1, True/False, Yes/No, etc.)
        unique_vals = set(series_clean.unique())
        
        # Perfect binary patterns
        binary_patterns = [
            {0, 1}, {'0', '1'}, {True, False}, {'true', 'false'},
            {'yes', 'no'}, {'y', 'n'}, {'male', 'female'}, {'m', 'f'},
            {'positive', 'negative'}, {'pass', 'fail'}, {'approved', 'rejected'}
        ]
        
        # Convert to lowercase strings for comparison
        vals_lower = {str(v).lower() for v in unique_vals}
        
        for pattern in binary_patterns:
            pattern_lower = {str(v).lower() for v in pattern}
            if vals_lower == pattern_lower:
                return 1.0
        
        # Any 2 unique values are still binary
        return 0.8
    
    elif unique_count == 1:
        return 0.5  # Single value can be binary but less ideal
    
    return 0.0


def _get_target_suitability_score(col_name: str, series: pd.Series) -> float:
    """Calculate target suitability score. Lower = better candidate."""
    col_lower = col_name.lower()
    score = 10.0  # Base score
    
    # Strong target name patterns (lower score = better)
    strong_patterns = ['outcome', 'target', 'label', 'result', 'income']
    medium_patterns = ['decision', 'approved', 'hired', 'class', 'category']
    
    for pattern in strong_patterns:
        if pattern in col_lower:
            score -= 5.0
            break
    
    for pattern in medium_patterns:
        if pattern in col_lower:
            score -= 3.0
            break
    
    # Binary columns are preferred targets
    if _is_binary_column(series):
        score -= 2.0
    
    # Low cardinality categorical is good
    unique_count = series.nunique()
    if 2 <= unique_count <= 5:
        score -= 1.0
    
    return score


def analyze_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the structure of a dataset to provide intelligent suggestions."""
    return {
        "total_columns": len(df.columns),
        "total_rows": len(df),
        "sensitive_candidates": detect_sensitive_features(df),
        "binary_columns": detect_binary_columns(df),
        "target_candidates": detect_target_candidates(df),
        "column_info": {
            col: {
                "dtype": str(df[col].dtype),
                "unique_count": df[col].nunique(),
                "null_count": df[col].isnull().sum(),
                "is_binary": _is_binary_column(df[col]),
                "is_sensitive_candidate": col in detect_sensitive_features(df),
                "is_target_candidate": col in detect_target_candidates(df)
            }
            for col in df.columns
        }
    }


def _ensure_binary_series(s: pd.Series) -> Optional[pd.Series]:
    """Attempt to coerce a series to binary values {0,1}.

    Returns coerced series or None if not possible.
    """
    s_no_na = s.dropna()
    unique_vals = pd.unique(s_no_na)
    if len(unique_vals) <= 2:
        # Map truthy/strings like "yes"/"true" to 1, others to 0
        mapping = {}
        # Build mapping deterministically
        vals_sorted = sorted(list(unique_vals), key=lambda x: str(x))
        if len(vals_sorted) == 2:
            mapping[vals_sorted[0]] = 0
            mapping[vals_sorted[1]] = 1
        else:
            # Single unique value -> all zeros
            mapping[vals_sorted[0]] = 0
        return s.map(lambda x: mapping.get(x, 0))
    return None


def compute_demographic_parity(
    df: pd.DataFrame,
    sensitive_feature: str,
    target_col: str,
) -> Optional[float]:
    """Compute demographic parity difference using Fairlearn.

    Requires a binary target column. If target is not binary, attempts coercion.
    """
    if sensitive_feature not in df.columns or target_col not in df.columns:
        return None

    y = df[target_col]
    y_bin = _ensure_binary_series(y)
    if y_bin is None:
        return None

    # DPD signature needs y_true, y_pred; DPD does not depend on y_true, but
    # we pass the same binary series for both to satisfy signature.
    sens = df[sensitive_feature]
    try:
        dpd = demographic_parity_difference(y_true=y_bin, y_pred=y_bin, sensitive_features=sens)
        return float(dpd)
    except Exception:
        return None


def compute_equal_opportunity(
    df: pd.DataFrame,
    sensitive_feature: str,
    y_true_col: str,
    y_pred_col: Optional[str] = None,
) -> Optional[float]:
    """Compute equal opportunity difference.

    If y_pred_col is None, attempts to use y_true as a naive proxy; returns None
    if binary coercion fails.
    """
    if sensitive_feature not in df.columns or y_true_col not in df.columns:
        return None

    y_true = df[y_true_col]
    y_true_bin = _ensure_binary_series(y_true)
    if y_true_bin is None:
        return None

    if y_pred_col and y_pred_col in df.columns:
        y_pred = df[y_pred_col]
        y_pred_bin = _ensure_binary_series(y_pred)
        if y_pred_bin is None:
            return None
    else:
        # Fallback proxy (not ideal, but allows MVP operation)
        y_pred_bin = y_true_bin

    sens = df[sensitive_feature]
    try:
        eod = equal_opportunity_difference(y_true=y_true_bin, y_pred=y_pred_bin, sensitive_features=sens)
        return float(eod)
    except Exception:
        return None


def aggregate_fairness_score(metrics: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Aggregate available metric differences into a 0–100 fairness score.

    Scoring: for each metric difference `d` (ideal is 0), score_i = max(0, 100 - min(100, abs(d) * 100)).
    Final score is the average over available metrics.
    """
    diffs = [v for v in metrics.values() if v is not None]
    if not diffs:
        return {"fairness_score": None, "component_scores": {}, "available_metrics": []}

    comp_scores = {}
    for name, val in metrics.items():
        if val is None:
            continue
        comp_scores[name] = max(0.0, 100.0 - min(100.0, abs(val) * 100.0))

    fairness = float(np.mean(list(comp_scores.values()))) if comp_scores else None
    return {
        "fairness_score": fairness,
        "component_scores": comp_scores,
        "available_metrics": list(comp_scores.keys()),
    }


def analyze_bias(
    df: pd.DataFrame,
    sensitive_feature: Optional[str] = None,
    target_col: Optional[str] = None,
    predictions_col: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level bias analysis entrypoint.

    - Auto-detect sensitive feature if not provided
    - Compute DPD and EOD where possible
    - Return breakdown and fairness score
    """
    if sensitive_feature is None:
        candidates = detect_sensitive_features(df)
        sensitive_feature = candidates[0] if candidates else None

    if not sensitive_feature:
        return {
            "error": "No sensitive feature detected. Provide one or include columns like 'gender', 'region', 'age'.",
        }

    # Prefer a reasonable default for target
    if target_col is None:
        # Common names
        for c in ["outcome", "label", "target", "y", "approved", "hired"]:
            if c in df.columns:
                target_col = c
                break

    if target_col is None:
        return {
            "error": "No target/outcome column found. Provide one (e.g., 'outcome').",
        }

    dpd = compute_demographic_parity(df, sensitive_feature, target_col)
    eod = compute_equal_opportunity(df, sensitive_feature, target_col, predictions_col)

    metrics = {
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eod,
    }
    agg = aggregate_fairness_score(metrics)

    result = {
        "sensitive_feature": sensitive_feature,
        "target": target_col,
        "metrics": metrics,
        "fairness_score": agg.get("fairness_score"),
        "component_scores": agg.get("component_scores"),
        "available_metrics": agg.get("available_metrics"),
    }
    return result