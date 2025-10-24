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


def compute_statistical_parity_difference(
    df: pd.DataFrame,
    sensitive_feature: str,
    target_col: str,
) -> Optional[float]:
    """Statistical Parity Difference: max(group positive rate) - min(group positive rate).
    Requires binary-coercible target.
    """
    if sensitive_feature not in df.columns or target_col not in df.columns:
        return None
    y_bin = _ensure_binary_series(df[target_col])
    if y_bin is None:
        return None
    try:
        rates = df.assign(_y=y_bin).groupby(sensitive_feature)['_y'].mean(numeric_only=True)
        if len(rates) < 2:
            return None
        return float(rates.max() - rates.min())
    except Exception:
        return None


def compute_disparate_impact_ratio(
    df: pd.DataFrame,
    sensitive_feature: str,
    target_col: str,
) -> Optional[float]:
    """Disparate Impact Ratio: min(group positive rate) / max(group positive rate).
    Ideal ≈ 1.0. Returns None if cannot compute.
    """
    if sensitive_feature not in df.columns or target_col not in df.columns:
        return None
    y_bin = _ensure_binary_series(df[target_col])
    if y_bin is None:
        return None
    try:
        rates = df.assign(_y=y_bin).groupby(sensitive_feature)['_y'].mean(numeric_only=True)
        if len(rates) < 2:
            return None
        mx = float(rates.max())
        mn = float(rates.min())
        if mx == 0:
            return None
        return float(mn / mx)
    except Exception:
        return None


def compute_predictive_equality_difference(
    df: pd.DataFrame,
    sensitive_feature: str,
    y_true_col: str,
    y_pred_col: Optional[str] = None,
) -> Optional[float]:
    """Predictive Equality Difference: difference in false positive rates across groups.
    Requires binary y_true and binary/coercible y_pred.
    If y_pred_col is None, returns None.
    """
    if sensitive_feature not in df.columns or y_true_col not in df.columns:
        return None
    if not y_pred_col or y_pred_col not in df.columns:
        return None
    y_true_bin = _ensure_binary_series(df[y_true_col])
    y_pred_bin = _ensure_binary_series(df[y_pred_col])
    if y_true_bin is None or y_pred_bin is None:
        return None
    try:
        data = df.assign(_yt=y_true_bin, _yp=y_pred_bin)
        # False positives: yp=1 while yt=0
        def fpr(group):
            g = group
            negatives = (g['_yt'] == 0).sum()
            if negatives == 0:
                return 0.0
            fp = ((g['_yp'] == 1) & (g['_yt'] == 0)).sum()
            return float(fp) / float(negatives)
        rates = data.groupby(sensitive_feature).apply(fpr)
        if len(rates) < 2:
            return None
        return float(rates.max() - rates.min())
    except Exception:
        return None


def aggregate_fairness_score(metrics: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Aggregate available metric differences into a 0–100 fairness score.

    Scoring: for each metric difference `d` (ideal is 0), score_i = max(0, 100 - min(100, abs(d) * 100)).
    For ratio metrics like Disparate Impact Ratio (ideal is 1), use d = abs(1 - r).
    """
    diffs = []
    comp_scores = {}
    for name, val in metrics.items():
        if val is None:
            continue
        if name == 'disparate_impact_ratio':
            d = abs(1.0 - float(val))
        else:
            d = abs(float(val))
        diffs.append(d)
        comp_scores[name] = max(0.0, 100.0 - min(100.0, d * 100.0))

    if not diffs:
        return {"fairness_score": None, "component_scores": {}, "available_metrics": []}

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
    spd = compute_statistical_parity_difference(df, sensitive_feature, target_col)
    dirv = compute_disparate_impact_ratio(df, sensitive_feature, target_col)
    ped = compute_predictive_equality_difference(df, sensitive_feature, target_col, predictions_col)

    metrics = {
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eod,
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dirv,
        "predictive_equality_difference": ped,
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


def generate_bias_explanation(metrics: Dict[str, Any], sensitive_feature: str) -> str:
    """Generate a plain-English bias explanation using a free LLM API.
    - Uses Hugging Face Inference API when HUGGINGFACE_API_TOKEN is set.
    - Optionally supports Ollama via OLLAMA_URL if provided.
    - Gracefully falls back to a templated summary if no API is configured.
    """
    import os, json, requests
    dpd = metrics.get('demographic_parity_difference')
    eod = metrics.get('equal_opportunity_difference')
    spd = metrics.get('statistical_parity_difference')
    dirv = metrics.get('disparate_impact_ratio')
    ped = metrics.get('predictive_equality_difference')

    base_summary = (
        f"Analysis across '{sensitive_feature}' groups: "
        + (f"DPD={dpd:.3f}. " if isinstance(dpd, (int, float)) else "")
        + (f"SPD={spd:.3f}. " if isinstance(spd, (int, float)) else "")
        + (f"DIR={dirv:.3f} (ideal≈1). " if isinstance(dirv, (int, float)) else "")
        + (f"EOD={eod:.3f}. " if isinstance(eod, (int, float)) else "")
        + (f"PED={ped:.3f}. " if isinstance(ped, (int, float)) else "")
    )
    prompt = (
        "You are a helpful data fairness assistant. Given these metric values, "
        "explain in one short paragraph what they indicate about potential bias, "
        "using plain language and percentages where helpful.\n\n"
        f"Sensitive feature: {sensitive_feature}\n"
        f"Metrics JSON: {json.dumps({k: (float(v) if v is not None else None) for k,v in metrics.items()})}\n"
        "Explain the main takeaway and suggest a simple next step."
    )

    # Try Hugging Face Inference API
    hf_token = os.environ.get('HUGGINGFACE_API_TOKEN')
    hf_model = os.environ.get('HUGGINGFACE_MODEL', 'google/flan-t5-large')
    if hf_token:
        try:
            headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
            payload = {"inputs": prompt}
            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    text = data[0].get('generated_text') or data[0].get('summary_text')
                else:
                    text = data.get('generated_text') or str(data)
                if text:
                    return text.strip()
        except Exception:
            pass

    # Try Ollama (local) if available
    ollama_url = os.environ.get('OLLAMA_URL')  # e.g., http://localhost:11434/api/generate
    ollama_model = os.environ.get('OLLAMA_MODEL', 'mistral')
    if ollama_url:
        try:
            payload = {"model": ollama_model, "prompt": prompt}
            resp = requests.post(ollama_url, json=payload, timeout=30)
            if resp.status_code == 200:
                # Ollama streams; try to assemble
                text = resp.text
                if text:
                    return text.strip()
        except Exception:
            pass

    # Fallback templated explanation
    parts = []
    if isinstance(dpd, (int, float)):
        parts.append(f"Demographic parity difference of {dpd:.2f} suggests outcome rates differ by {abs(dpd)*100:.1f}% across groups.")
    if isinstance(spd, (int, float)):
        parts.append(f"Statistical parity difference of {spd:.2f} indicates selection rate gap of {abs(spd)*100:.1f}%.")
    if isinstance(dirv, (int, float)):
        parts.append(f"Disparate impact ratio of {dirv:.2f} (ideal≈1) implies relative selection rates differ.")
    if isinstance(eod, (int, float)):
        parts.append(f"Equal opportunity difference of {eod:.2f} suggests true positive rates differ.")
    if isinstance(ped, (int, float)):
        parts.append(f"Predictive equality difference of {ped:.2f} indicates false positive rates vary.")
    if not parts:
        parts.append("Insufficient metrics to assess bias; provide a binary target and optionally predictions.")
    parts.append("Consider reviewing group-level rates, balancing datasets, or adjusting decision thresholds.")
    return " ".join(parts)


def explain_feature_influence(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Compute simple SHAP-based feature influence for outcomes.
    - Tries to use scikit-learn for a lightweight model and SHAP for importances.
    - Gracefully degrades to correlation-based proxy if SHAP or sklearn missing.
    Returns a dict with feature_importances list of {feature, importance}.
    """
    try:
        import shap  # type: ignore
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except Exception:
        # Fallback: correlation-based proxy importance (numeric only)
        try:
            y = _ensure_binary_series(df[target_col])
            if y is None:
                return {"feature_importances": [], "explanation_available": False, "reason": "Target not binary."}
            X = df.drop(columns=[target_col])
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            imps = []
            for c in num_cols:
                try:
                    corr = abs(np.corrcoef(X[c].fillna(0), y.fillna(0))[0,1])
                    imps.append({"feature": c, "importance": float(corr)})
                except Exception:
                    continue
            imps.sort(key=lambda x: x['importance'], reverse=True)
            return {"feature_importances": imps[:20], "explanation_available": False, "reason": "SHAP/sklearn not available; used correlation proxy."}
        except Exception:
            return {"feature_importances": [], "explanation_available": False, "reason": "Explainability failed."}

    # Prepare data
    y = _ensure_binary_series(df[target_col])
    if y is None:
        return {"feature_importances": [], "explanation_available": False, "reason": "Target not binary."}
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X = X.fillna(0)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y.fillna(0), test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classification, shap_values is list; take class 1
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = shap_values[1]
        else:
            sv = shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        imps = [{"feature": f, "importance": float(mean_abs[i])} for i, f in enumerate(X.columns)]
        imps.sort(key=lambda x: x['importance'], reverse=True)
        return {"feature_importances": imps[:20], "explanation_available": True}
    except Exception:
        return {"feature_importances": [], "explanation_available": False, "reason": "SHAP computation failed."}