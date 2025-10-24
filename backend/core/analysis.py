"""Fairness metrics and bias calculations.

Implements core fairness metrics and aggregate scoring, plus dataset bias analysis.
"""
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

# --- Fairness metrics ---

def compute_statistical_parity_difference(df: pd.DataFrame, sensitive: str, target: str) -> float:
    """Difference in positive outcome rates between groups."""
    groups = df[sensitive].dropna().unique()
    if len(groups) < 2:
        return 0.0
    rates = []
    for g in groups:
        sub = df[df[sensitive] == g]
        if sub.empty:
            continue
        pos_rate = sub[target].mean() if pd.api.types.is_numeric_dtype(df[target]) else (sub[target] == 1).mean()
        rates.append(pos_rate)
    if not rates:
        return 0.0
    return float(np.max(rates) - np.min(rates))

def compute_disparate_impact_ratio(df: pd.DataFrame, sensitive: str, target: str) -> float:
    """Ratio of positive rates (min/max)."""
    groups = df[sensitive].dropna().unique()
    if len(groups) < 2:
        return 1.0
    rates = []
    for g in groups:
        sub = df[df[sensitive] == g]
        if sub.empty:
            continue
        pos_rate = sub[target].mean() if pd.api.types.is_numeric_dtype(df[target]) else (sub[target] == 1).mean()
        rates.append(pos_rate if pos_rate > 0 else 1e-9)
    if not rates:
        return 1.0
    return float(np.min(rates) / np.max(rates))

def compute_predictive_equality_difference(df: pd.DataFrame, sensitive: str, target: str, predictions_col: Optional[str] = None) -> float:
    """Difference in false positive rates across groups.

    If predictions are absent, uses target as proxy.
    """
    pred = predictions_col or target
    if pred not in df.columns:
        return 0.0
    groups = df[sensitive].dropna().unique()
    if len(groups) < 2:
        return 0.0
    fprates = []
    for g in groups:
        sub = df[df[sensitive] == g]
        if sub.empty:
            continue
        # false positive: pred==1 while true label==0
        if pd.api.types.is_numeric_dtype(df[pred]) and pd.api.types.is_numeric_dtype(df[target]):
            fp = ((sub[pred] >= 0.5) & (sub[target] == 0)).sum()
            negatives = (sub[target] == 0).sum()
        else:
            fp = ((sub[pred] == 1) & (sub[target] == 0)).sum()
            negatives = (sub[target] == 0).sum()
        rate = fp / negatives if negatives > 0 else 0.0
        fprates.append(rate)
    if not fprates:
        return 0.0
    return float(np.max(fprates) - np.min(fprates))

# --- Aggregate scoring ---

def aggregate_fairness_score(spd: float, diratio: float, ped: float) -> Tuple[float, Dict[str, float]]:
    """Aggregate into a 0–100 fairness score with component weights."""
    # Normalize components: lower is better for SPD/PED, higher is better for DIR
    comp_spd = max(0.0, 1.0 - min(1.0, abs(spd)))
    comp_dir = max(0.0, min(1.0, diratio))
    comp_ped = max(0.0, 1.0 - min(1.0, abs(ped)))
    # Weighted average
    weights = {"SPD": 0.4, "DIR": 0.3, "PED": 0.3}
    score = 100.0 * (
        comp_spd * weights["SPD"] + comp_dir * weights["DIR"] + comp_ped * weights["PED"]
    )
    return score, {"SPD": comp_spd, "DIR": comp_dir, "PED": comp_ped}

# --- Analysis orchestrator ---

def analyze_bias(csv_bytes: bytes, sensitive_feature: str, target: str, predictions_col: Optional[str] = None) -> Dict[str, object]:
    """Run fairness analysis on a CSV payload.

    Returns metrics and aggregate fairness score. Explainability handled separately.
    """
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    if sensitive_feature not in df.columns or target not in df.columns:
        return {"error": "Selected columns not found in dataset"}
    try:
        spd = compute_statistical_parity_difference(df, sensitive_feature, target)
        diratio = compute_disparate_impact_ratio(df, sensitive_feature, target)
        ped = compute_predictive_equality_difference(df, sensitive_feature, target, predictions_col)
        fairness_score, component_scores = aggregate_fairness_score(spd, diratio, ped)
        metrics = {
            "statistical_parity_difference": spd,
            "disparate_impact_ratio": diratio,
            "predictive_equality_difference": ped,
        }
        return {
            "fairness_score": fairness_score,
            "metrics": metrics,
            "component_scores": component_scores,
        }
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

# --- Correction suggestions ---

def suggest_bias_corrections(
    df: pd.DataFrame,
    sensitive_feature: str,
    target: str,
    predictions_col: Optional[str],
    metrics: Dict[str, float],
) -> List[str]:
    """Heuristic suggestions to reduce bias based on metrics and data profile.

    Produces actionable, model-agnostic steps (resampling, reweighting, thresholding, calibration, data collection).
    """
    suggestions: List[str] = []

    # Group imbalance
    try:
        grp = df[sensitive_feature].value_counts(dropna=True)
        if not grp.empty:
            min_grp = int(grp.min())
            max_grp = int(grp.max())
            if max_grp > 0 and (min_grp / max_grp) < 0.5:
                suggestions.append("Oversample minority groups or undersample majority to balance by sensitive feature.")
                suggestions.append("Use stratified sampling for train/validation/test splits to preserve group ratios.")
    except Exception:
        pass

    # Target imbalance
    try:
        pos = int((df[target] == 1).sum()) if target in df.columns else None
        neg = int((df[target] == 0).sum()) if target in df.columns else None
        if pos is not None and neg is not None:
            if min(pos, neg) / max(pos, neg) < 0.6:
                suggestions.append("Apply class weights or resampling to address target imbalance (e.g., WeightedLoss, SMOTE).")
    except Exception:
        pass

    spd = metrics.get("statistical_parity_difference")
    diratio = metrics.get("disparate_impact_ratio")
    ped = metrics.get("predictive_equality_difference")

    # Statistical parity gap
    if isinstance(spd, (int, float)) and abs(spd) >= 0.1:
        suggestions.append("Reweigh training samples to equalize selection rates across groups (Kamiran & Calders reweighing).")
        suggestions.append("Audit and remove proxy features highly correlated with the sensitive attribute.")
        suggestions.append("Consider group-specific thresholds to equalize selection rates.")

    # Disparate impact
    if isinstance(diratio, (int, float)) and diratio < 0.8:
        suggestions.append("Review decision threshold; raise for advantaged group or lower for disadvantaged group to target DIR≈1.")
        suggestions.append("Calibrate model probabilities (Platt scaling/Isotonic) and re-evaluate group rates.")

    # Predictive equality (false positives)
    if isinstance(ped, (int, float)) and abs(ped) >= 0.05:
        suggestions.append("Tune thresholds per group to reduce false positive disparities (equalized odds post-processing).")
        suggestions.append("Review labeling quality; inconsistent labels across groups can inflate FPR gaps.")

    # Predictions provided? enable more guidance
    if predictions_col and predictions_col in df.columns:
        suggestions.append("If predictions are probabilistic, pick thresholds per group via ROC/PR curves to balance TPR/FPR.")

    # General process improvements
    suggestions.append("Perform subgroup performance auditing and monitor fairness metrics during training (early stopping if fairness degrades).")
    suggestions.append("Collect additional high-quality data for underrepresented groups where feasible.")

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for s in suggestions:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped