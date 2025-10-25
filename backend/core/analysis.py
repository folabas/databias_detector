"""Fairness metrics and bias calculations.

Implements core fairness metrics and aggregate scoring, plus dataset bias analysis.
"""
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from .config import settings

# --- Helpers ---

def _ensure_binary_series(s: pd.Series) -> Optional[pd.Series]:
    """Coerce a series with ≤2 unique non-null values to binary {0,1}.

    Deterministic mapping: lexicographically smaller value -> 0, larger -> 1.
    Single unique value -> all zeros. Return None if >2 unique non-null values.
    """
    s_no_na = s.dropna()
    unique_vals = pd.unique(s_no_na)
    if len(unique_vals) <= 2:
        vals_sorted = sorted(list(unique_vals), key=lambda x: str(x))
        mapping = {}
        if len(vals_sorted) == 2:
            mapping[vals_sorted[0]] = 0
            mapping[vals_sorted[1]] = 1
        else:
            mapping[vals_sorted[0]] = 0
        return s.map(lambda x: mapping.get(x, 0))
    return None

# New: sanitize sensitive groups to avoid extremes from rare/misc categories

def _sanitize_sensitive_groups(df: pd.DataFrame, sensitive: str) -> pd.DataFrame:
    try:
        if sensitive not in df.columns:
            return df
        df_s = df.copy()
        name = sensitive.lower()
        if ("sex" in name) or ("gender" in name):
            s = df_s[sensitive].astype(str).str.strip().str.lower()
            df_s[sensitive] = np.where(
                s.str.startswith("m"), "M",
                np.where(s.str.startswith("f"), "F", np.nan)
            )
        df_s = df_s.dropna(subset=[sensitive])
        vc = df_s[sensitive].value_counts(dropna=True)
        min_count = max(5, int(0.01 * len(df_s)))
        large_groups = [g for g, c in vc.items() if c >= min_count]
        if len(large_groups) >= 2:
            df_s = df_s[df_s[sensitive].isin(large_groups)]
            vc = df_s[sensitive].value_counts(dropna=True)
        if len(vc.index) > 2:
            allowed = list(vc.index[:2])
            df_s = df_s[df_s[sensitive].isin(allowed)]
        return df_s
    except Exception:
        return df

# --- Fairness metrics ---

def compute_statistical_parity_difference(df: pd.DataFrame, sensitive: str, target: str) -> float:
    """Difference in positive outcome rates between groups.

    Uses binary coercion for non-numeric targets to avoid degenerate zero rates.
    """
    if sensitive not in df.columns or target not in df.columns:
        return 0.0
    df_s = _sanitize_sensitive_groups(df, sensitive)
    groups = df_s[sensitive].dropna().unique()
    if len(groups) < 2:
        return 0.0
    # Prepare binary target when needed
    y = df_s[target]
    if not pd.api.types.is_numeric_dtype(y):
        y_bin = _ensure_binary_series(y)
        if y_bin is None:
            return 0.0
        y = y_bin
    rates = []
    for g in groups:
        sub_idx = (df_s[sensitive] == g)
        sub = y[sub_idx]
        if sub.empty:
            continue
        pos_rate = float(sub.mean())
        rates.append(pos_rate)
    if not rates:
        return 0.0
    return float(np.max(rates) - np.min(rates))

def compute_disparate_impact_ratio(df: pd.DataFrame, sensitive: str, target: str) -> float:
    """Ratio of positive rates (min/max).

    Uses binary coercion for non-numeric targets to avoid degenerate zero rates.
    """
    if sensitive not in df.columns or target not in df.columns:
        return 1.0
    df_s = _sanitize_sensitive_groups(df, sensitive)
    groups = df_s[sensitive].dropna().unique()
    if len(groups) < 2:
        return 1.0
    y = df_s[target]
    if not pd.api.types.is_numeric_dtype(y):
        y_bin = _ensure_binary_series(y)
        if y_bin is None:
            return 1.0
        y = y_bin
    rates = []
    for g in groups:
        sub_idx = (df_s[sensitive] == g)
        sub = y[sub_idx]
        if sub.empty:
            continue
        pos_rate = float(sub.mean())
        rates.append(pos_rate if pos_rate > 0 else 1e-9)
    if not rates:
        return 1.0
    return float(np.min(rates) / np.max(rates))

def compute_predictive_equality_difference(df: pd.DataFrame, sensitive: str, target: str, predictions_col: Optional[str] = None) -> float:
    """Difference in false positive rates across groups.

    Robustly bins y_true and y_pred into 0/1. If predictions are probabilistic
    (numeric with >2 unique values), threshold at 0.5.
    """
    if sensitive not in df.columns or target not in df.columns:
        return 0.0
    pred = predictions_col or target
    if pred not in df.columns:
        return 0.0
    df_s = _sanitize_sensitive_groups(df, sensitive)
    groups = df_s[sensitive].dropna().unique()
    if len(groups) < 2:
        return 0.0
    # y_true binning
    y_true = df_s[target]
    if not pd.api.types.is_numeric_dtype(y_true):
        y_true_bin = _ensure_binary_series(y_true)
        if y_true_bin is None:
            return 0.0
        y_true = y_true_bin
    else:
        uniq = pd.unique(y_true.dropna())
        if len(uniq) > 2:
            y_true = (y_true >= 0.5).astype(int)
        else:
            y_true = y_true.astype(int)
    # y_pred binning
    y_pred = df_s[pred]
    if pd.api.types.is_numeric_dtype(y_pred):
        if y_pred.dropna().nunique() > 2:
            y_pred = (y_pred >= 0.5).astype(int)
        else:
            y_pred = y_pred.astype(int)
    else:
        y_pred_bin = _ensure_binary_series(y_pred)
        if y_pred_bin is None:
            return 0.0
        y_pred = y_pred_bin
    fprates = []
    for g in groups:
        idx = (df_s[sensitive] == g)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.empty:
            continue
        negatives = (yt == 0).sum()
        if negatives == 0:
            fprates.append(0.0)
            continue
        fp = ((yp == 1) & (yt == 0)).sum()
        rate = float(fp) / float(negatives)
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
            if max_grp > 0 and (min_grp / max_grp) < settings.GROUP_RATIO_MIN:
                suggestions.append("Oversample minority groups or undersample majority to balance by sensitive feature.")
                suggestions.append("Use stratified sampling for train/validation/test splits to preserve group ratios.")
    except Exception:
        pass

    # Target imbalance
    try:
        y = df[target]
        if not pd.api.types.is_numeric_dtype(y):
            y_bin = _ensure_binary_series(y)
            y = y_bin if y_bin is not None else y
        pos = int((y == 1).sum()) if target in df.columns else None
        neg = int((y == 0).sum()) if target in df.columns else None
        if pos is not None and neg is not None:
            if min(pos, neg) / max(pos, neg) < settings.TARGET_RATIO_MIN:
                suggestions.append("Apply class weights or resampling to address target imbalance (e.g., WeightedLoss, SMOTE).")
    except Exception:
        pass

    spd = metrics.get("statistical_parity_difference")
    diratio = metrics.get("disparate_impact_ratio")
    ped = metrics.get("predictive_equality_difference")

    # Statistical parity gap
    if isinstance(spd, (int, float)) and abs(spd) >= settings.SPD_THRESHOLD:
        suggestions.append("Reweigh training samples to equalize selection rates across groups (Kamiran & Calders reweighing).")
        suggestions.append("Audit and remove proxy features highly correlated with the sensitive attribute.")
        suggestions.append("Consider group-specific thresholds to equalize selection rates.")

    # Disparate impact
    if isinstance(diratio, (int, float)) and diratio < settings.DIR_MIN_RATIO:
        suggestions.append("Review decision threshold; raise for advantaged group or lower for disadvantaged group to target DIR≈1.")
        suggestions.append("Calibrate model probabilities (Platt scaling/Isotonic) and re-evaluate group rates.")

    # Predictive equality (false positives)
    if isinstance(ped, (int, float)) and abs(ped) >= settings.PED_THRESHOLD:
        suggestions.append("Tune thresholds per group to reduce false positive disparities (equalized odds post-processing).")
        suggestions.append("Review labeling quality; inconsistent labels across groups can inflate FPR gaps.")

    # Predictions provided? enable more guidance
    if predictions_col and predictions_col in df.columns:
        suggestions.append("If predictions are probabilistic, pick thresholds per group via ROC/PR curves to balance TPR/FPR.")

    # General process improvements (configurable)
    if settings.ALWAYS_ON_TIPS:
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