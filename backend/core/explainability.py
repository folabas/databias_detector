"""Explainability utilities for feature influence.

Provides SHAP-based feature importance with a graceful correlation fallback.
"""
from typing import Dict, Any, Optional
import pandas as pd
from .config import settings


def explain_feature_influence(df: pd.DataFrame, target: str, predictions_col: Optional[str] = None) -> Dict[str, Any]:
    """Return feature importances using SHAP when available; otherwise correlation fallback.

    Performance-aware: optionally samples rows and can skip SHAP based on settings.
    """
    # If SHAP is disabled via settings, go straight to correlation fallback
    if not settings.USE_SHAP_EXPLAINABILITY:
        return _correlation_fallback(df, target)

    # Sample rows to cap compute cost
    if isinstance(settings.EXPLAIN_MAX_ROWS, int) and settings.EXPLAIN_MAX_ROWS > 0:
        try:
            if df.shape[0] > settings.EXPLAIN_MAX_ROWS:
                df = df.sample(n=settings.EXPLAIN_MAX_ROWS, random_state=0)
        except Exception:
            pass
    try:
        import shap  # type: ignore
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        # Prepare features (basic encoding for categorical)
        X = df.drop(columns=[target])
        y = df[target]
        # If predictions are provided, prefer those as proxy label when numeric probabilities
        if predictions_col and predictions_col in df.columns:
            y_pred = df[predictions_col]
            if y_pred.dropna().nunique() > 2 and y_pred.dtype.kind in "fc":
                y = (y_pred >= 0.5).astype(int)
        X = X.copy()
        for c in X.columns:
            if X[c].dtype == object:
                le = LabelEncoder()
                try:
                    X[c] = le.fit_transform(X[c].astype(str))
                except Exception:
                    X[c] = 0
        # Fit simple model
        try:
            model = RandomForestClassifier(
                n_estimators=getattr(settings, "EXPLAIN_RF_TREES", 50),
                max_depth=getattr(settings, "EXPLAIN_RF_MAX_DEPTH", 6),
                random_state=0,
            )
            model.fit(X.fillna(0), y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.fillna(0))
            # Use mean absolute shap values
            import numpy as np
            vals = np.abs(shap_values[1]).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
            importances = [{"feature": f, "importance": float(v)} for f, v in zip(X.columns, vals)]
            importances.sort(key=lambda x: x["importance"], reverse=True)
            return {"feature_importances": importances}
        except Exception:
            raise
    except Exception:
        return _correlation_fallback(df, target)


def _correlation_fallback(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Correlation-based explainability fallback with label encoding for categoricals."""
    try:
        from sklearn.preprocessing import LabelEncoder
        corr: Dict[str, float] = {}

        # Ensure target is numeric; if categorical, label-encode it
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        if target_numeric.isna().all() or df[target].dtype == object or df[target].nunique() <= 20:
            le_t = LabelEncoder()
            try:
                target_numeric = pd.Series(le_t.fit_transform(df[target].astype(str).fillna('missing')))
            except Exception:
                target_numeric = pd.to_numeric(df[target].astype('category').cat.codes, errors="coerce")

        for c in df.columns:
            if c == target:
                continue
            try:
                # Try numeric correlation first
                feature_numeric = pd.to_numeric(df[c], errors="coerce")
                if not feature_numeric.isna().all():
                    corr_val = abs(feature_numeric.corr(target_numeric))
                    if corr_val == corr_val:  # Check for NaN
                        corr[c] = float(corr_val)
                        continue

                # For categorical features, use label encoding then correlation
                if df[c].dtype == object or df[c].nunique() <= 10:
                    le = LabelEncoder()
                    try:
                        feature_encoded = le.fit_transform(df[c].astype(str).fillna('missing'))
                        corr_val = abs(pd.Series(feature_encoded).corr(target_numeric))
                        corr[c] = float(corr_val) if corr_val == corr_val else 0.0
                    except Exception:
                        corr[c] = 0.0
                else:
                    corr[c] = 0.0
            except Exception:
                corr[c] = 0.0
        importances = [{"feature": k, "importance": v} for k, v in sorted(corr.items(), key=lambda x: x[1], reverse=True)]
        return {"feature_importances": importances, "reason": "Correlation-based importance (SHAP unavailable)"}
    except Exception:
        return {"reason": "Explainability unavailable"}