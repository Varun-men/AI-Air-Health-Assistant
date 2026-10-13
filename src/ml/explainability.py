"""
AURA Explainability Service
-----------------------------
Generates SHAP-based feature contribution explanations
for the risk model's predictions.

Interface:
    explain_prediction(data: dict) -> dict
"""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Features to highlight in the UI explanation
DISPLAY_FEATURES = ["pm25", "pm10", "no2", "so2", "co", "o3",
                    "temperature", "humidity", "wind_speed", "aqi"]


def explain_prediction(data: dict) -> dict:
    """
    Generate a feature contribution explanation for a risk prediction.

    Args:
        data: same dict passed to predict_risk()

    Returns:
        {
          method: str                      ("shap" | "feature_importance" | "fallback")
          contributions: {feature: value}  positive = increases risk
          top_factors: list                top 3 features by absolute contribution
          explanation_text: str            human-readable summary
          available: bool
        }
    """
    try:
        return _shap_explanation(data)
    except ImportError:
        logger.warning("[Explainability] SHAP not installed; using feature importance.")
        return _feature_importance_explanation(data)
    except Exception as e:
        logger.error(f"[Explainability] SHAP failed: {e}")
        return _feature_importance_explanation(data)


# ─── SHAP explanation ──────────────────────────────────────────────────────────

def _shap_explanation(data: dict) -> dict:
    import shap
    from src.ml.risk_model import _load, _build_feature_row

    model, feature_columns = _load()
    row = _build_feature_row(data, feature_columns)

    # TreeExplainer works with Random Forest and GBM
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(row)

    # For classifiers predict_proba returns one array per class
    if isinstance(shap_vals, list):
        # Use the highest-risk class
        shap_vals = shap_vals[-1]

    contributions = {
        col: round(float(shap_vals[0][i]), 4)
        for i, col in enumerate(feature_columns)
        if col in DISPLAY_FEATURES
    }

    return _package_result(contributions, "shap")


# ─── Feature importance fallback ───────────────────────────────────────────────

def _feature_importance_explanation(data: dict) -> dict:
    """
    Use model's built-in feature_importances_ as a simpler proxy.
    Multiplied by the input value to give a directional contribution.
    """
    try:
        from src.ml.risk_model import _load, _build_feature_row

        model, feature_columns = _load()
        row = _build_feature_row(data, feature_columns)

        # Get base estimator for MultiOutput or direct model
        base = getattr(model, "estimators_", None)
        if base:
            importances = np.mean([e.feature_importances_ for e in base], axis=0)
        else:
            importances = model.feature_importances_

        contributions = {
            col: round(float(importances[i] * row[0][i]), 4)
            for i, col in enumerate(feature_columns)
            if col in DISPLAY_FEATURES
        }
        return _package_result(contributions, "feature_importance")

    except Exception as e:
        logger.error(f"[Explainability] Feature importance failed: {e}")
        return _fallback(data)


# ─── Rule-based fallback ───────────────────────────────────────────────────────

def _fallback(data: dict) -> dict:
    """
    Construct rough contributions from raw pollutant values
    when the model is unavailable.
    """
    weights = {
        "pm25": 0.40, "pm10": 0.20, "no2": 0.15,
        "aqi":  0.10, "o3":   0.08, "temperature": 0.04,
        "humidity": 0.03,
    }
    contributions = {}
    for feat, w in weights.items():
        val = data.get(feat)
        if val is not None:
            contributions[feat] = round(float(val) * w / 100, 4)

    return _package_result(contributions, "fallback")


# ─── Packaging ─────────────────────────────────────────────────────────────────

def _package_result(contributions: dict, method: str) -> dict:
    if not contributions:
        return {"method": method, "contributions": {}, "top_factors": [],
                "explanation_text": "Explanation unavailable.", "available": False}

    # Sort by absolute contribution
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_factors = [k for k, _ in sorted_contribs[:3]]

    total = sum(abs(v) for v in contributions.values()) or 1
    pct   = {k: round(abs(v) / total * 100, 1) for k, v in sorted_contribs}

    top_names = ", ".join(top_factors)
    explanation_text = (
        f"The model's prediction was most influenced by: {top_names}. "
        f"These factors together account for the majority of the estimated risk."
    )

    return {
        "method":           method,
        "contributions":    contributions,
        "pct_contributions": pct,
        "top_factors":      top_factors,
        "explanation_text": explanation_text,
        "available":        True,
    }
