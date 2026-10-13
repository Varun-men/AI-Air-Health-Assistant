"""
AURA Risk Model Service
-----------------------
Wraps the existing trained Random Forest model (best_aqi_model.pkl)
into a clean, callable service.

Interface:
    predict_risk(data: dict) -> dict
"""
import joblib
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
MODEL_PATH   = _ROOT / "best_aqi_model.pkl"
FEATURE_PATH = _ROOT / "feature_columns.pkl"

MODEL_VERSION = "rf_v1"

# Risk level thresholds (0–100 score)
RISK_THRESHOLDS = [
    (0,  25,  "LOW"),
    (25, 50,  "MODERATE"),
    (50, 75,  "HIGH"),
    (75, 100, "VERY HIGH"),
]


def _score_to_level(score: float) -> str:
    for lo, hi, label in RISK_THRESHOLDS:
        if lo <= score < hi:
            return label
    return "VERY HIGH"


# ─── Model loader (lazy, cached) ───────────────────────────────────────────────
_model = None
_feature_columns = None


def _load():
    global _model, _feature_columns
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        _feature_columns = joblib.load(FEATURE_PATH)
        logger.info(f"[RiskModel] Loaded model from {MODEL_PATH}")
    return _model, _feature_columns


# ─── Public API ────────────────────────────────────────────────────────────────

def predict_risk(data: dict) -> dict:
    """
    Predict environmental health risk from a data record.

    Args:
        data: dict with keys matching air_quality table columns
              (aqi, pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed, ...)

    Returns:
        {
            risk_score:  float  0–100
            risk_level:  str    LOW / MODERATE / HIGH / VERY HIGH
            model_version: str
            input_features: dict   (features actually fed to model)
            prediction_available: bool
        }
    """
    try:
        model, feature_columns = _load()
        row = _build_feature_row(data, feature_columns)

        # Model may return probability or direct class
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            # Use highest-risk class probability scaled to 0-100
            risk_score = round(float(np.max(proba)) * 100, 1)
        else:
            pred = model.predict(row)[0]
            # Normalise raw prediction to 0-100 if numeric
            risk_score = round(min(float(pred), 100), 1)

        risk_level = _score_to_level(risk_score)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "model_version": MODEL_VERSION,
            "input_features": dict(zip(feature_columns, row[0])),
            "prediction_available": True,
        }

    except Exception as e:
        logger.error(f"[RiskModel] Prediction failed: {e}")
        return {
            "risk_score": None,
            "risk_level": "UNKNOWN",
            "model_version": MODEL_VERSION,
            "input_features": {},
            "prediction_available": False,
            "error": str(e),
        }


def get_feature_columns() -> list:
    """Return the list of feature columns the model expects."""
    _, fc = _load()
    return list(fc)


# ─── Feature builder ───────────────────────────────────────────────────────────

def _build_feature_row(data: dict, feature_columns) -> np.ndarray:
    """
    Build a 2D numpy array (1 row) aligned to feature_columns.
    Missing values are filled with 0 (safe fallback for tree models).
    """
    from datetime import datetime

    # Enrich with time features if timestamp available
    ts = data.get("timestamp")
    if isinstance(ts, datetime):
        data = {**data, "hour": ts.hour, "day_of_week": ts.weekday(), "month": ts.month}

    row = [float(data.get(col, 0) or 0) for col in feature_columns]
    return np.array(row).reshape(1, -1)
