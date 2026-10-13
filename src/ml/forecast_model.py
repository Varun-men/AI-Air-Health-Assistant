"""
AURA Forecast Model Service
----------------------------
Predicts future AQI at +1h, +3h, +6h, +12h horizons
using historical readings from the database.

Strategy:
  - Uses a rolling-lag feature set (last 6 readings → next AQI)
  - Model: scikit-learn GradientBoostingRegressor
  - Falls back to recent trend extrapolation if < 6 readings available

Interface:
    train_forecast_model(city)        → trains + saves model
    forecast_air_quality(city, db)    → returns list of predictions
"""
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _ROOT / "models"
_MODEL_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 3, 6, 12]           # hours ahead
N_LAGS   = 6                        # how many past readings to use as features
MODEL_VERSION = "gbr_forecast_v1"


def _model_path(city: str) -> Path:
    return _MODEL_DIR / f"forecast_{city.lower().replace(' ', '_')}.pkl"


# ─── Training ──────────────────────────────────────────────────────────────────

def train_forecast_model(city: str, db) -> bool:
    """
    Train a forecast model for `city` using data from the database.
    Saves model to models/forecast_<city>.pkl.
    Returns True if training succeeded.
    """
    from src.database.crud import get_recent_air_quality
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    history = get_recent_air_quality(db, city, hours=168)   # last 7 days
    aqi_vals = [r.aqi for r in history if r.aqi is not None]

    if len(aqi_vals) < N_LAGS + max(HORIZONS) + 1:
        logger.warning(f"[Forecast] Not enough data for {city} ({len(aqi_vals)} readings).")
        return False

    X, y = _build_training_set(aqi_vals)
    if len(X) < 10:
        return False

    base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, y)

    joblib.dump(model, _model_path(city))
    logger.info(f"[Forecast] Model trained for {city}, {len(X)} samples.")
    return True


def _build_training_set(aqi_vals: list):
    """Build lag-feature matrix and multi-output target."""
    X, y = [], []
    # targets: AQI at t+1, t+3, t+6, t+12 (index offsets assuming ~1h intervals)
    horizon_idx = HORIZONS   # [1, 3, 6, 12]
    max_h = max(horizon_idx)

    for i in range(N_LAGS, len(aqi_vals) - max_h):
        features = aqi_vals[i - N_LAGS: i]                    # last N readings
        targets  = [aqi_vals[i + h] for h in horizon_idx]     # future values
        X.append(features)
        y.append(targets)

    return np.array(X), np.array(y)


# ─── Inference ─────────────────────────────────────────────────────────────────

def forecast_air_quality(city: str, db) -> list:
    """
    Predict AQI at +1h, +3h, +6h, +12h for a city.

    Returns list of dicts:
        [
          { horizon: "+1h",  predicted_aqi: 214.5, risk_level: "HIGH" },
          ...
        ]
    """
    from src.database.crud import get_recent_air_quality
    from src.ml.risk_model import _score_to_level

    history = get_recent_air_quality(db, city, hours=24)
    aqi_vals = [r.aqi for r in history if r.aqi is not None]

    if len(aqi_vals) < N_LAGS:
        logger.info(f"[Forecast] Insufficient data for {city}; using trend fallback.")
        return _trend_fallback(aqi_vals, city)

    # Try loading trained model
    mp = _model_path(city)
    if not mp.exists():
        logger.info(f"[Forecast] No saved model for {city}; training now...")
        if not train_forecast_model(city, db):
            return _trend_fallback(aqi_vals, city)

    try:
        model = joblib.load(mp)
        features = np.array(aqi_vals[-N_LAGS:]).reshape(1, -1)
        preds = model.predict(features)[0]   # shape: (4,)

        results = []
        now = datetime.utcnow()
        for i, h in enumerate(HORIZONS):
            predicted_aqi = round(float(np.clip(preds[i], 0, 500)), 1)
            risk_score = _aqi_to_risk_score(predicted_aqi)
            results.append({
                "horizon":       f"+{h}h",
                "hours_ahead":   h,
                "predicted_for": now + timedelta(hours=h),
                "predicted_aqi": predicted_aqi,
                "risk_level":    _score_to_level(risk_score),
                "risk_score":    risk_score,
                "model_version": MODEL_VERSION,
            })
        return results

    except Exception as e:
        logger.error(f"[Forecast] Inference failed for {city}: {e}")
        return _trend_fallback(aqi_vals, city)


def _trend_fallback(aqi_vals: list, city: str) -> list:
    """Simple linear trend extrapolation when model is unavailable."""
    from src.ml.risk_model import _score_to_level

    if not aqi_vals:
        return []

    base = aqi_vals[-1]
    if len(aqi_vals) >= 3:
        trend = (aqi_vals[-1] - aqi_vals[-3]) / 2   # avg change per reading
    else:
        trend = 0.0

    results = []
    now = datetime.utcnow()
    for h in HORIZONS:
        predicted_aqi = round(float(np.clip(base + trend * h, 0, 500)), 1)
        risk_score = _aqi_to_risk_score(predicted_aqi)
        results.append({
            "horizon":       f"+{h}h",
            "hours_ahead":   h,
            "predicted_for": now + timedelta(hours=h),
            "predicted_aqi": predicted_aqi,
            "risk_level":    _score_to_level(risk_score),
            "risk_score":    risk_score,
            "model_version": "trend_fallback",
        })
    return results


def _aqi_to_risk_score(aqi: float) -> float:
    """Map AQI to a 0-100 risk score for display."""
    return round(min(aqi / 5.0, 100.0), 1)
