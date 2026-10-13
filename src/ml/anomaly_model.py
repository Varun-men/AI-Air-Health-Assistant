"""
AURA Anomaly Model Service
---------------------------
Detects abnormal pollution readings using two complementary methods:

  1. Z-score    — fast, no training needed, uses 24h rolling stats
  2. Isolation Forest — trained on 7-day history, better for multivariate spikes

Interface:
    detect_anomaly(city, current_reading, db) -> dict
"""
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _ROOT / "models"
_MODEL_DIR.mkdir(exist_ok=True)

Z_THRESHOLD  = 3.0     # standard deviations
IF_CONTAMINATION = 0.05  # expected anomaly fraction for Isolation Forest

ANOMALY_FEATURES = ["pm25", "pm10", "no2", "aqi"]   # columns to use


def _if_path(city: str) -> Path:
    return _MODEL_DIR / f"isoforest_{city.lower().replace(' ', '_')}.pkl"


# ─── Training ──────────────────────────────────────────────────────────────────

def train_anomaly_model(city: str, db) -> bool:
    """
    Train an Isolation Forest on the last 7 days of data for a city.
    Saves to models/isoforest_<city>.pkl.
    """
    from src.database.crud import get_recent_air_quality
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import pickle

    history = get_recent_air_quality(db, city, hours=168)
    rows = _extract_features(history)

    if len(rows) < 20:
        logger.warning(f"[Anomaly] Not enough data for IF training in {city}.")
        return False

    X = np.array(rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=IF_CONTAMINATION,
        random_state=42,
    )
    model.fit(X_scaled)

    joblib.dump({"model": model, "scaler": scaler}, _if_path(city))
    logger.info(f"[Anomaly] Isolation Forest trained for {city}, {len(rows)} samples.")
    return True


# ─── Inference ─────────────────────────────────────────────────────────────────

def detect_anomaly(city: str, current_reading: dict, db) -> dict:
    """
    Detect if the current reading is anomalous.

    Returns:
        {
          is_anomaly:   bool
          method:       str    ("z_score" | "isolation_forest" | "combined")
          z_score:      float  (PM2.5 Z-score vs 24h history)
          if_score:     float  (Isolation Forest anomaly score, lower = more anomalous)
          severity:     str    ("NORMAL" | "ELEVATED" | "SPIKE" | "EXTREME")
          reason:       str
        }
    """
    from src.database.crud import get_recent_air_quality

    history = get_recent_air_quality(db, city, hours=24)
    result  = _z_score_check(current_reading, history)
    result  = _isolation_forest_check(city, current_reading, history, result, db)
    result["severity"] = _compute_severity(result)
    result["reason"]   = _build_reason(result, current_reading, city)

    if result["is_anomaly"]:
        logger.warning(f"[Anomaly] {city}: {result['reason']}")

    return result


# ─── Z-score check ─────────────────────────────────────────────────────────────

def _z_score_check(current: dict, history: list) -> dict:
    pm25_vals = [r.pm25 for r in history if r.pm25 is not None]
    current_pm25 = current.get("pm25")

    if len(pm25_vals) < 5 or current_pm25 is None:
        return {"is_anomaly": False, "method": "z_score", "z_score": 0.0, "if_score": None}

    mean, std = np.mean(pm25_vals), np.std(pm25_vals)
    z = abs(current_pm25 - mean) / std if std > 0 else 0.0

    return {
        "is_anomaly": z >= Z_THRESHOLD,
        "method":     "z_score",
        "z_score":    round(z, 2),
        "if_score":   None,
        "pm25_mean":  round(mean, 2),
        "pm25_std":   round(std, 2),
    }


# ─── Isolation Forest check ────────────────────────────────────────────────────

def _isolation_forest_check(city: str, current: dict, history: list, prev: dict, db) -> dict:
    mp = _if_path(city)

    # Auto-train if model doesn't exist and we have enough data
    if not mp.exists():
        train_anomaly_model(city, db)

    if not mp.exists():
        return prev  # not enough data, return z-score result only

    try:
        artifact = joblib.load(mp)
        model, scaler = artifact["model"], artifact["scaler"]

        row = [float(current.get(f) or 0) for f in ANOMALY_FEATURES]
        X_scaled = scaler.transform(np.array(row).reshape(1, -1))
        if_pred  = model.predict(X_scaled)[0]   # -1 = anomaly, 1 = normal
        if_score = float(model.score_samples(X_scaled)[0])

        is_if_anomaly = if_pred == -1
        combined = prev["is_anomaly"] or is_if_anomaly
        method   = "combined" if (prev["is_anomaly"] and is_if_anomaly) else (
                   "isolation_forest" if is_if_anomaly else prev["method"])

        return {**prev, "is_anomaly": combined, "method": method, "if_score": round(if_score, 4)}

    except Exception as e:
        logger.error(f"[Anomaly] IF inference failed: {e}")
        return prev


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _extract_features(history: list) -> list:
    rows = []
    for r in history:
        vals = [getattr(r, f, None) for f in ANOMALY_FEATURES]
        if all(v is not None for v in vals):
            rows.append([float(v) for v in vals])
    return rows


def _compute_severity(result: dict) -> str:
    z = result.get("z_score", 0)
    if not result["is_anomaly"]:
        return "NORMAL"
    if z < 4:
        return "ELEVATED"
    if z < 6:
        return "SPIKE"
    return "EXTREME"


def _build_reason(result: dict, reading: dict, city: str) -> str:
    if not result["is_anomaly"]:
        return "Reading within expected range."
    z      = result.get("z_score", 0)
    pm25   = reading.get("pm25", "?")
    mean   = result.get("pm25_mean", "?")
    return (
        f"Pollution spike detected in {city}. "
        f"PM2.5 = {pm25} µg/m³ "
        f"(24h mean: {mean}, Z-score: {z}). "
        f"Severity: {result.get('severity', '?')}."
    )
