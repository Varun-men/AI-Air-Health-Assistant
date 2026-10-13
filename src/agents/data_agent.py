"""
AURA Data Agent — the Perception layer.

Responsibilities:
  1. Fetch live AQI from OpenAQ
  2. Fetch weather from OpenWeatherMap
  3. Validate the combined record
  4. Deduplicate against database
  5. Detect anomalies (Z-score on last 24h readings)
  6. Store to database
  7. Log every step to agent_logs

Can be triggered manually or by the scheduler.
"""
import time
import logging
import json
import numpy as np
from datetime import datetime
from typing import Optional

from src.data.openaq_client import OpenAQClient
from src.data.weather_client import WeatherClient
from src.data.validator import validate_record, clean_record
from src.database.connection import get_db
from src.database.crud import (
    insert_air_quality, get_recent_air_quality,
    is_duplicate, log_agent_action, insert_alert
)
from src.utils.config import SUPPORTED_CITIES

logger = logging.getLogger(__name__)

ANOMALY_Z_THRESHOLD = 3.0   # flag if value > 3 std-devs from 24h mean


class DataAgent:
    """
    The Perception Agent for AURA.
    Observe → Validate → Detect → Store → Log.
    """

    def __init__(self):
        self.aq_client = OpenAQClient()
        self.weather_client = WeatherClient()
        self.name = "DATA_AGENT"

    # ─── Public API ────────────────────────────────────────────────────────────

    def run_for_city(self, city: str) -> Optional[dict]:
        """Full pipeline for a single city. Returns stored record or None."""
        t0 = time.time()
        logger.info(f"[DataAgent] Starting pipeline for {city}.")

        with get_db() as db:
            # 1. Fetch AQ data
            record = self._fetch_aq(db, city)
            if record is None:
                return None

            # 2. Merge weather
            record = self._merge_weather(db, city, record)

            # 3. Validate
            record = self._validate(db, city, record)
            if record is None:
                return None

            # 4. Deduplicate
            if self._is_duplicate(db, city, record):
                return None

            # 5. Anomaly detection
            self._check_anomaly(db, city, record)

            # 6. Store
            row = insert_air_quality(db, record)
            elapsed = round(time.time() - t0, 2)
            log_agent_action(
                db, self.name, "STORE_MEASUREMENT", "insert_air_quality",
                f"Stored reading for {city}",
                f"AQI={record.get('aqi')} PM2.5={record.get('pm25')}",
                "SUCCESS", elapsed,
            )
            logger.info(f"[DataAgent] Stored record for {city} in {elapsed}s.")
            return record

    def run_all_cities(self):
        """Run the full pipeline for all configured cities."""
        for city in SUPPORTED_CITIES:
            try:
                self.run_for_city(city)
            except Exception as e:
                logger.error(f"[DataAgent] Error processing {city}: {e}")

    # ─── Private steps ─────────────────────────────────────────────────────────

    def _fetch_aq(self, db, city: str) -> Optional[dict]:
        record = self.aq_client.fetch_latest(city)
        if record:
            log_agent_action(db, self.name, "FETCH_AIR_QUALITY", "fetch_latest",
                             f"Fetched AQ data for {city}", "OK", "SUCCESS")
        else:
            log_agent_action(db, self.name, "FETCH_AIR_QUALITY", "fetch_latest",
                             f"No AQ data returned for {city}", "", "FAILED")
        return record

    def _merge_weather(self, db, city: str, record: dict) -> dict:
        lat = record.get("latitude")
        lon = record.get("longitude")
        weather = self.weather_client.fetch_current(city, lat=lat, lon=lon)
        if weather:
            record.update(weather)
            log_agent_action(db, self.name, "FETCH_WEATHER", "fetch_current",
                             f"Merged weather for {city}", str(weather), "SUCCESS")
        else:
            log_agent_action(db, self.name, "FETCH_WEATHER", "fetch_current",
                             "Weather API unavailable", "", "SKIPPED")
        return record

    def _validate(self, db, city: str, record: dict) -> Optional[dict]:
        valid, issues = validate_record(record)
        if not valid:
            log_agent_action(db, self.name, "VALIDATE_DATA", "validate_record",
                             f"Validation failed for {city}",
                             "; ".join(issues), "FAILED")
            logger.warning(f"[DataAgent] Rejected record for {city}: {issues}")
            return None
        # Partial clean: null out individual bad fields but keep the record
        cleaned = clean_record(record)
        log_agent_action(db, self.name, "VALIDATE_DATA", "validate_record",
                         f"Record validated for {city}", "OK", "SUCCESS")
        return cleaned

    def _is_duplicate(self, db, city: str, record: dict) -> bool:
        src_ts = record.get("source_timestamp")
        if src_ts and is_duplicate(db, city, src_ts):
            log_agent_action(db, self.name, "DEDUPLICATE", "is_duplicate",
                             f"Duplicate record skipped for {city}",
                             str(src_ts), "SKIPPED")
            logger.info(f"[DataAgent] Duplicate skipped for {city}.")
            return True
        return False

    def _check_anomaly(self, db, city: str, record: dict):
        """Z-score anomaly detection on PM2.5 vs last 24h."""
        history = get_recent_air_quality(db, city, hours=24)
        if len(history) < 5:
            return   # not enough history yet

        pm25_vals = [r.pm25 for r in history if r.pm25 is not None]
        current_pm25 = record.get("pm25")
        if not pm25_vals or current_pm25 is None:
            return

        mean, std = np.mean(pm25_vals), np.std(pm25_vals)
        if std == 0:
            return

        z = abs(current_pm25 - mean) / std
        if z >= ANOMALY_Z_THRESHOLD:
            msg = (
                f"Pollution spike detected in {city}. "
                f"PM2.5 = {current_pm25:.1f} µg/m³ "
                f"(24h mean: {mean:.1f}, Z-score: {z:.1f})."
            )
            logger.warning(f"[DataAgent] ANOMALY — {msg}")
            log_agent_action(db, self.name, "DETECT_ANOMALY", "z_score",
                             f"PM2.5 spike in {city}", msg, "SUCCESS")
            insert_alert(db, {
                "alert_type": "POLLUTION_SPIKE",
                "severity":   "WARNING" if z < 5 else "CRITICAL",
                "message":    msg,
                "trigger_data": json.dumps({
                    "city": city, "pm25": current_pm25,
                    "mean": round(mean, 2), "z_score": round(z, 2),
                }),
                "status": "ACTIVE",
            })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.database.connection import init_db
    init_db()
    agent = DataAgent()
    agent.run_for_city("Delhi")
