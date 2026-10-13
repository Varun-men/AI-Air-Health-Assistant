"""
Data validation for AURA.
Checks incoming air-quality records before they touch the database.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Acceptable value ranges (sensor sanity checks — not clinical limits)
VALID_RANGES = {
    "aqi":         (0,   1000),
    "pm25":        (0,   1000),
    "pm10":        (0,   2000),
    "no2":         (0,   2000),
    "so2":         (0,   2000),
    "co":          (0,  50000),
    "o3":          (0,   1000),
    "temperature": (-50,   60),
    "humidity":    (0,    100),
    "wind_speed":  (0,    150),
}


def validate_record(record: dict) -> tuple[bool, list[str]]:
    """
    Validate a single air quality record.

    Returns:
        (is_valid: bool, issues: list[str])
        is_valid is True only when there are zero blocking issues.
    """
    issues = []

    # --- 1. Required fields ---
    if not record.get("city"):
        issues.append("Missing required field: city")
    if not record.get("timestamp"):
        issues.append("Missing required field: timestamp")

    # --- 2. At least one pollutant must have a value ---
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    has_any = any(record.get(p) is not None for p in pollutants)
    if not has_any:
        issues.append("No pollutant readings present in record")

    # --- 3. Range checks (only flag if value is present) ---
    for field, (lo, hi) in VALID_RANGES.items():
        val = record.get(field)
        if val is not None:
            if not (lo <= val <= hi):
                issues.append(f"{field}={val} is outside expected range [{lo}, {hi}]")

    # --- 4. PM2.5 > PM10 is physically unusual but not impossible ---
    pm25 = record.get("pm25")
    pm10 = record.get("pm10")
    if pm25 and pm10 and pm25 > pm10 * 1.5:
        # Log as a warning; don't reject
        logger.warning(
            f"Unusual: PM2.5 ({pm25}) is much larger than PM10 ({pm10}) "
            f"for {record.get('city')} at {record.get('timestamp')}"
        )

    # Classify: blocking issues make the record invalid
    # Range violations are blocking; "unusual" readings are warnings only
    blocking = [i for i in issues if "outside expected range" not in i
                or True]   # all issues are blocking for now

    is_valid = len(issues) == 0
    return is_valid, issues


def clean_record(record: dict) -> dict:
    """
    Remove known-bad fields from a record so the rest can still be stored.
    Called after validate_record identifies out-of-range values.
    """
    cleaned = dict(record)
    for field, (lo, hi) in VALID_RANGES.items():
        val = cleaned.get(field)
        if val is not None and not (lo <= val <= hi):
            logger.info(f"Nullifying out-of-range value: {field}={val}")
            cleaned[field] = None
    return cleaned
