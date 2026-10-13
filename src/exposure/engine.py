"""
AURA Exposure Engine
--------------------
Translates raw environmental risk into Estimated Personal Exposure Risk.
Accounts for activity intensity (respiration rate), duration, and mask usage.

Important: This produces an estimated risk score (0-100), not a clinical diagnosis.
"""
import logging

logger = logging.getLogger(__name__)

# Multipliers based roughly on ventilation/respiration rates
ACTIVITY_MULTIPLIERS = {
    "Indoor": 0.4,
    "Resting": 0.8,
    "Walking": 1.0,
    "Cycling": 1.5,
    "Running": 2.0,
    "Other": 1.0
}

def get_duration_multiplier(minutes: int) -> float:
    """Scale exposure based on time spent outdoors."""
    if minutes <= 15:
        return 0.5
    elif minutes <= 30:
        return 0.8
    elif minutes <= 60:
        return 1.0
    elif minutes <= 120:
        return 1.3
    else:
        return 1.6


def calculate_exposure(
    base_risk_score: float, 
    activity: str = "Walking", 
    duration_minutes: int = 60, 
    mask_usage: bool = False
) -> dict:
    """
    Calculate the personalized exposure risk score.
    
    Args:
        base_risk_score: Float 0-100 representing environmental risk.
        activity: String matching ACTIVITY_MULTIPLIERS keys.
        duration_minutes: Integer minutes of exposure.
        mask_usage: Boolean indicating if protective equipment (e.g., N95) is used.
        
    Returns:
        dict: { estimated_exposure, risk_level, factors }
    """
    try:
        # 1. Base factors
        act_mult = ACTIVITY_MULTIPLIERS.get(activity, 1.0)
        dur_mult = get_duration_multiplier(duration_minutes)
        mask_mult = 0.35 if mask_usage else 1.0  # Assumes ~65% effective risk reduction for proper mask

        # 2. Compute raw exposure
        raw_exposure = base_risk_score * act_mult * dur_mult * mask_mult

        # 3. Clamp to 0-100 scale
        exposure = min(max(raw_exposure, 0.0), 100.0)
        
        # 4. Determine Level
        if exposure < 25:
            level = "LOW"
        elif exposure < 50:
            level = "MODERATE"
        elif exposure < 75:
            level = "HIGH"
        else:
            level = "VERY HIGH"

        return {
            "estimated_exposure": round(exposure, 1),
            "risk_level": level,
            "factors": {
                "base_environmental_risk": round(base_risk_score, 1),
                "activity_multiplier": act_mult,
                "duration_multiplier": dur_mult,
                "mask_multiplier": mask_mult
            }
        }
    except Exception as e:
        logger.error(f"[ExposureEngine] Failed to calculate exposure: {e}")
        return {
            "estimated_exposure": base_risk_score,
            "risk_level": "UNKNOWN",
            "factors": {}
        }
