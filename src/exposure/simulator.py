"""
AURA What-If Simulator
----------------------
Simulates alternative activities, durations, and times using the Forecast Model 
and the Exposure Engine to find lower-risk options for the user.
"""
import logging
from typing import List, Dict, Any

from src.exposure.engine import calculate_exposure
from src.ml.forecast_model import forecast_air_quality

logger = logging.getLogger(__name__)

def simulate_alternatives(
    city: str, 
    db, 
    activity: str, 
    duration_minutes: int, 
    mask_usage: bool = False, 
    current_env_risk: float = 50.0
) -> Dict[str, Any]:
    """
    Evaluate the current plan against future forecasted timeframes.
    
    Returns:
        Dict containing the current plan exposure and a list of alternatives.
    """
    logger.info(f"[Simulator] Running simulation for {city} | {activity} | {duration_minutes}m")
    
    # 1. Baseline (The "NOW" plan)
    current_plan = calculate_exposure(current_env_risk, activity, duration_minutes, mask_usage)
    current_plan["time_label"] = "NOW"
    current_plan["hours_ahead"] = 0
    current_plan["diff_from_current"] = 0.0
    
    # 2. Get Forecasts
    forecasts = forecast_air_quality(city, db)
    alternatives = []
    
    # 3. Simulate each forecasted horizon
    for f in forecasts:
        alt = calculate_exposure(
            base_risk_score=f["risk_score"], 
            activity=activity, 
            duration_minutes=duration_minutes, 
            mask_usage=mask_usage
        )
        
        alt["time_label"] = f["horizon"]
        alt["hours_ahead"] = f["hours_ahead"]
        alt["predicted_aqi"] = f["predicted_aqi"]
        alt["diff_from_current"] = round(alt["estimated_exposure"] - current_plan["estimated_exposure"], 1)
        
        alternatives.append(alt)
        
    # Sort alternatives by lowest exposure
    best_options = sorted(alternatives, key=lambda x: x["estimated_exposure"])

    return {
        "current_plan": current_plan,
        "alternatives": alternatives,
        "best_alternative": best_options[0] if best_options else None
    }

def optimize_activity(
    env_risk: float, 
    target_max_exposure: float = 50.0,
    base_activity: str = "Running",
    base_duration: int = 60
) -> Dict[str, Any]:
    """
    If time cannot be changed, what else can the user change to lower exposure?
    Simulates changing activity or reducing duration.
    """
    suggestions = []
    orig = calculate_exposure(env_risk, base_activity, base_duration, False)["estimated_exposure"]
    
    # Try reducing duration
    durations_to_test = [45, 30, 15]
    # Try mask first
    res_mask = calculate_exposure(env_risk, base_activity, base_duration, True)
    if res_mask["estimated_exposure"] <= target_max_exposure:
        suggestions.append(f"Wear an N95 mask (Exposure drops from {orig} to {res_mask['estimated_exposure']})")
    elif res_mask["estimated_exposure"] < orig:
        suggestions.append(f"Wear an N95 mask to cut exposure by 60% (Exposure: {res_mask['estimated_exposure']})")

    # Try changing duration
    for d in durations_to_test:
        if d >= base_duration: continue
        res_d = calculate_exposure(env_risk, base_activity, d, False)
        if res_d["estimated_exposure"] <= target_max_exposure:
            suggestions.append(f"Reduce duration to {d} minutes (Exposure: {res_d['estimated_exposure']})")
            break

    # Try changing activity
    if base_activity in ["Running", "Cycling"]:
        res_act = calculate_exposure(env_risk, "Walking", base_duration, False)
        if res_act["estimated_exposure"] <= target_max_exposure:
            suggestions.append(f"Switch activity from {base_activity} to Walking (Exposure: {res_act['estimated_exposure']})")

    # If still empty or strict, try combined Mask + Duration / Walking
    if len(suggestions) < 2:
        res_comb = calculate_exposure(env_risk, "Walking", 30, True)
        suggestions.append(f"Combine Walking for 30m with an N95 mask (Exposure: {res_comb['estimated_exposure']})")

    return {
        "original_exposure": orig,
        "suggestions": suggestions
    }
