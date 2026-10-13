"""
AURA Decision Agent (Proactive Monitor)
---------------------------------------
Runs continuously or triggered by the Data Agent.
Looks for environmental spikes and cross-references them with user schedules
to generate proactive, personalized alerts.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import List

from src.database.connection import get_db
from src.database.models import User, Alert
from src.database.crud import (
    get_latest_air_quality, 
    insert_alert, 
    log_agent_action
)
from src.ml.forecast_model import forecast_air_quality
from src.ml.anomaly_model import detect_anomaly
from src.exposure.engine import calculate_exposure

logger = logging.getLogger(__name__)

class DecisionAgent:
    def __init__(self):
        self.name = "DECISION_AGENT"

    def evaluate_proactive_alerts(self, city: str):
        """
        Evaluate if any users in the city need an alert right now.
        """
        logger.info(f"[DecisionAgent] Evaluating proactive alerts for {city}...")
        
        with get_db() as db:
            # 1. Check current conditions & anomalies
            latest_aq = get_latest_air_quality(db, city)
            if not latest_aq:
                return
                
            reading_dict = {
                "aqi": latest_aq.aqi, "pm25": latest_aq.pm25, "pm10": latest_aq.pm10,
                "no2": latest_aq.no2, "timestamp": latest_aq.timestamp
            }
            anomaly_res = detect_anomaly(city, reading_dict, db)
            
            # 2. Check future forecast
            forecasts = forecast_air_quality(city, db)
            
            # 3. Identify Risk Windows (Current or Future)
            risk_windows = []
            
            # Add current time if anomalous or HIGH risk
            if anomaly_res.get("is_anomaly") or (latest_aq.aqi and latest_aq.aqi > 150):
                risk_windows.append({
                    "time": datetime.utcnow(),
                    "reason": anomaly_res.get("reason", "Current AQI is HIGH."),
                    "aqi": latest_aq.aqi,
                    "is_current": True
                })
                
            # Add future times if HIGH risk
            for f in forecasts:
                if f.get("risk_score", 0) >= 50.0:  # HIGH or VERY HIGH
                    risk_windows.append({
                        "time": f["predicted_for"],
                        "reason": f"Forecast predicts {f['risk_level']} pollution (+{f['hours_ahead']}h).",
                        "aqi": f["predicted_aqi"],
                        "is_current": False
                    })

            if not risk_windows:
                log_agent_action(db, self.name, "CHECK_ALERTS", "", "No risk windows detected.", "SUCCESS")
                return

            # 4. Check affected users
            users: List[User] = db.query(User).filter(User.city == city).all()
            alerts_generated = 0
            
            for user in users:
                for window in risk_windows:
                    if self._user_overlaps_with_window(user, window["time"]):
                        # Calculate their exact exposure if they proceed
                        exp = calculate_exposure(
                            base_risk_score=min((window["aqi"] or 100) / 5.0, 100.0),
                            activity=user.primary_activity,
                            duration_minutes=int(user.outdoor_hours * 60),
                            mask_usage=user.mask_usage
                        )
                        
                        # Only alert if their personal exposure is high
                        if exp["estimated_exposure"] >= 50.0:
                            msg = (
                                f"⚠️ AURA Alert: Your planned {user.primary_activity} "
                                f"around {user.usual_outdoor_start} overlaps with elevated pollution. "
                                f"{window['reason']} Your estimated exposure risk would be {exp['risk_level']}."
                            )
                            
                            insert_alert(db, {
                                "user_id": user.id,
                                "alert_type": "EXPOSURE_WARNING",
                                "severity": "WARNING",
                                "message": msg,
                                "trigger_data": json.dumps({"aqi": window["aqi"], "exposure": exp["estimated_exposure"]}),
                                "status": "ACTIVE"
                            })
                            alerts_generated += 1
                            break # Don't alert the same user twice in one run
            
            log_agent_action(
                db, self.name, "GENERATE_ALERTS", "evaluate_proactive_alerts",
                f"Evaluated {len(users)} users. Generated {alerts_generated} alerts.",
                "SUCCESS"
            )

    def _user_overlaps_with_window(self, user: User, window_time: datetime) -> bool:
        """
        Check if the risk window overlaps with the user's usual outdoor schedule.
        (Simplified logic: checks if the hour matches).
        """
        if not user.usual_outdoor_start:
            return False
            
        try:
            start_hour = int(user.usual_outdoor_start.split(":")[0])
            # If the risk window is within 1 hour of their start time
            return abs(window_time.hour - start_hour) <= 1
        except Exception:
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DecisionAgent()
    agent.evaluate_proactive_alerts("Delhi")
