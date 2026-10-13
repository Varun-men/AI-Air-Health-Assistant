"""
AURA Supervisor Agent (Orchestrator)
------------------------------------
The brain behind the 'Ask AURA' chat feature.
Takes a user prompt, determines intent, orchestrates the simulator/exposure engine tools,
and generates a personalized, agentic response.
"""
import logging
from datetime import datetime
from typing import Dict, Any

from src.database.connection import get_db
from src.database.models import User
from src.database.crud import get_latest_air_quality, log_agent_action
from src.exposure.simulator import simulate_alternatives, optimize_activity
from src.ml.forecast_model import forecast_air_quality

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self):
        self.name = "SUPERVISOR_AGENT"
        
    def ask_aura(self, user_id: int, query: str, city_override: str = None) -> Dict[str, Any]:
        """
        Entry point for conversational queries.
        In a full implementation, this uses Gemini's Function Calling.
        For robust hackathon reliability, we implement deterministic intent routing 
        with fallback LLM formatting.
        """
        logger.info(f"[Supervisor] Processing query for User {user_id}: '{query}'")
        
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"response": "Error: User profile not found.", "tools_used": []}
                
            # Use city_override from request (user's currently selected city on frontend)
            # Fall back to user's DB city, then to Delhi
            city = city_override or user.city or "Delhi"
            tools_used = []
            
            # Context gathering
            latest_aq = get_latest_air_quality(db, city)
            current_risk = (latest_aq.aqi / 5.0) if (latest_aq and latest_aq.aqi) else 50.0
            
            # --- INTENT ROUTING ---
            query_lower = query.lower()
            
            # Intent 1: "What if I can't change the time?" (Constraint Optimization)
            if "can't change" in query_lower or "cannot change" in query_lower or "fixed time" in query_lower:
                log_agent_action(db, self.name, "ORCHESTRATE", "optimize_activity", "User requested constraint optimization", "SUCCESS")
                tools_used.append("optimize_activity")
                
                opt_res = optimize_activity(
                    env_risk=current_risk,
                    target_max_exposure=50.0,
                    base_activity=user.primary_activity,
                    base_duration=int(user.outdoor_hours * 60)
                )
                
                response = f"I understand you cannot change your {user.primary_activity} time.\n\n"
                response += f"Your current estimated exposure is **{opt_res['original_exposure']}**. To lower this to a safer level, I recommend:\n"
                for s in opt_res["suggestions"]:
                    response += f"- {s}\n"
                if not opt_res["suggestions"]:
                    response += "- Please reconsider going out, or minimize your exposure to 15 minutes."
            
            # Intent 2: "When should I go?" / "Can I go now?" (Simulation)
            else:
                log_agent_action(db, self.name, "ORCHESTRATE", "simulate_alternatives", "User requested time simulation", "SUCCESS")
                tools_used.append("simulate_alternatives")
                
                sim_res = simulate_alternatives(
                    city=city, 
                    db=db, 
                    activity=user.primary_activity, 
                    duration_minutes=int(user.outdoor_hours * 60), 
                    mask_usage=user.mask_usage,
                    current_env_risk=current_risk
                )
                
                curr = sim_res["current_plan"]
                best = sim_res["best_alternative"]
                
                response = f"Based on current data in {city} (AQI: {latest_aq.aqi if latest_aq else 'Unknown'}), "
                
                if curr['estimated_exposure'] >= 75.0:
                    response += f"your exposure risk if you go NOW is **{curr['risk_level']}** ({curr['estimated_exposure']}). I strongly advise against it.\n\n"
                else:
                    response += f"your exposure risk if you go NOW is **{curr['risk_level']}** ({curr['estimated_exposure']}).\n\n"
                
                if best and best['diff_from_current'] < -5.0:
                    response += f"**AURA Recommendation:**\nMove your activity to **{best['time_label']}**. "
                    response += f"The forecasted AQI drops to {best['predicted_aqi']}, which will lower your personal exposure by {abs(best['diff_from_current'])} points."
                else:
                    response += "The forecast does not show significant improvement in the next 12 hours. Please wear a mask if you must go outside."

            return {
                "response": response,
                "tools_used": tools_used,
                "agent": "AURA Supervisor"
            }
