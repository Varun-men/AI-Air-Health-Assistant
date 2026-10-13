"""
AURA FastAPI Backend
Serves the frontend and exposes REST endpoints for live data, ML predictions,
exposure calculations, forecasts, and agent activity.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import logging

# Initialise AURA backend
from src.database.connection import init_db, get_db
from src.database.crud import (
    get_latest_air_quality, get_recent_air_quality,
    get_active_alerts, get_recent_agent_logs
)
from src.database.models import User
from src.ml.risk_model import predict_risk
from src.ml.forecast_model import forecast_air_quality
from src.ml.anomaly_model import detect_anomaly
from src.ml.explainability import explain_prediction
from src.exposure.engine import calculate_exposure
from src.exposure.simulator import simulate_alternatives, optimize_activity
from src.agents.supervisor import SupervisorAgent
from src.data.scheduler import start_scheduler, trigger_now
from src.utils.config import SUPPORTED_CITIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura.api")

init_db()
start_scheduler()

app = FastAPI(title="AURA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

# ── Pydantic models ───────────────────────────────────────────────────────────

class ExposureRequest(BaseModel):
    base_risk_score: float
    activity: str = "Walking"
    duration_minutes: int = 60
    mask_usage: bool = False

class SimulateRequest(BaseModel):
    city: str = "Delhi"
    activity: str = "Walking"
    duration_minutes: int = 60
    mask_usage: bool = False

class ChatRequest(BaseModel):
    user_id: int = 1
    query: str
    city: str = "Delhi"

class OptimizeRequest(BaseModel):
    env_risk: float = 75.0
    target_max_exposure: float = 50.0
    base_activity: str = "Running"
    base_duration: int = 60

class RefreshRequest(BaseModel):
    city: str | None = None

class LoginRequest(BaseModel):
    email_or_name: str
    password: str = "password"

class RegisterRequest(BaseModel):
    name: str
    city: str = "Delhi"
    email: str = ""
    password: str = "password"
    primary_activity: str = "Running"
    mask_usage: bool = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj

def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    return {
        "city":        row.city,
        "aqi":         float(row.aqi) if row.aqi is not None else None,
        "pm25":        float(row.pm25) if row.pm25 is not None else None,
        "pm10":        float(row.pm10) if row.pm10 is not None else None,
        "no2":         float(row.no2) if row.no2 is not None else None,
        "so2":         float(row.so2) if row.so2 is not None else None,
        "co":          float(row.co) if row.co is not None else None,
        "o3":          float(row.o3) if row.o3 is not None else None,
        "temperature": float(row.temperature) if row.temperature is not None else None,
        "humidity":    float(row.humidity) if row.humidity is not None else None,
        "wind_speed":  float(row.wind_speed) if row.wind_speed is not None else None,
        "timestamp":   row.timestamp.isoformat() if row.timestamp else None,
        "source":      row.source,
    }

def _minutes_ago(ts_str: str | None) -> int | None:
    if not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str)
        return int((datetime.utcnow() - ts).total_seconds() / 60)
    except Exception:
        return None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "AURA API running. Frontend not found at frontend/index.html."}

@app.get("/api/cities")
def get_cities():
    return {"cities": SUPPORTED_CITIES}

@app.get("/api/live/{city}")
def get_live_data(city: str):
    """Main dashboard endpoint — returns all live data + ML analysis for a city worldwide."""
    clean_city = city.strip()
    with get_db() as db:
        row = get_latest_air_quality(db, clean_city)

        # On-demand live fetch for ANY city in the world if missing or stale
        if row is None:
            try:
                from src.agents.data_agent import DataAgent
                agent = DataAgent()
                rec = agent.run_for_city(clean_city)
                if rec:
                    row = get_latest_air_quality(db, clean_city)
            except Exception as e:
                logger.error(f"[API] On-demand fetch failed for {clean_city}: {e}")

        d = _row_to_dict(row)

        if d is None:
            return JSONResponse({
                "city": clean_city, "data_available": False,
                "message": f"Unable to retrieve sensor telemetry for '{clean_city}'. Please check the city name."
            })

        # ML analysis
        risk_result  = predict_risk(d)
        anomaly      = detect_anomaly(city, d, db)
        explanation  = explain_prediction(d)
        forecasts    = forecast_air_quality(city, db)

        # Alerts
        alerts = get_active_alerts(db)
        alert_list = [
            {"type": a.alert_type, "severity": a.severity, "message": a.message[:200]}
            for a in alerts if a.alert_type
        ][:3]

        mins = _minutes_ago(d["timestamp"])
        freshness = "live" if (mins or 999) < 30 else ("stale" if (mins or 999) < 120 else "very_stale")

        return to_native({
            "city":           city,
            "data_available": True,
            "freshness":      freshness,
            "updated_mins_ago": mins,
            "environment":    d,
            "risk": {
                "score":     risk_result.get("risk_score"),
                "level":     risk_result.get("risk_level"),
                "available": bool(risk_result.get("prediction_available", False)),
            },
            "anomaly": {
                "is_anomaly": bool(anomaly.get("is_anomaly", False)),
                "severity":   anomaly.get("severity"),
                "reason":     anomaly.get("reason"),
                "z_score":    anomaly.get("z_score"),
            },
            "explanation": {
                "top_factors":      explanation.get("top_factors"),
                "explanation_text": explanation.get("explanation_text"),
                "pct_contributions": explanation.get("pct_contributions", {}),
            },
            "forecast": [
                {
                    "horizon":       f["horizon"],
                    "predicted_aqi": f["predicted_aqi"],
                    "risk_level":    f["risk_level"],
                    "risk_score":    f["risk_score"],
                }
                for f in forecasts
            ],
            "alerts": alert_list,
        })

@app.post("/api/exposure")
def get_exposure(req: ExposureRequest):
    result = calculate_exposure(
        req.base_risk_score, req.activity, req.duration_minutes, req.mask_usage
    )
    return to_native(result)

@app.post("/api/simulate")
def get_simulation(req: SimulateRequest):
    with get_db() as db:
        row = get_latest_air_quality(db, req.city)
        current_risk = min((row.aqi / 5.0) if (row and row.aqi) else 50.0, 100.0)
        result = simulate_alternatives(
            req.city, db, req.activity, req.duration_minutes,
            mask_usage=req.mask_usage, current_env_risk=current_risk
        )
        return to_native(result)

@app.post("/api/optimize")
def get_optimization(req: OptimizeRequest):
    return to_native(optimize_activity(
        env_risk=req.env_risk,
        target_max_exposure=req.target_max_exposure,
        base_activity=req.base_activity,
        base_duration=req.base_duration
    ))

@app.post("/api/chat")
def chat(req: ChatRequest):
    uid = req.user_id if req.user_id and req.user_id > 0 else 1
    sup = SupervisorAgent()
    return sup.ask_aura(uid, req.query, city_override=req.city)

@app.get("/api/geocode")
def geocode_city(q: str):
    """Proxy to OpenWeatherMap geocoding API for global city search."""
    import os, requests
    api_key = os.getenv("WEATHER_API_KEY", "") or os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {"results": []}
    url = "https://api.openweathermap.org/geo/1.0/direct"
    try:
        resp = requests.get(url, params={"q": q, "limit": 7, "appid": api_key}, timeout=5.0)
        if resp.status_code != 200:
            return {"results": []}
        data = resp.json()
        results = [
            {
                "name": item.get("name", ""),
                "country": item.get("country", ""),
                "state": item.get("state", ""),
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "label": f"{item.get('name', '')}{', ' + item.get('state','') if item.get('state') else ''}, {item.get('country','')}"
            }
            for item in data if isinstance(item, dict)
        ]
        return {"results": results}
@app.get("/api/reverse_geocode")
def reverse_geocode(lat: float, lon: float):
    """Resolve lat/lon to city name using OpenWeatherMap reverse geocoding API."""
    import os, requests
    api_key = os.getenv("WEATHER_API_KEY", "") or os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {"city": "Unknown", "country": ""}
    url = "https://api.openweathermap.org/geo/1.0/reverse"
    try:
        resp = requests.get(url, params={"lat": lat, "lon": lon, "limit": 1, "appid": api_key}, timeout=5.0)
        if resp.status_code == 200 and resp.json():
            item = resp.json()[0]
            name = item.get("name", "Unknown")
            country = item.get("country", "")
            return {"city": name, "country": country, "label": f"{name}, {country}"}
    except Exception as e:
        logger.warning(f"Reverse geocode error: {e}")
    return {"city": "Unknown", "country": ""}




@app.get("/api/activity")
def get_activity(limit: int = 20):
    with get_db() as db:
        logs = get_recent_agent_logs(db, limit=limit)
        return {
            "logs": [
                {
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                    "agent":     l.agent_name,
                    "action":    l.action,
                    "tool":      l.tool_name,
                    "result":    l.result_summary,
                    "status":    l.status,
                }
                for l in logs
            ]
        }

@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    with get_db() as db:
        user = db.query(User).filter(
            (User.name.ilike(req.email_or_name.strip()))
        ).first()
        if not user:
            user = User(
                name=req.email_or_name.strip().title(),
                city="Delhi",
                primary_activity="Running",
                mask_usage=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        return to_native({
            "status": "success",
            "user": {
                "id": user.id,
                "name": user.name,
                "city": user.city,
                "primary_activity": user.primary_activity or "Walking",
                "mask_usage": bool(user.mask_usage),
                "outdoor_hours": user.outdoor_hours or 1.0
            },
            "token": f"aura_token_{user.id}"
        })

@app.post("/api/auth/register")
def auth_register(req: RegisterRequest):
    with get_db() as db:
        user = User(
            name=req.name.strip(),
            city=req.city,
            primary_activity=req.primary_activity,
            mask_usage=req.mask_usage,
            outdoor_hours=1.0
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return to_native({
            "status": "success",
            "user": {
                "id": user.id,
                "name": user.name,
                "city": user.city,
                "primary_activity": user.primary_activity,
                "mask_usage": bool(user.mask_usage),
                "outdoor_hours": 1.0
            },
            "token": f"aura_token_{user.id}"
        })

@app.post("/api/refresh")
def refresh_data(req: RefreshRequest):
    """Manually trigger a data fetch cycle."""
    try:
        trigger_now()
        return {"status": "ok", "message": "Data refresh triggered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend static files (CSS, JS, images)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
