"""
Central configuration loader for AURA.
Reads all settings from environment variables (loaded via .env).
"""
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─── API Keys ──────────────────────────────────────────────────────────────────
OPENAQ_API_KEY: str = os.getenv("OPENAQ_API_KEY", "")
WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ─── Database ──────────────────────────────────────────────────────────────────
# Fallback to SQLite for local dev if MySQL URL is not set
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///aura_dev.db"
)

# ─── Scheduler ─────────────────────────────────────────────────────────────────
REFRESH_INTERVAL_MINUTES: int = int(os.getenv("REFRESH_INTERVAL_MINUTES", "15"))

# ─── App ───────────────────────────────────────────────────────────────────────
DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
APP_ENV: str = os.getenv("APP_ENV", "development")

# ─── Supported cities ──────────────────────────────────────────────────────────
SUPPORTED_CITIES = [
    "Delhi",
    "Mumbai",
    "Lucknow",
    "Jaipur",
    "Bangalore",
    "Kolkata",
    "Chennai",
    "Hyderabad",
]

# ─── OpenAQ V3 API ─────────────────────────────────────────────────────────────
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# ─── OpenWeatherMap ────────────────────────────────────────────────────────────
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
