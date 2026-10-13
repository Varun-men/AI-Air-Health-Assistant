"""
OpenWeatherMap API client for AURA.
Fetches current weather (temperature, humidity, wind speed) for a city.

Docs: https://openweathermap.org/current
"""
import requests
import logging
from datetime import datetime, timezone
from typing import Optional
from src.utils.config import WEATHER_API_KEY, WEATHER_BASE_URL

logger = logging.getLogger(__name__)


class WeatherClient:
    """Fetches current weather data from OpenWeatherMap."""

    def __init__(self):
        if not WEATHER_API_KEY:
            logger.warning("WEATHER_API_KEY not set — weather data will be unavailable.")

    def fetch_current(self, city: str, lat: float = None, lon: float = None) -> Optional[dict]:
        """
        Fetch current weather for a city or coordinates worldwide.
        Returns a partial dict with keys: temperature, humidity, wind_speed
        to be merged with an AirQuality record, or None on failure.
        """
        import os
        key = WEATHER_API_KEY or os.getenv("WEATHER_API_KEY", "") or os.getenv("OPENWEATHER_API_KEY", "")
        if not key:
            return None

        params = {"appid": key, "units": "metric"}
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        else:
            params["q"] = city

        try:
            resp = requests.get(
                f"{WEATHER_BASE_URL}/weather",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            return {
                "temperature": data.get("main", {}).get("temp"),
                "humidity":    data.get("main", {}).get("humidity"),
                "wind_speed":  data.get("wind", {}).get("speed"),
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error for {city}: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WeatherClient()
    result = client.fetch_current("Delhi")
    print(result)
