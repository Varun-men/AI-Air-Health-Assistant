"""
OpenAQ V3 API client for AURA.
Fetches live air quality measurements for Indian cities.

Docs: https://docs.openaq.org/
"""
import requests
import logging
from datetime import datetime, timezone
from typing import Optional
from src.utils.config import OPENAQ_API_KEY, OPENAQ_BASE_URL

logger = logging.getLogger(__name__)

# City → approximate lat/lon for location-based queries
CITY_COORDS = {
    "Delhi":     (28.6139, 77.2090),
    "Mumbai":    (19.0760, 72.8777),
    "Lucknow":   (26.8467, 80.9462),
    "Jaipur":    (26.9124, 75.7873),
    "Bangalore": (12.9716, 77.5946),
    "Kolkata":   (22.5726, 88.3639),
    "Chennai":   (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
}

# OpenAQ parameter → our column name
PARAM_MAP = {
    "pm25": "pm25",
    "pm10": "pm10",
    "no2":  "no2",
    "so2":  "so2",
    "co":   "co",
    "o3":   "o3",
}


class OpenAQClient:
    """Fetches live air-quality data from the OpenAQ V3 API."""

    def __init__(self):
        self.headers = {}
        if OPENAQ_API_KEY:
            self.headers["X-API-Key"] = OPENAQ_API_KEY
        else:
            logger.warning("OPENAQ_API_KEY not set — rate limits may apply.")

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        url = f"{OPENAQ_BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenAQ HTTP error: {e} — {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAQ request failed: {e}")
        return None

    def fetch_latest(self, city: str, lat: float = None, lon: float = None) -> Optional[dict]:
        """
        Fetch the latest measurements for a city.
        Supports any city worldwide by geocoding coordinates dynamically,
        querying OpenAQ, and falling back to OpenWeatherMap Air Pollution API.
        """
        import os
        weather_key = os.getenv("WEATHER_API_KEY", "") or os.getenv("OPENWEATHER_API_KEY", "")

        # 1. Resolve coordinates
        if lat is None or lon is None:
            coords = CITY_COORDS.get(city)
            if coords:
                lat, lon = coords
            elif weather_key:
                try:
                    geo_resp = requests.get(
                        "https://api.openweathermap.org/geo/1.0/direct",
                        params={"q": city, "limit": 1, "appid": weather_key},
                        timeout=8
                    )
                    if geo_resp.status_code == 200 and geo_resp.json():
                        first = geo_resp.json()[0]
                        lat, lon = float(first["lat"]), float(first["lon"])
                        logger.info(f"Geocoded '{city}' -> ({lat}, {lon})")
                except Exception as e:
                    logger.warning(f"Geocoding failed for {city}: {e}")

        if lat is None or lon is None:
            logger.warning(f"Could not determine coordinates for '{city}'.")
            return None

        # 2. Try OpenAQ V3 by coordinates
        data = self._get("measurements", {
            "coordinates": f"{lat},{lon}",
            "radius": 35000,        # 35 km radius
            "limit": 100,
            "order_by": "datetime",
            "sort": "desc",
        })

        if data and data.get("results"):
            parsed = self._parse(data["results"], city, lat, lon)
            if parsed.get("pm25") is not None or parsed.get("pm10") is not None:
                return parsed

        # 3. Fallback to OpenWeatherMap Air Pollution API (covers 100% of the globe)
        if weather_key:
            try:
                air_resp = requests.get(
                    "http://api.openweathermap.org/data/2.5/air_pollution",
                    params={"lat": lat, "lon": lon, "appid": weather_key},
                    timeout=10
                )
                if air_resp.status_code == 200:
                    air_data = air_resp.json()
                    if air_data.get("list"):
                        item = air_data["list"][0]
                        comp = item.get("components", {})
                        pm25 = round(float(comp.get("pm2_5", 25.0)), 2)
                        pm10 = round(float(comp.get("pm10", 35.0)), 2)
                        no2  = round(float(comp.get("no2", 15.0)), 2)
                        so2  = round(float(comp.get("so2", 5.0)), 2)
                        co   = round(float(comp.get("co", 400.0) / 1000.0), 2)  # mg/m3
                        o3   = round(float(comp.get("o3", 30.0)), 2)
                        dt_val = item.get("dt")
                        src_ts = datetime.fromtimestamp(dt_val, timezone.utc).replace(tzinfo=None) if dt_val else None

                        return {
                            "city": city,
                            "latitude": lat,
                            "longitude": lon,
                            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
                            "source": "OpenWeather Air (Global)",
                            "fetched_at": datetime.utcnow(),
                            "source_timestamp": src_ts,
                            "pm25": pm25, "pm10": pm10, "no2": no2,
                            "so2": so2,   "co": co,     "o3": o3,
                            "aqi": self._pm25_to_aqi(pm25),
                        }
            except Exception as e:
                logger.error(f"OpenWeather air pollution fallback failed for {city}: {e}")

        logger.warning(f"No air quality data available from any provider for {city}.")
        return None


    def _parse(self, results: list, city: str, lat: float, lon: float) -> dict:
        """Collapse per-parameter rows into one flat record."""
        record = {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
            "source": "OpenAQ",
            "fetched_at": datetime.utcnow(),
            "source_timestamp": None,
            "pm25": None, "pm10": None, "no2": None,
            "so2": None,  "co":   None, "o3":  None,
            "aqi": None,
        }

        for r in results:
            param = r.get("parameter", "").lower()
            value = r.get("value")
            unit  = r.get("unit", "")

            # Skip negative / clearly erroneous sensor values
            if value is None or value < 0:
                continue

            if param in PARAM_MAP and record[PARAM_MAP[param]] is None:
                record[PARAM_MAP[param]] = round(float(value), 2)

                # Capture source timestamp from first valid reading
                if record["source_timestamp"] is None:
                    raw_ts = r.get("date", {}).get("utc")
                    if raw_ts:
                        try:
                            record["source_timestamp"] = datetime.fromisoformat(
                                raw_ts.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                        except ValueError:
                            pass

        # Simple AQI estimate from PM2.5 (US EPA linear approximation)
        if record["pm25"] is not None:
            record["aqi"] = self._pm25_to_aqi(record["pm25"])

        return record

    @staticmethod
    def _pm25_to_aqi(pm25: float) -> float:
        """
        Convert PM2.5 µg/m³ to approximate AQI using US EPA breakpoints.
        This is an estimate — not a replacement for official AQI calculations.
        """
        breakpoints = [
            (0,    12.0,   0,   50),
            (12.1, 35.4,  51,  100),
            (35.5, 55.4, 101,  150),
            (55.5, 150.4,151,  200),
            (150.5,250.4,201,  300),
            (250.5,350.4,301,  400),
            (350.5,500.4,401,  500),
        ]
        for c_lo, c_hi, i_lo, i_hi in breakpoints:
            if c_lo <= pm25 <= c_hi:
                aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
                return round(aqi, 1)
        return round(min(pm25 * 2.1, 500), 1)   # fallback cap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenAQClient()
    result = client.fetch_latest("Delhi")
    import json
    print(json.dumps(result, indent=2, default=str))
