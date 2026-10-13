import requests
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAQClient:
    """
    Client for fetching live air quality data from the OpenAQ API.
    """
    BASE_URL = "https://api.openaq.org/v2"

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.headers = {}
        if self.api_key:
            self.headers['X-API-Key'] = self.api_key

    def fetch_latest_measurements(self, city="Delhi", limit=10):
        """
        Fetches the latest measurements for a specific city.
        """
        url = f"{self.BASE_URL}/measurements"
        params = {
            "city": city,
            "limit": limit,
            "order_by": "datetime",
            "sort": "desc"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from OpenAQ: {e}")
            return []

    def parse_measurements_to_db_format(self, results, city="Delhi"):
        """
        Parses OpenAQ results into a flattened dictionary matching the AirQuality schema.
        Note: OpenAQ returns individual rows per parameter (e.g., one row for PM2.5, one for PM10).
        This function aggregates them into a single reading for the current timestamp.
        """
        if not results:
            return None
            
        record = {
            "city": city,
            "timestamp": datetime.utcnow(),
            "source": "OpenAQ",
            "pm25": None,
            "pm10": None,
            "no2": None,
            "so2": None,
            "co": None,
            "o3": None
        }

        for res in results:
            parameter = res.get('parameter')
            value = res.get('value')
            if parameter == 'pm25':
                record['pm25'] = value
            elif parameter == 'pm10':
                record['pm10'] = value
            elif parameter == 'no2':
                record['no2'] = value
            elif parameter == 'so2':
                record['so2'] = value
            elif parameter == 'co':
                record['co'] = value
            elif parameter == 'o3':
                record['o3'] = value

        # Simple AQI estimation based on PM2.5 (for demonstration)
        # In the full app, the ML model handles risk prediction
        if record['pm25']:
            record['aqi'] = record['pm25'] * 2.5 # placeholder logic
            
        return record

if __name__ == "__main__":
    client = OpenAQClient()
    print("Fetching live data for Delhi...")
    results = client.fetch_latest_measurements(city="Delhi", limit=50)
    parsed = client.parse_measurements_to_db_format(results)
    print(json.dumps(parsed, indent=2, default=str))
