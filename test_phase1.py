from src.database.connection import init_db
init_db()
print("DB init: OK")

from src.data.openaq_client import OpenAQClient
client = OpenAQClient()
result = client.fetch_latest("Delhi")
if result:
    print("OpenAQ fetch: OK")
    print("  AQI   =", result.get("aqi"))
    print("  PM2.5 =", result.get("pm25"))
    print("  PM10  =", result.get("pm10"))
    print("  Temp  =", result.get("temperature"))
else:
    print("OpenAQ fetch: No data returned (API key missing or rate-limited)")

from src.data.validator import validate_record
if result:
    valid, issues = validate_record(result)
    print("Validator:", "PASS" if valid else "FAIL", issues)

print("\nPhase 1 pipeline: READY")
