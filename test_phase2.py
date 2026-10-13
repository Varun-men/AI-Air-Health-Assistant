from src.database.connection import init_db, get_db
from src.ml.risk_model import predict_risk, get_feature_columns
from src.ml.forecast_model import forecast_air_quality
from src.ml.anomaly_model import detect_anomaly
from src.ml.explainability import explain_prediction

print("--- Testing Phase 2 ML Services ---")

# Mock data simulating an air_quality row
mock_data = {
    "aqi": 214,
    "pm25": 164.5,
    "pm10": 210.0,
    "no2": 45.2,
    "so2": 12.1,
    "co": 1.2,
    "o3": 34.5,
    "temperature": 32.5,
    "humidity": 65.0,
    "wind_speed": 12.4
}
city = "Delhi"

print("\n1. Risk Model")
risk_result = predict_risk(mock_data)
print(f"Risk Score: {risk_result.get('risk_score')}")
print(f"Risk Level: {risk_result.get('risk_level')}")
print(f"Features mapped: {len(risk_result.get('input_features', {}))}")

print("\n2. Explainability Service")
explain_result = explain_prediction(mock_data)
print(f"Method: {explain_result.get('method')}")
print(f"Top Factors: {explain_result.get('top_factors')}")
print(f"Explanation: {explain_result.get('explanation_text')}")

init_db()

with get_db() as db:
    print("\n3. Forecast Model (using fallback/empty db)")
    forecast_result = forecast_air_quality(city, db)
    for f in forecast_result:
        print(f"  {f['horizon']}: AQI {f['predicted_aqi']} ({f['risk_level']})")

    print("\n4. Anomaly Model")
    anomaly_result = detect_anomaly(city, mock_data, db)
    print(f"Is Anomaly: {anomaly_result.get('is_anomaly')}")
    print(f"Severity: {anomaly_result.get('severity')}")
    print(f"Reason: {anomaly_result.get('reason')}")

print("\nPhase 2 ML Services: READY")
