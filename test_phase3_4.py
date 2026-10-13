from src.database.connection import init_db, get_db
from src.exposure.engine import calculate_exposure
from src.exposure.simulator import simulate_alternatives, optimize_activity
import json

print("--- Testing Phase 3 & 4: Exposure Engine & Simulator ---")

base_env_risk = 75.0 # HIGH environmental risk

# 1. Test Exposure Engine Multipliers
print("\n1. EXPOSURE ENGINE TEST")
cases = [
    ("Walking", 30, False),
    ("Running", 60, False),
    ("Running", 60, True),  # With mask
    ("Indoor", 120, False)
]

for act, dur, mask in cases:
    res = calculate_exposure(base_env_risk, act, dur, mask)
    print(f"[{act} for {dur}m, Mask:{mask}] -> Exposure: {res['estimated_exposure']} ({res['risk_level']})")


# 2. Test Simulator
print("\n2. WHAT-IF SIMULATOR TEST")
init_db()
with get_db() as db:
    city = "Delhi"
    sim_res = simulate_alternatives(city, db, activity="Running", duration_minutes=90, current_env_risk=base_env_risk)
    
    print("CURRENT PLAN:")
    curr = sim_res["current_plan"]
    print(f"  Time: {curr['time_label']} | Exposure: {curr['estimated_exposure']} ({curr['risk_level']})")
    
    print("ALTERNATIVES:")
    for alt in sim_res["alternatives"]:
        print(f"  Time: {alt['time_label']} | AQI: {alt['predicted_aqi']} | Exposure: {alt['estimated_exposure']} (Diff: {alt['diff_from_current']})")

    best = sim_res["best_alternative"]
    if best:
        print(f"\nBEST ALTERNATIVE: Go at {best['time_label']} to reduce exposure by {abs(best['diff_from_current'])} points.")


# 3. Test Activity Optimizer (If time is fixed)
print("\n3. CONSTRAINT OPTIMIZER (Time is fixed)")
opt_res = optimize_activity(base_env_risk, target_max_exposure=50.0, base_activity="Running", base_duration=60)
print(f"Original Exposure: {opt_res['original_exposure']}")
print("To get below 50.0 exposure without changing time, you can:")
for sugg in opt_res["suggestions"]:
    print(f"  - {sugg}")

print("\nPhase 3 & 4 Services: READY")
