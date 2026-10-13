from src.database.connection import init_db, get_db
from src.database.models import User
from src.database.crud import upsert_user, insert_air_quality, get_recent_agent_logs, get_active_alerts
from src.agents.decision_agent import DecisionAgent
from src.agents.supervisor import SupervisorAgent
from datetime import datetime

print("--- Testing Phase 6 & 7: The Agentic Layer ---")

init_db()

# ─── Setup: seed user + high-AQI reading (separate session, committed first) ───
with get_db() as db:
    upsert_user(db, {
        "name": "TestUser", "city": "Delhi", "latitude": 28.6, "longitude": 77.2,
        "outdoor_hours": 1.0, "primary_activity": "Running",
        "usual_outdoor_start": "19:00", "mask_usage": False
    })
    insert_air_quality(db, {
        "city": "Delhi", "timestamp": datetime.now(),
        "aqi": 280.0, "pm25": 190.0, "pm10": 240.0, "source": "test"
    })
    # Capture user id before session closes
    user = db.query(User).filter_by(name="TestUser").first()
    uid = user.id
    print(f"User '{user.name}' ready (id={uid})")

# ─── 1. Decision Agent (runs its own session internally) ───────────────────────
print("\n1. DECISION AGENT (Proactive)")
da = DecisionAgent()
da.evaluate_proactive_alerts("Delhi")

with get_db() as db:
    alerts = get_active_alerts(db)
    print(f"Generated Alerts: {len(alerts)}")
    if alerts:
        print(f"Alert TYPE: {alerts[0].alert_type}")
        print(f"Alert SEVERITY: {alerts[0].severity}")
        print(f"Alert MSG (truncated): {alerts[0].message.encode('ascii', errors='replace').decode()[:120]}")

# ─── 2. Supervisor Agent (also uses its own session internally) ────────────────
print("\n2. SUPERVISOR AGENT (Ask AURA)")
sup = SupervisorAgent()

print("\nQuery A: 'Should I go running now?'")
ans_a = sup.ask_aura(uid, "Should I go running now?")
print(f"Tools used: {ans_a['tools_used']}")
print(f"Response:\n{ans_a['response']}")

print("\nQuery B: 'I cannot change the time, what should I do?'")
ans_b = sup.ask_aura(uid, "I cannot change the time, what should I do?")
print(f"Tools used: {ans_b['tools_used']}")
print(f"Response:\n{ans_b['response']}")

# ─── 3. Agent Observability ────────────────────────────────────────────────────
print("\n3. AGENT OBSERVABILITY (agent_logs)")
with get_db() as db:
    logs = get_recent_agent_logs(db, limit=5)
    for log in logs:
        print(f"  [{log.agent_name}] {log.action} -> {log.result_summary}")

print("\nPhase 6 & 7 Services: READY")
