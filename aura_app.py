"""
AURA — Autonomous Urban Risk Agent
Main Streamlit Application (8 Pages)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64, time, json

# ── DB & Backend ──────────────────────────────────────────────────────────────
from src.database.connection import init_db, get_db
from src.database.models import User
from src.database.crud import (
    get_latest_air_quality, get_recent_air_quality, get_recent_agent_logs,
    get_active_alerts, upsert_user, log_agent_action
)
from src.ml.risk_model import predict_risk
from src.ml.forecast_model import forecast_air_quality
from src.ml.anomaly_model import detect_anomaly
from src.ml.explainability import explain_prediction
from src.exposure.engine import calculate_exposure
from src.exposure.simulator import simulate_alternatives, optimize_activity
from src.agents.supervisor import SupervisorAgent
from src.agents.decision_agent import DecisionAgent
from src.data.scheduler import start_scheduler, trigger_now
from src.utils.config import SUPPORTED_CITIES

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

if "scheduler_started" not in st.session_state:
    start_scheduler()
    st.session_state.scheduler_started = True

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA — Autonomous Urban Risk Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Logo ──────────────────────────────────────────────────────────────────────
try:
    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    logo_b64 = ""

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  (preserving original premium dark design + new AURA additions)
# ═══════════════════════════════════════════════════════════════════════════════
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
<style>
:root {
    --bg-primary: #06080f;
    --bg-secondary: #0c1021;
    --bg-card: rgba(15, 20, 40, 0.65);
    --bg-card-hover: rgba(20, 28, 58, 0.75);
    --border-card: rgba(100, 120, 255, 0.12);
    --border-glow: rgba(139, 92, 246, 0.3);
    --text-primary: #f0f2ff;
    --text-secondary: rgba(200, 210, 255, 0.7);
    --text-muted: rgba(160, 170, 220, 0.5);
    --accent-violet: #8b5cf6;
    --accent-cyan: #22d3ee;
    --accent-pink: #ec4899;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --gradient-primary: linear-gradient(135deg, #8b5cf6, #22d3ee);
    --shadow-glow: 0 0 40px rgba(139, 92, 246, 0.15);
    --radius-lg: 20px; --radius-md: 14px; --radius-sm: 10px;
    --transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
* { font-family: 'Inter', sans-serif !important; }
h1,h2,h3,h4,h5,h6 { font-family: 'Outfit', sans-serif !important; }
.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139,92,246,0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(34,211,238,0.06), transparent);
    color: var(--text-primary);
}
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: rgba(6,8,15,0.85) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-card);
}
.block-container { padding: 1rem 2rem 4rem 2rem; max-width: 1300px; }

/* ── NAVBAR ── */
.aura-navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: rgba(10,14,30,0.80);
    backdrop-filter: blur(24px);
    border: 1px solid var(--border-card); border-radius: 16px;
    margin-bottom: 20px;
}
.aura-logo {
    font-family: 'Outfit', sans-serif !important;
    font-size: 22px; font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.aura-tagline { font-size: 11px; color: var(--text-muted); margin-top: 1px; }

/* ── CARDS ── */
.aura-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
    transition: var(--transition);
}
.aura-card:hover { border-color: var(--border-glow); background: var(--bg-card-hover); }

/* ── RISK BADGE ── */
.risk-badge {
    display: inline-block; padding: 6px 18px; border-radius: 50px;
    font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
}
.risk-low      { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.risk-moderate { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.risk-high     { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.risk-veryhigh { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── METRIC CARD ── */
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border-card);
    border-radius: var(--radius-md); padding: 20px; text-align: center;
}
.metric-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 42px; font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-label { font-size: 12px; color: var(--text-muted); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-sub   { font-size: 13px; color: var(--text-secondary); margin-top: 4px; }

/* ── ALERT BANNER ── */
.alert-banner {
    padding: 14px 20px; border-radius: var(--radius-md);
    margin-bottom: 12px; font-size: 14px;
}
.alert-warning  { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #f59e0b; }
.alert-critical { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #ef4444; }
.alert-info     { background: rgba(34,211,238,0.08); border: 1px solid rgba(34,211,238,0.2); color: #22d3ee; }
.alert-success  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #10b981; }

/* ── AGENT LOG ── */
.agent-log-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; border-radius: 10px;
    margin-bottom: 6px; background: rgba(15,20,40,0.5);
    border: 1px solid var(--border-card); font-size: 13px;
}
.agent-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-success { background: #10b981; box-shadow: 0 0 6px #10b981; }
.dot-warn    { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
.dot-fail    { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* ── CHAT ── */
.chat-msg-user  { background: rgba(139,92,246,0.15); border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 8px 0; border: 1px solid rgba(139,92,246,0.25); }
.chat-msg-aura  { background: rgba(34,211,238,0.08); border-radius: 12px 12px 12px 4px; padding: 12px 16px; margin: 8px 0; border: 1px solid rgba(34,211,238,0.2); }
.chat-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }

/* ── SECTION HEADING ── */
.section-title { font-size: 18px; font-weight: 700; color: var(--text-primary); margin-bottom: 16px; }
.section-sub   { font-size: 13px; color: var(--text-muted); margin-top: -12px; margin-bottom: 16px; }

/* ── STALE DATA ── */
.stale-badge { font-size: 11px; color: var(--accent-amber); background: rgba(245,158,11,0.1); border-radius: 6px; padding: 2px 8px; }
.live-badge  { font-size: 11px; color: var(--accent-emerald); background: rgba(16,185,129,0.1); border-radius: 6px; padding: 2px 8px; }

/* Streamlit widget styling */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #22d3ee) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 24px !important; transition: var(--transition) !important;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
div[data-testid="stSelectbox"] > div { background: var(--bg-card) !important; border-color: var(--border-card) !important; border-radius: 10px !important; }
div[data-testid="stSlider"] .rc-slider-track { background: var(--accent-violet) !important; }
</style>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def risk_css(level: str) -> str:
    m = {"LOW": "risk-low", "MODERATE": "risk-moderate", "HIGH": "risk-high", "VERY HIGH": "risk-veryhigh"}
    return m.get(level, "risk-moderate")

def aqi_color(aqi):
    if aqi is None: return "#888"
    if aqi <= 50:   return "#10b981"
    if aqi <= 100:  return "#f59e0b"
    if aqi <= 150:  return "#f97316"
    if aqi <= 200:  return "#ef4444"
    if aqi <= 300:  return "#a855f7"
    return "#7f1d1d"

def freshness_label(ts):
    if ts is None: return '<span class="stale-badge">⚠ No data</span>'
    diff = datetime.utcnow() - ts
    mins = int(diff.total_seconds() / 60)
    if mins < 30:  return f'<span class="live-badge">● Live · {mins}m ago</span>'
    if mins < 120: return f'<span class="stale-badge">⚠ {mins}m ago</span>'
    return f'<span class="stale-badge">⚠ Stale · {mins//60}h ago</span>'

def navbar(page: str):
    pages = ["Dashboard", "Environment", "My Exposure", "What-If", "Forecast", "Ask AURA", "Agent Activity", "Profile"]
    links = " ".join(
        f'<span style="padding:6px 14px;border-radius:8px;font-size:13px;font-weight:600;'
        f'{"background:rgba(139,92,246,0.2);color:#a78bfa;border:1px solid rgba(139,92,246,0.3)" if p==page else "color:rgba(200,210,255,0.6)"}">{p}</span>'
        for p in pages
    )
    st.markdown(f"""
    <div class="aura-navbar">
        <div>
            <div class="aura-logo">🌍 AURA</div>
            <div class="aura-tagline">Autonomous Urban Risk Agent</div>
        </div>
        <div style="display:flex;gap:6px;">{links}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:          st.session_state.page = "Dashboard"
if "city"  not in st.session_state:         st.session_state.city = "Delhi"
if "chat_history" not in st.session_state:  st.session_state.chat_history = []
if "user_id" not in st.session_state:       st.session_state.user_id = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌍 AURA Navigation")
    pages = ["Dashboard", "Environment", "My Exposure", "What-If", "Forecast", "Ask AURA", "Agent Activity", "Profile"]
    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p

    st.markdown("---")
    st.session_state.city = st.selectbox("📍 City", SUPPORTED_CITIES, index=SUPPORTED_CITIES.index(st.session_state.city))
    if st.button("🔄 Refresh Data Now", use_container_width=True):
        with st.spinner("Fetching live data..."):
            trigger_now()
        st.success("Data refreshed!")
        st.rerun()

page = st.session_state.page
city = st.session_state.city


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA (loaded once per page render)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_latest(city):
    with get_db() as db:
        row = get_latest_air_quality(db, city)
        if row is None:
            return None
        return {
            "aqi": row.aqi, "pm25": row.pm25, "pm10": row.pm10,
            "no2": row.no2, "temperature": row.temperature,
            "humidity": row.humidity, "wind_speed": row.wind_speed,
            "timestamp": row.timestamp, "source": row.source
        }

@st.cache_data(ttl=300)
def load_history(city, hours=24):
    with get_db() as db:
        rows = get_recent_air_quality(db, city, hours=hours)
        return [{"timestamp": r.timestamp, "aqi": r.aqi, "pm25": r.pm25} for r in rows]

@st.cache_data(ttl=300)
def load_forecasts(city):
    with get_db() as db:
        return forecast_air_quality(city, db)

@st.cache_data(ttl=30)
def load_agent_logs(limit=30):
    with get_db() as db:
        logs = get_recent_agent_logs(db, limit=limit)
        return [{"ts": l.timestamp, "agent": l.agent_name, "action": l.action,
                 "tool": l.tool_name, "result": l.result_summary, "status": l.status}
                for l in logs]

latest    = load_latest(city)
forecasts = load_forecasts(city)
history   = load_history(city)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    navbar("Dashboard")

    if latest is None:
        st.markdown('<div class="alert-banner alert-warning">⚠ No live data yet. Use the sidebar to refresh or add your OpenAQ API key to .env</div>', unsafe_allow_html=True)
        aqi_display = "—"
        risk_result = {"risk_score": None, "risk_level": "UNKNOWN"}
    else:
        risk_result = predict_risk(latest)
        aqi_display = int(latest["aqi"]) if latest["aqi"] else "—"

    ts = latest["timestamp"] if latest else None

    # ── Active alerts ──
    with get_db() as db:
        alerts = get_active_alerts(db)

    if alerts:
        for a in alerts[:2]:
            sev = "alert-critical" if a.severity == "CRITICAL" else "alert-warning"
            st.markdown(f'<div class="alert-banner {sev}">🔔 {a.message[:180]}</div>', unsafe_allow_html=True)

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="-webkit-text-fill-color:{aqi_color(latest['aqi'] if latest else None)}">{aqi_display}</div>
            <div class="metric-label">Current AQI</div>
            <div class="metric-sub">{freshness_label(ts)}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        score = risk_result.get("risk_score")
        score_disp = f"{int(score)}" if score is not None else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{score_disp}</div>
            <div class="metric-label">Environmental Risk /100</div>
            <div class="metric-sub"><span class="risk-badge {risk_css(risk_result['risk_level'])}">{risk_result['risk_level']}</span></div>
        </div>""", unsafe_allow_html=True)

    with col3:
        pm25_val = f"{latest['pm25']:.0f}" if (latest and latest.get("pm25")) else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pm25_val}</div>
            <div class="metric-label">PM2.5 µg/m³</div>
            <div class="metric-sub" style="color:#888">WHO limit: 15 µg/m³</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        next_aqi = f"{forecasts[0]['predicted_aqi']:.0f}" if forecasts else "—"
        trend_arrow = ""
        if forecasts and latest and latest["aqi"]:
            diff = forecasts[0]["predicted_aqi"] - latest["aqi"]
            trend_arrow = "↑" if diff > 5 else ("↓" if diff < -5 else "→")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{next_aqi} <span style="font-size:24px">{trend_arrow}</span></div>
            <div class="metric-label">Predicted AQI (+1h)</div>
            <div class="metric-sub" style="color:#888">Model forecast</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])

    # ── AQI trend chart ──
    with col_left:
        st.markdown('<div class="section-title">📈 24h AQI Trend + Forecast</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if history:
            df_hist = pd.DataFrame(history).dropna(subset=["aqi"])
            fig.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["aqi"],
                mode="lines+markers", name="Observed AQI",
                line=dict(color="#8b5cf6", width=2),
                marker=dict(size=5)
            ))
        if forecasts:
            now = datetime.utcnow()
            fx = [now] + [f["predicted_for"] for f in forecasts]
            fy = [latest["aqi"] if latest else 0] + [f["predicted_aqi"] for f in forecasts]
            fig.add_trace(go.Scatter(
                x=fx, y=fy, mode="lines+markers", name="Forecast",
                line=dict(color="#22d3ee", width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond")
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d2ff",
            xaxis=dict(showgrid=False, color="#555"),
            yaxis=dict(showgrid=True, gridcolor="rgba(100,120,255,0.08)", color="#555"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=260
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Pollutants ──
    with col_right:
        st.markdown('<div class="section-title">🧪 Pollutants</div>', unsafe_allow_html=True)
        if latest:
            for label, key, limit in [
                ("PM2.5", "pm25", 250), ("PM10", "pm10", 350),
                ("NO₂", "no2", 200), ("Temperature", "temperature", 50)
            ]:
                val = latest.get(key)
                if val:
                    pct = min(val / limit * 100, 100)
                    color = "#ef4444" if pct > 70 else "#f59e0b" if pct > 40 else "#10b981"
                    st.markdown(f"""
                    <div style="margin-bottom:14px">
                        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:4px">
                            <span>{label}</span><span style="font-weight:600;color:{color}">{val:.1f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px">
                            <div style="width:{pct:.0f}%;background:{color};border-radius:4px;height:6px;transition:width 0.5s ease"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-banner alert-info">No live sensor data yet.</div>', unsafe_allow_html=True)

    # ── Action buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⚡ What-If Simulator", use_container_width=True):
            st.session_state.page = "What-If"; st.rerun()
    with c2:
        if st.button("🤖 Ask AURA", use_container_width=True):
            st.session_state.page = "Ask AURA"; st.rerun()
    with c3:
        if st.button("📊 View Forecast", use_container_width=True):
            st.session_state.page = "Forecast"; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Environment":
    navbar("Environment")
    st.markdown(f'<div class="section-title">🌍 Live Environment — {city}</div>', unsafe_allow_html=True)

    if latest:
        exp = explain_prediction(latest)
        st.markdown('<div class="section-title" style="font-size:15px">🔍 Feature Contribution (Why this risk?)</div>', unsafe_allow_html=True)
        pct = exp.get("pct_contributions", {})
        factors = sorted(pct.items(), key=lambda x: x[1], reverse=True)[:6]
        for feat, val in factors:
            color = "#ef4444" if val > 30 else "#f59e0b" if val > 15 else "#10b981"
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
                    <span style="color:var(--text-secondary)">{feat.upper()}</span>
                    <span style="color:{color};font-weight:700">{val:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:8px">
                    <div style="width:{val:.0f}%;background:{color};border-radius:4px;height:8px"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="alert-banner alert-info">💡 {exp.get("explanation_text", "")}</div>', unsafe_allow_html=True)

        # Anomaly check
        with get_db() as db:
            anom = detect_anomaly(city, latest, db)
        if anom["is_anomaly"]:
            st.markdown(f'<div class="alert-banner alert-critical">🚨 ANOMALY DETECTED — {anom["reason"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-banner alert-success">✅ No anomalies detected. Air quality is within expected range.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-banner alert-warning">⚠ No live data. Add API keys to .env and refresh.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MY EXPOSURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "My Exposure":
    navbar("My Exposure")
    st.markdown('<div class="section-title">👤 My Estimated Personal Exposure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">This is a model-estimated exposure score, not a medical diagnosis.</div>', unsafe_allow_html=True)

    col_f, col_r = st.columns([1, 1])
    with col_f:
        activity = st.selectbox("🏃 Activity", ["Walking", "Running", "Cycling", "Indoor", "Resting", "Other"])
        duration = st.slider("⏱ Duration (minutes)", 10, 180, 60, step=5)
        mask = st.checkbox("😷 Wearing N95 mask")

    env_risk = (latest["aqi"] / 5.0) if (latest and latest.get("aqi")) else 50.0
    env_risk = min(env_risk, 100.0)
    result = calculate_exposure(env_risk, activity, duration, mask)

    with col_r:
        score = result["estimated_exposure"]
        lvl   = result["risk_level"]
        color = {"LOW": "#10b981", "MODERATE": "#f59e0b", "HIGH": "#f97316", "VERY HIGH": "#ef4444"}.get(lvl, "#888")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#555"),
                bar=dict(color=color),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, 25],  color="rgba(16,185,129,0.1)"),
                    dict(range=[25, 50], color="rgba(245,158,11,0.1)"),
                    dict(range=[50, 75], color="rgba(249,115,22,0.1)"),
                    dict(range=[75, 100],color="rgba(239,68,68,0.1)"),
                ],
            ),
            number=dict(font=dict(size=52, color=color), suffix="/100"),
            domain=dict(x=[0,1], y=[0,1])
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d2ff",
            margin=dict(l=20, r=20, t=20, b=20), height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div style="text-align:center"><span class="risk-badge {risk_css(lvl)}">{lvl} EXPOSURE RISK</span></div>', unsafe_allow_html=True)

    # Factor breakdown
    f = result["factors"]
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:28px">{f["base_environmental_risk"]:.0f}</div><div class="metric-label">Env Risk Base</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:28px">{f["activity_multiplier"]}x</div><div class="metric-label">Activity Factor ({activity})</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:28px">{f["duration_multiplier"]}x</div><div class="metric-label">Duration Factor ({duration}m)</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "What-If":
    navbar("What-If")
    st.markdown('<div class="section-title">⚡ What-If Exposure Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Change your plan and see how your estimated exposure risk changes instantly.</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])
    with col_in:
        activity = st.selectbox("Activity", ["Walking", "Running", "Cycling", "Indoor", "Resting"])
        duration = st.slider("Duration (minutes)", 10, 180, 60, step=5)
        mask = st.checkbox("Wearing N95 mask")
        st.markdown("---")
        st.markdown("**If you CANNOT change your time:**")
        constrained = st.checkbox("Show constraint-optimized suggestions")

    env_risk = min((latest["aqi"] / 5.0) if (latest and latest.get("aqi")) else 50.0, 100.0)

    with get_db() as db:
        sim = simulate_alternatives(city, db, activity, duration, mask_usage=mask, current_env_risk=env_risk)

    curr = sim["current_plan"]
    alts = sim["alternatives"]
    best = sim["best_alternative"]

    with col_out:
        # Current plan gauge
        score = curr["estimated_exposure"]
        lvl   = curr["risk_level"]
        color = {"LOW": "#10b981", "MODERATE": "#f59e0b", "HIGH": "#f97316", "VERY HIGH": "#ef4444"}.get(lvl, "#888")
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px">
            <div class="metric-label">CURRENT PLAN (NOW)</div>
            <div class="metric-value" style="-webkit-text-fill-color:{color}">{score:.0f}/100</div>
            <div class="metric-sub"><span class="risk-badge {risk_css(lvl)}">{lvl}</span></div>
        </div>""", unsafe_allow_html=True)

        if best and best["diff_from_current"] < -5:
            saving = abs(best["diff_from_current"])
            st.markdown(f'<div class="alert-banner alert-success">✅ AURA Recommendation: Go at <b>{best["time_label"]}</b> instead — saves <b>{saving:.0f} points</b> of exposure.</div>', unsafe_allow_html=True)

    # Alternative timeline
    st.markdown("### 📊 Exposure Across Time Windows")
    if alts:
        labels = ["NOW"] + [a["time_label"] for a in alts]
        values = [curr["estimated_exposure"]] + [a["estimated_exposure"] for a in alts]
        colors = [aqi_color(v * 5) for v in values]

        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.0f}" for v in values],
            textposition="outside"
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d2ff",
            yaxis=dict(range=[0, 115], showgrid=True, gridcolor="rgba(100,120,255,0.08)"),
            xaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=30, b=0), height=280,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Constraint optimization
    if constrained:
        st.markdown("### 🔧 Constraint Optimizer — Time is Fixed")
        opt = optimize_activity(env_risk, target_max_exposure=50.0, base_activity=activity, base_duration=duration)
        if opt["suggestions"]:
            for s in opt["suggestions"]:
                st.markdown(f'<div class="alert-banner alert-info">💡 {s}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-banner alert-critical">⚠ No single mitigation brings exposure below 50. Consider postponing or going indoors.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    navbar("Forecast")
    st.markdown(f'<div class="section-title">🔮 AQI Forecast — {city}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model-estimated future AQI. Actual values may differ based on real-world conditions.</div>', unsafe_allow_html=True)

    if not forecasts:
        st.markdown('<div class="alert-banner alert-warning">⚠ Not enough historical data to generate a forecast yet.</div>', unsafe_allow_html=True)
    else:
        # Forecast cards
        cols = st.columns(len(forecasts))
        for i, f in enumerate(forecasts):
            lvl = f["risk_level"]
            color = {"LOW": "#10b981", "MODERATE": "#f59e0b", "HIGH": "#f97316", "VERY HIGH": "#ef4444"}.get(lvl, "#888")
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{f['horizon']}</div>
                    <div class="metric-value" style="-webkit-text-fill-color:{color}">{f['predicted_aqi']:.0f}</div>
                    <div class="metric-sub"><span class="risk-badge {risk_css(lvl)}">{lvl}</span></div>
                </div>""", unsafe_allow_html=True)

        # Forecast line chart
        st.markdown("<br>", unsafe_allow_html=True)
        now = datetime.utcnow()
        times  = [now] + [f["predicted_for"] for f in forecasts]
        aqis   = [latest["aqi"] if latest else 0] + [f["predicted_aqi"] for f in forecasts]
        labels = ["NOW"] + [f["horizon"] for f in forecasts]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=aqis, mode="lines+markers+text",
            text=[f"{a:.0f}" for a in aqis], textposition="top center",
            line=dict(color="#22d3ee", width=3),
            marker=dict(size=10, color="#22d3ee"),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.06)"
        ))
        fig.add_hline(y=150, line_dash="dash", line_color="#f97316", annotation_text="UNHEALTHY threshold")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c8d2ff",
            yaxis=dict(showgrid=True, gridcolor="rgba(100,120,255,0.08)"),
            xaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=20, b=0), height=300, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        model_ver = forecasts[0].get("model_version", "unknown")
        st.markdown(f'<div style="font-size:11px;color:var(--text-muted);margin-top:-8px">Model: {model_ver} · Forecast is a prediction, not a guarantee.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASK AURA (Chat)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Ask AURA":
    navbar("Ask AURA")
    st.markdown('<div class="section-title">🤖 Ask AURA</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask about your exposure, plans, or safety — AURA uses live data and your profile, not general knowledge.</div>', unsafe_allow_html=True)

    # Ensure user exists for demo
    with get_db() as db:
        user = db.query(User).filter(User.city == city).first()
        if user:
            st.session_state.user_id = user.id
        else:
            st.session_state.user_id = None
            st.markdown('<div class="alert-banner alert-warning">⚠ No user profile found. Go to the <b>Profile</b> page to set up your profile first.</div>', unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user"><div class="chat-label" style="color:#a78bfa">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-aura"><div class="chat-label" style="color:#22d3ee">🤖 AURA</div>{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("tools"):
                st.markdown(f'<div style="font-size:11px;color:var(--text-muted);margin:-6px 0 8px 0">Tools used: {", ".join(msg["tools"])}</div>', unsafe_allow_html=True)

    # Input
    suggestions = ["Can I go running now?", "I cannot change my time, what should I do?", "When is the best time to go outside?"]
    st.markdown("<br>", unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns(3)
    for i, (col, sugg) in enumerate(zip([col_s1, col_s2, col_s3], suggestions)):
        with col:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state._quick_query = sugg

    query = st.chat_input("Ask AURA anything about your air quality & exposure...")
    if hasattr(st.session_state, "_quick_query"):
        query = st.session_state._quick_query
        del st.session_state._quick_query

    if query and st.session_state.user_id:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("AURA is thinking..."):
            sup = SupervisorAgent()
            result = sup.ask_aura(st.session_state.user_id, query)
        st.session_state.chat_history.append({
            "role": "aura",
            "content": result["response"],
            "tools": result["tools_used"]
        })
        st.rerun()

    elif query and not st.session_state.user_id:
        st.warning("Please create a user profile first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AGENT ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Activity":
    navbar("Agent Activity")
    st.markdown('<div class="section-title">🤖 AURA Agent Activity Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Live view of every action AURA takes autonomously. This demonstrates the agentic loop.</div>', unsafe_allow_html=True)

    auto_refresh = st.toggle("Auto-refresh every 10s", value=False)

    logs = load_agent_logs(limit=40)

    agent_colors = {
        "DATA_AGENT": "#8b5cf6", "DECISION_AGENT": "#22d3ee",
        "SUPERVISOR_AGENT": "#10b981", "FORECAST_AGENT": "#f59e0b"
    }

    if not logs:
        st.markdown('<div class="alert-banner alert-info">No agent activity yet. Use the sidebar refresh button to trigger the agents.</div>', unsafe_allow_html=True)
    else:
        for log in logs:
            dot_class = "dot-success" if log["status"] == "SUCCESS" else ("dot-warn" if log["status"] == "SKIPPED" else "dot-fail")
            color = agent_colors.get(log["agent"], "#888")
            ts_str = log["ts"].strftime("%H:%M:%S") if log["ts"] else "—"
            result_str = (log["result"] or "")[:80]
            st.markdown(f"""
            <div class="agent-log-row">
                <div class="agent-dot {dot_class}"></div>
                <span style="font-size:11px;color:#555;width:65px;flex-shrink:0">{ts_str}</span>
                <span style="font-weight:700;font-size:12px;color:{color};width:150px;flex-shrink:0">{log['agent']}</span>
                <span style="color:var(--text-secondary);font-size:12px;flex:1">{log['action']} · {result_str}</span>
            </div>""", unsafe_allow_html=True)

    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Profile":
    navbar("Profile")
    st.markdown('<div class="section-title">👤 Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your profile lets AURA personalize exposure estimates and proactive alerts.</div>', unsafe_allow_html=True)

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name     = st.text_input("Your Name", value="Demo User")
            p_city   = st.selectbox("City", SUPPORTED_CITIES)
            activity = st.selectbox("Primary Outdoor Activity", ["Walking", "Running", "Cycling", "Other"])
            duration = st.slider("Typical Outdoor Duration (hours)", 0.5, 6.0, 1.0, step=0.5)
        with col2:
            start_time = st.time_input("Usual Outdoor Start Time", value=datetime.strptime("18:00", "%H:%M"))
            end_time   = st.time_input("Usual Outdoor End Time",   value=datetime.strptime("19:00", "%H:%M"))
            travel     = st.selectbox("Travel Mode", ["Walking", "Metro", "Car", "Bike", "Cycling"])
            mask       = st.checkbox("Usually wear a mask outdoors")

        submitted = st.form_submit_button("💾 Save Profile")
        if submitted:
            profile = {
                "name": name, "city": p_city,
                "outdoor_hours": duration,
                "primary_activity": activity,
                "usual_outdoor_start": start_time.strftime("%H:%M"),
                "usual_outdoor_end":   end_time.strftime("%H:%M"),
                "travel_mode": travel,
                "mask_usage": mask
            }
            with get_db() as db:
                user = upsert_user(db, profile)
                st.session_state.user_id = user.id
            st.success(f"✅ Profile saved! AURA will now personalize alerts for {name} in {p_city}.")

    # Show existing profile
    with get_db() as db:
        user = db.query(User).filter(User.city == city).first()
    if user:
        st.markdown(f"""
        <div class="aura-card" style="margin-top:20px">
            <div class="section-title" style="font-size:15px">Current Profile</div>
            <div style="color:var(--text-secondary);font-size:14px;line-height:2">
                <b>Name:</b> {user.name} &nbsp; | &nbsp;
                <b>City:</b> {user.city} &nbsp; | &nbsp;
                <b>Activity:</b> {user.primary_activity}<br>
                <b>Outdoor Time:</b> {user.usual_outdoor_start} – {user.usual_outdoor_end} &nbsp; | &nbsp;
                <b>Duration:</b> {user.outdoor_hours}h &nbsp; | &nbsp;
                <b>Mask:</b> {'Yes' if user.mask_usage else 'No'}
            </div>
        </div>""", unsafe_allow_html=True)
