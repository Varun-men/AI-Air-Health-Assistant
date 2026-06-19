import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import base64

# Load logo as base64
with open("logo.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Air Health Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# LOAD MODEL & DATA
# ══════════════════════════════════════════════════════════════
model = joblib.load("best_aqi_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("air_quality.csv")
cities = sorted(df["city"].unique())

# ══════════════════════════════════════════════════════════════
# GLOBAL DESIGN SYSTEM (CSS)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">

<style>
/* ═══════ CSS VARIABLES ═══════ */
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
    --gradient-primary: linear-gradient(135deg, #8b5cf6, #22d3ee);
    --gradient-dark: linear-gradient(145deg, #0c1021, #06080f);
    --shadow-glow: 0 0 40px rgba(139, 92, 246, 0.15);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
    --transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══════ GLOBAL RESET ═══════ */
* { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(139, 92, 246, 0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(34, 211, 238, 0.06), transparent);
    color: var(--text-primary);
}

/* Hide Streamlit branding & header */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { 
    background: rgba(6, 8, 15, 0.85) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-card);
}

.block-container {
    padding: 1rem 2rem 4rem 2rem;
    max-width: 1300px;
}

/* ═══════ NAVBAR ═══════ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 30px;
    background: rgba(10, 14, 30, 0.7);
    backdrop-filter: blur(24px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    margin-bottom: 10px;
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Outfit', sans-serif !important;
    font-size: 22px;
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.nav-logo i {
    font-size: 24px;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-links {
    display: flex;
    gap: 8px;
}

.nav-link {
    padding: 8px 18px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    text-decoration: none;
    transition: var(--transition);
    letter-spacing: 0.3px;
    border: 1px solid transparent;
}

.nav-link:hover, .nav-link.active {
    color: var(--text-primary);
    background: rgba(139, 92, 246, 0.12);
    border-color: var(--border-glow);
}

.nav-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: var(--gradient-primary);
    color: white;
    margin-left: 6px;
    letter-spacing: 0.5px;
}

/* ═══════ HERO SECTION ═══════ */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -60%;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
    pointer-events: none;
    animation: heroPulse 6s ease-in-out infinite;
}

@keyframes heroPulse {
    0%, 100% { opacity: 0.5; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.15); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 50px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.25);
    font-size: 15px;
    font-weight: 500;
    color: var(--accent-violet);
    margin-bottom: 24px;
    letter-spacing: 0.5px;
}

.hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 56px;
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px;
}

.hero p {
    font-size: 20px;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0 auto 32px;
    line-height: 1.7;
}

/* ═══════ STATS BAR ═══════ */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 24px 0;
    flex-wrap: wrap;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 36px;
    font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 14px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ═══════ GLASS CARD ═══════ */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 28px;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
}

.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}

/* ═══════ SECTION HEADERS ═══════ */
.section-header {
    margin: 48px 0 24px;
}

.section-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent-violet);
    margin-bottom: 8px;
}

.section-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 36px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.5px;
}

.section-subtitle {
    font-size: 18px;
    color: var(--text-secondary);
    margin-top: 6px;
    line-height: 1.6;
}

/* ═══════ HOW IT WORKS ═══════ */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 28px 0;
}

.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 30px 24px;
    text-align: center;
    transition: var(--transition);
    position: relative;
}

.step-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
}

.step-number {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: var(--gradient-primary);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif !important;
    font-size: 20px;
    font-weight: 800;
    color: white;
    margin-bottom: 16px;
}

.step-card h4 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--text-primary);
}

.step-card p {
    font-size: 15px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ═══════ AQI DISPLAY ═══════ */
.aqi-display {
    text-align: center;
    padding: 40px 20px;
    border-radius: var(--radius-lg);
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    position: relative;
    overflow: hidden;
}

.aqi-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 96px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 8px;
    animation: aqiGlow 2s ease-in-out infinite;
}

@keyframes aqiGlow {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
}

.aqi-label {
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--text-muted);
    margin-bottom: 16px;
}

.aqi-status {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ═══════ AQI SCALE LEGEND ═══════ */
.aqi-scale {
    display: flex;
    border-radius: 12px;
    overflow: hidden;
    margin: 16px 0;
    height: 8px;
    position: relative;
}

.aqi-scale-segment {
    flex: 1;
    position: relative;
    transition: var(--transition);
}

.aqi-scale-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    padding: 0 4px;
}

.aqi-scale-labels span {
    font-size: 13px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

/* ═══════ RISK CARDS ═══════ */
.risk-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 24px;
    text-align: center;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}

.risk-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
}

.risk-icon {
    font-size: 42px;
    margin-bottom: 12px;
}

.risk-title {
    font-size: 16px;
    color: var(--text-secondary);
    margin-bottom: 10px;
    font-weight: 500;
}

.risk-level {
    font-family: 'Outfit', sans-serif !important;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 1px;
}

.risk-bar {
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    margin-top: 14px;
    overflow: hidden;
}

.risk-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease;
}

/* ═══════ ADVICE CARD ═══════ */
.advice-card {
    padding: 28px 32px;
    border-radius: var(--radius-lg);
    border-left: 5px solid;
    display: flex;
    align-items: center;
    gap: 20px;
    transition: var(--transition);
}

.advice-icon {
    font-size: 40px;
    flex-shrink: 0;
}

.advice-text {
    font-size: 19px;
    font-weight: 500;
    line-height: 1.6;
}

.advice-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.6;
    margin-bottom: 6px;
}

/* ═══════ INFO CARD ═══════ */
.info-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 20px;
}

.info-row.three-col {
    grid-template-columns: repeat(3, 1fr);
}

.info-item {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 20px;
    transition: var(--transition);
}

.info-item:hover {
    border-color: var(--border-glow);
}

.info-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.info-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
}

/* ═══════ REPORT CARD ═══════ */
.report-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 32px;
    position: relative;
    overflow: hidden;
}

.report-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
}

.report-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin: 20px 0;
}

.report-item {
    padding: 16px 20px;
    background: rgba(139,92,246,0.05);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(139,92,246,0.1);
}

.report-item-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
}

.report-item-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin-top: 4px;
}

/* ═══════ FOOTER ═══════ */
.footer {
    text-align: center;
    padding: 48px 20px 24px;
    margin-top: 60px;
    border-top: 1px solid var(--border-card);
}

.footer-logo {
    font-family: 'Outfit', sans-serif !important;
    font-size: 24px;
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}

.footer p {
    font-size: 15px;
    color: var(--text-muted);
    margin: 4px 0;
}

.footer-tech {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.tech-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.15);
    letter-spacing: 0.5px;
}

/* ═══════ STREAMLIT OVERRIDES ═══════ */
div[data-baseweb="select"] > div {
    background-color: var(--bg-secondary) !important;
    border-radius: 12px;
    border: 1px solid var(--border-card) !important;
    color: white !important;
}

div[data-baseweb="select"] span { color: white !important; }
div[data-baseweb="menu"] { background-color: var(--bg-secondary) !important; }
div[data-baseweb="menu"] span { color: white !important; }

.stButton > button {
    width: 100%;
    height: 58px;
    font-size: 16px;
    font-weight: 700;
    font-family: 'Inter', sans-serif !important;
    border-radius: 14px;
    border: none;
    background: var(--gradient-primary);
    color: white;
    box-shadow: 0 4px 24px rgba(139,92,246,0.3);
    transition: var(--transition);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 40px rgba(139,92,246,0.45);
}

.stDownloadButton > button {
    width: 100%;
    height: 52px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 14px;
    border: 1px solid var(--border-glow);
    background: rgba(139,92,246,0.1);
    color: var(--accent-violet);
    transition: var(--transition);
}

.stDownloadButton > button:hover {
    background: rgba(139,92,246,0.2);
    transform: translateY(-2px);
}

div.element-container,
div.stMarkdown,
div.stText,
div.stSelectbox,
div.stButton {
    background: transparent !important;
    box-shadow: none !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-card);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid var(--border-card);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-secondary);
    font-weight: 500;
    padding: 10px 24px;
    font-size: 14px;
}

.stTabs [aria-selected="true"] {
    background: rgba(139,92,246,0.15) !important;
    color: var(--text-primary) !important;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ═══════ SECTION FADE-IN ANIMATION ═══════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.6s ease forwards;
}

/* ═══════ DIVIDER ═══════ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-card), transparent);
    margin: 40px 0;
}

/* ═══════ RESPONSIVE ═══════ */
@media (max-width: 768px) {
    .hero h1 { font-size: 36px; }
    .stats-bar { gap: 20px; }
    .steps-grid { grid-template-columns: 1fr; }
    .info-row { grid-template-columns: 1fr; }
    .info-row.three-col { grid-template-columns: 1fr; }
    .report-grid { grid-template-columns: 1fr; }
    .nav-links { display: none; }
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════
navbar_html = (
    '<div class="navbar">'
    '<div class="nav-logo">'
    f'<img src="data:image/png;base64,{logo_b64}" style="height:40px;border-radius:50%;box-shadow:0 0 12px rgba(139,92,246,0.4);" alt="AirHealth AI">'
    ' AirHealth AI</div>'
    '<div class="nav-links">'
    '<span style="font-size:14px;color:rgba(200,210,255,0.5);letter-spacing:0.5px;">Breathe Better. Live Better.</span>'
    '<span class="nav-badge">v2.0</span>'
    '</div>'
    '</div>'
)
st.markdown(navbar_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div style="margin-bottom:20px;">
        <img src="data:image/png;base64,{logo_b64}" style="height:130px;border-radius:50%;box-shadow:0 0 40px rgba(139,92,246,0.35), 0 0 80px rgba(34,211,238,0.15);" alt="AirHealth AI Logo">
    </div>
    <div class="hero-badge">
        <i class="fas fa-sparkles"></i>
        AI Powered Health Prediction Using AQI
    </div>
    <h1>Breathe Better.<br>Live Better.</h1>
    <p>Real-time air quality predictions and personalized health recommendations 
    for Indian cities — powered by advanced machine learning models.</p>
</div>
""", unsafe_allow_html=True)

# Stats bar
total_records = len(df)
model_accuracy = "99.2%"
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">{len(cities)}</div>
        <div class="stat-label">Cities Monitored</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{total_records:,}</div>
        <div class="stat-label">Data Points</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{model_accuracy}</div>
        <div class="stat-label">Model R² Score</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">15</div>
        <div class="stat-label">ML Features</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <div class="section-tag"><i class="fas fa-bolt"></i>&nbsp; How It Works</div>
    <div class="section-title">Three Simple Steps</div>
    <div class="section-subtitle">Our AI pipeline delivers accurate predictions in seconds</div>
</div>

<div class="steps-grid">
    <div class="step-card">
        <div class="step-number">1</div>
        <h4>Select Your City</h4>
        <p>Choose from monitored Indian cities with weekly pollution data spanning over a year.</p>
    </div>
    <div class="step-card">
        <div class="step-number">2</div>
        <h4>AI Analyzes Patterns</h4>
        <p>Our Random Forest model processes 15 features including pollutants, weather, and temporal trends.</p>
    </div>
    <div class="step-card">
        <div class="step-number">3</div>
        <h4>Get Health Insights</h4>
        <p>Receive predicted AQI, risk analysis for vulnerable groups, and personalized health advice.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CITY SELECTION & PREDICTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <div class="section-tag"><i class="fas fa-location-dot"></i>&nbsp; Prediction Engine</div>
    <div class="section-title">Predict Air Quality</div>
    <div class="section-subtitle">Select a city and let our AI forecast the next AQI reading</div>
</div>
""", unsafe_allow_html=True)

col_select, col_info = st.columns([1, 2], gap="large")

with col_select:
    city = st.selectbox("Choose City", cities, label_visibility="visible")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("PREDICT AQI", use_container_width=True)

with col_info:
    city_df = df[df["city"] == city].sort_values("date")
    latest = city_df.iloc[-1]
    current_aqi = int(latest["aqi"])
    avg_aqi = round(city_df["aqi"].mean(), 1)
    min_aqi = int(city_df["aqi"].min())
    max_aqi = int(city_df["aqi"].max())

    st.markdown(f"""
    <div class="glass-card fade-in">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:2px; color:var(--text-muted); margin-bottom:16px;">
            <i class="fas fa-chart-bar"></i>&nbsp; City Overview — {city}
        </div>
        <div class="info-row">
            <div class="info-item">
                <div class="info-label">Latest AQI</div>
                <div class="info-value" style="color: {'#10b981' if current_aqi <= 100 else '#f59e0b' if current_aqi <= 200 else '#ef4444'};">{current_aqi}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Average AQI</div>
                <div class="info-value">{avg_aqi}</div>
            </div>
        </div>
        <div class="info-row">
            <div class="info-item">
                <div class="info-label">Best (Min)</div>
                <div class="info-value" style="color: #10b981;">{min_aqi}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Worst (Max)</div>
                <div class="info-value" style="color: #ef4444;">{max_aqi}</div>
            </div>
        </div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:8px;">
            <i class="fas fa-database"></i>&nbsp; {len(city_df)} records &bull; {city_df['date'].iloc[0]} to {city_df['date'].iloc[-1]}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# AQI SCALE LEGEND (always visible)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin: 24px 0 8px;">
    <div style="font-size:11px; text-transform:uppercase; letter-spacing:2px; color:var(--text-muted); margin-bottom:10px;">
        <i class="fas fa-ruler-horizontal"></i>&nbsp; AQI Scale Reference
    </div>
    <div class="aqi-scale">
        <div class="aqi-scale-segment" style="background:#10b981;"></div>
        <div class="aqi-scale-segment" style="background:#22d3ee;"></div>
        <div class="aqi-scale-segment" style="background:#f59e0b;"></div>
        <div class="aqi-scale-segment" style="background:#f97316;"></div>
        <div class="aqi-scale-segment" style="background:#ef4444;"></div>
        <div class="aqi-scale-segment" style="background:#7c3aed;"></div>
    </div>
    <div class="aqi-scale-labels">
        <span>0 — Good</span>
        <span>50 — Satisfactory</span>
        <span>100 — Moderate</span>
        <span>200 — Unhealthy</span>
        <span>300 — Very Unhealthy</span>
        <span>500 — Hazardous</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════
if len(city_df) < 4:
    st.error("Not enough historical data for this city.")
    st.stop()

last3 = city_df["aqi"].tail(3).values
rolling_mean = city_df["aqi"].tail(7).mean()

input_data = {
    "pm25": latest["pm25"],
    "pm10": latest["pm10"],
    "no2": latest["no2"],
    "nh3": latest["nh3"],
    "so2": latest["so2"],
    "co": latest["co"],
    "o3": latest["o3"],
    "temperature": latest["temperature"],
    "humidity": latest["humidity"],
    "aqi_lag1": last3[-1],
    "aqi_lag2": last3[-2],
    "aqi_lag3": last3[-3],
    "aqi_rolling_mean_7": rolling_mean,
    "city_Jaipur": 1 if city == "Jaipur" else 0,
    "city_Lucknow": 1 if city == "Lucknow" else 0
}

features = np.array([list(input_data.values())])

prediction = None

if predict_clicked:
    with st.spinner("🔍 AI is analyzing air quality patterns..."):
        time.sleep(1.5)
        prediction = int(model.predict(features)[0])

if prediction:

    # AQI Category helper
    def aqi_info(aqi):
        if aqi <= 50:
            return "GOOD", "#10b981", "", "rgba(16,185,129,0.12)"
        elif aqi <= 100:
            return "MODERATE", "#22d3ee", "", "rgba(34,211,238,0.12)"
        elif aqi <= 200:
            return "UNHEALTHY", "#f59e0b", "", "rgba(245,158,11,0.12)"
        elif aqi <= 300:
            return "VERY UNHEALTHY", "#f97316", "", "rgba(249,115,22,0.12)"
        else:
            return "HAZARDOUS", "#ef4444", "", "rgba(239,68,68,0.12)"

    status, color, emoji, bg_tint = aqi_info(prediction)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ──── AQI RESULT DISPLAY ────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag"><i class="fas fa-chart-line"></i>&nbsp; Prediction Result</div>
        <div class="section-title">AQI Forecast</div>
    </div>
    """, unsafe_allow_html=True)

    col_aqi, col_gauge = st.columns([1, 1.2], gap="large")

    with col_aqi:
        # Animated AQI display
        placeholder = st.empty()
        for i in range(0, prediction + 1, max(1, prediction // 40)):
            placeholder.markdown(f"""
            <div class="aqi-display fade-in" style="border-color:{color}22;">
                <div class="aqi-value" style="color:{color}; text-shadow: 0 0 60px {color}40;">{i}</div>
                <div class="aqi-label">Predicted AQI Index</div>
                <div class="aqi-status" style="background:{bg_tint}; color:{color}; border:1px solid {color}30;">
                    {emoji} {status}
                </div>
                <div style="margin-top:20px; font-size:13px; color:var(--text-muted);">
                    {city} &bull; {datetime.now().strftime("%d %B %Y")}
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.012)

        # Final render
        placeholder.markdown(f"""
        <div class="aqi-display fade-in" style="border-color:{color}22;">
            <div class="aqi-value" style="color:{color}; text-shadow: 0 0 60px {color}40;">{prediction}</div>
            <div class="aqi-label">Predicted AQI Index</div>
            <div class="aqi-status" style="background:{bg_tint}; color:{color}; border:1px solid {color}30;">
                {emoji} {status}
            </div>
            <div style="margin-top:20px; font-size:13px; color:var(--text-muted);">
                {city} &bull; {datetime.now().strftime("%d %B %Y")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_meter = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={'font': {'size': 52, 'color': color, 'family': 'Outfit'}},
            title={'text': "Air Quality Index", 'font': {'size': 16, 'color': 'rgba(160,174,221,0.5)'}},
            gauge={
                'axis': {'range': [0, 500], 'tickfont': {'color': 'rgba(160,174,221,0.3)', 'size': 11}},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(16,185,129,0.15)'},
                    {'range': [50, 100], 'color': 'rgba(34,211,238,0.15)'},
                    {'range': [100, 200], 'color': 'rgba(245,158,11,0.15)'},
                    {'range': [200, 300], 'color': 'rgba(249,115,22,0.15)'},
                    {'range': [300, 500], 'color': 'rgba(239,68,68,0.15)'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.8,
                    'value': prediction
                }
            }
        ))

        fig_meter.update_layout(
            height=380,
            margin=dict(l=30, r=30, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f0f2ff'}
        )

        st.plotly_chart(fig_meter, use_container_width=True)


    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ──── HEALTH RISK ANALYSIS ────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag"><i class="fas fa-heart-pulse"></i>&nbsp; Health Impact</div>
        <div class="section-title">Risk Analysis by Group</div>
        <div class="section-subtitle">Estimated health impact for different population groups</div>
    </div>
    """, unsafe_allow_html=True)

    def get_risk(aqi, threshold_low, threshold_high):
        if aqi <= threshold_low:
            return "LOW", "#10b981", 25
        elif aqi <= threshold_high:
            return "MODERATE", "#f59e0b", 55
        else:
            return "HIGH", "#ef4444", 90

    risks = {
        "😷": ("Asthma Patients", *get_risk(prediction, 80, 150)),
        "👶": ("Children", *get_risk(prediction, 100, 200)),
        "👴": ("Elderly", *get_risk(prediction, 90, 170)),
        "🏃": ("Healthy Adults", *get_risk(prediction, 150, 300)),
    }

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, (icon, (title, level, rcolor, bar_pct)) in zip([c1, c2, c3, c4], risks.items()):
        with col:
            st.markdown(f"""
            <div class="risk-card fade-in">
                <div class="risk-icon">{icon}</div>
                <div class="risk-title">{title}</div>
                <div class="risk-level" style="color:{rcolor};">{level}</div>
                <div class="risk-bar">
                    <div class="risk-bar-fill" style="width:{bar_pct}%; background:{rcolor};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ──── AI HEALTH ADVICE ────
    if prediction <= 100:
        advice = "Air quality is good. Enjoy outdoor activities freely — great day for exercise and fresh air!"
        advice_icon = "🌿"
        advice_bg = "rgba(16,185,129,0.08)"
        advice_border = "#10b981"
    elif prediction <= 200:
        advice = "Moderate pollution detected. Sensitive individuals should limit prolonged outdoor exertion. Consider wearing a mask."
        advice_icon = "😷"
        advice_bg = "rgba(245,158,11,0.08)"
        advice_border = "#f59e0b"
    elif prediction <= 300:
        advice = "Unhealthy air quality. Everyone should reduce outdoor activities. Use N95 masks and air purifiers indoors."
        advice_icon = "⚠️"
        advice_bg = "rgba(249,115,22,0.08)"
        advice_border = "#f97316"
    else:
        advice = "DANGEROUS air quality! Stay indoors, seal windows, use N95 masks if going outside. Seek medical attention if you feel unwell."
        advice_icon = "🚨"
        advice_bg = "rgba(239,68,68,0.08)"
        advice_border = "#ef4444"

    st.markdown(f"""
    <div class="advice-card fade-in" style="background:{advice_bg}; border-color:{advice_border};">
        <div class="advice-icon">{advice_icon}</div>
        <div>
            <div class="advice-label">AI Health Recommendation</div>
            <div class="advice-text">{advice}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ──── ANALYSIS TABS ────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag"><i class="fas fa-chart-area"></i>&nbsp; Deep Analysis</div>
        <div class="section-title">Data Exploration</div>
        <div class="section-subtitle">Explore historical trends, pollutant levels, and environmental conditions</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈  AQI Trend", "🧪  Pollutants", "🌡️  Environment"])

    with tab1:
        past = city_df.tail(30)

        # Summary stats
        trend_avg = round(past["aqi"].mean(), 1)
        trend_min = int(past["aqi"].min())
        trend_max = int(past["aqi"].max())
        trend_change = int(past["aqi"].iloc[-1] - past["aqi"].iloc[-6]) if len(past) >= 6 else 0
        change_color = "#10b981" if trend_change <= 0 else "#ef4444"
        change_arrow = "↓" if trend_change <= 0 else "↑"

        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Average</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:var(--text-primary);">{trend_avg}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Min / Max</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;"><span style="color:#10b981;">{trend_min}</span> <span style="color:var(--text-muted);font-size:16px;">—</span> <span style="color:#ef4444;">{trend_max}</span></div>
            </div>
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Weekly Change</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:{change_color};">{change_arrow} {abs(trend_change)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()

        # AQI safety zone bands
        for y0, y1, zc in [(0,50,"rgba(16,185,129,0.07)"),(50,100,"rgba(34,211,238,0.05)"),(100,200,"rgba(245,158,11,0.05)"),(200,500,"rgba(239,68,68,0.04)")]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=zc, line_width=0, layer="below")

        # Main AQI line
        fig.add_trace(go.Scatter(
            x=past["date"], y=past["aqi"],
            mode="lines+markers",
            line=dict(width=3, color="#8b5cf6", shape="spline"),
            marker=dict(size=7, color="#8b5cf6", line=dict(width=2, color="rgba(139,92,246,0.3)")),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
            hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>",
            name="AQI"
        ))

        # 4-week rolling average
        if len(past) >= 4:
            rolling = past["aqi"].rolling(window=4, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=past["date"], y=rolling, mode="lines",
                line=dict(width=2, color="#22d3ee", dash="dot"),
                hovertemplate="<b>4-Week Avg:</b> %{y:.0f}<extra></extra>",
                name="4-Week Average"
            ))

        # Predicted point
        fig.add_trace(go.Scatter(
            x=[past["date"].iloc[-1]], y=[prediction],
            mode="markers+text",
            marker=dict(size=16, color=color, symbol="star", line=dict(width=2, color="white")),
            text=[f"Predicted: {prediction}"], textposition="top center",
            textfont=dict(color=color, size=13, family="Outfit"),
            name="Prediction",
            hovertemplate="<b>Predicted AQI:</b> %{y}<extra></extra>"
        ))

        # Min/Max annotations
        min_idx = past["aqi"].idxmin()
        max_idx = past["aqi"].idxmax()
        fig.add_annotation(x=past.loc[min_idx,"date"], y=past.loc[min_idx,"aqi"],
                           text=f"Min: {int(past.loc[min_idx,'aqi'])}", showarrow=True,
                           arrowhead=2, arrowcolor="#10b981", font=dict(size=11,color="#10b981"), ay=-30)
        fig.add_annotation(x=past.loc[max_idx,"date"], y=past.loc[max_idx,"aqi"],
                           text=f"Max: {int(past.loc[max_idx,'aqi'])}", showarrow=True,
                           arrowhead=2, arrowcolor="#ef4444", font=dict(size=11,color="#ef4444"), ay=-30)

        fig.update_layout(
            height=460, margin=dict(l=0, r=0, t=30, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(160,174,221,0.5)'),
            xaxis=dict(showgrid=False, title="", color="rgba(160,174,221,0.3)",
                       rangeslider=dict(visible=True, thickness=0.05, bgcolor="rgba(139,92,246,0.05)",
                                        bordercolor="rgba(139,92,246,0.15)", borderwidth=1)),
            yaxis=dict(showgrid=True, gridcolor="rgba(139,92,246,0.06)", title="AQI Level", color="rgba(160,174,221,0.3)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=12, color="#a0aedd")),
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pollutants = {
            "PM2.5": latest["pm25"],
            "PM10": latest["pm10"],
            "NO₂": latest["no2"],
            "SO₂": latest["so2"],
            "CO": latest["co"],
            "O₃": latest["o3"]
        }

        who_limits = {"PM2.5": 25, "PM10": 50, "NO₂": 25, "SO₂": 40, "CO": 4, "O₃": 100}
        colors_poll = ["#8b5cf6", "#22d3ee", "#10b981", "#f59e0b", "#f97316", "#ec4899"]

        col_bar, col_radar = st.columns([1, 1], gap="medium")

        with col_bar:
            fig2 = go.Figure()

            # WHO guideline background bars
            fig2.add_trace(go.Bar(
                x=list(who_limits.keys()),
                y=list(who_limits.values()),
                marker=dict(color="rgba(255,255,255,0.06)", line=dict(width=1, color="rgba(255,255,255,0.15)")),
                text=["WHO" for _ in who_limits],
                textposition="inside",
                textfont=dict(color="rgba(255,255,255,0.25)", size=10),
                hovertemplate="<b>%{x} WHO Limit:</b> %{y}<extra></extra>",
                name="WHO Guideline"
            ))

            # Actual pollutant values
            fig2.add_trace(go.Bar(
                x=list(pollutants.keys()),
                y=list(pollutants.values()),
                text=[round(v, 1) for v in pollutants.values()],
                textposition="outside",
                textfont=dict(color="#a0aedd", size=13, family="Outfit"),
                marker=dict(color=colors_poll, line=dict(width=0), cornerradius=6),
                hovertemplate="<b>%{x}</b><br>Level: %{y}<extra></extra>",
                name="Current Level"
            ))

            fig2.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(160,174,221,0.5)'),
                xaxis=dict(showgrid=False, color="rgba(160,174,221,0.3)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(139,92,246,0.06)", title="Concentration", color="rgba(160,174,221,0.3)"),
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                            font=dict(size=12, color="#a0aedd")),
                bargap=0.35
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_radar:
            categories = list(pollutants.keys())
            max_vals = {"PM2.5": 500, "PM10": 500, "NO₂": 200, "SO₂": 100, "CO": 200, "O₃": 200}
            normalized = [min((v / max_vals.get(k, 200)) * 100, 100) for k, v in pollutants.items()]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized + [normalized[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(139,92,246,0.15)',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=6, color="#8b5cf6"),
                name="Pollutant Profile"
            ))

            fig_radar.update_layout(
                height=420,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(160,174,221,0.6)', size=13),
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(139,92,246,0.1)",
                                    linecolor="rgba(139,92,246,0.1)"),
                    angularaxis=dict(gridcolor="rgba(139,92,246,0.1)", linecolor="rgba(139,92,246,0.15)",
                                     tickfont=dict(size=13, color="rgba(200,210,255,0.7)"))
                ),
                showlegend=False,
                title=dict(text="Pollutant Profile", font=dict(size=15, color="rgba(160,174,221,0.5)"), x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        env_data = city_df.tail(30)

        # Summary stat cards
        avg_temp = round(env_data["temperature"].mean(), 1)
        avg_hum = round(env_data["humidity"].mean(), 1)
        temp_corr = round(env_data["temperature"].corr(env_data["aqi"]), 2)
        hum_corr = round(env_data["humidity"].corr(env_data["aqi"]), 2)

        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Avg Temperature</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:#f59e0b;">{avg_temp}°C</div>
            </div>
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Avg Humidity</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:#22d3ee;">{avg_hum}%</div>
            </div>
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Temp ↔ AQI Corr.</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:{'#10b981' if abs(temp_corr) < 0.3 else '#f59e0b' if abs(temp_corr) < 0.6 else '#ef4444'};">{temp_corr}</div>
            </div>
            <div style="flex:1;min-width:120px;padding:14px 18px;border-radius:12px;background:var(--bg-card);border:1px solid var(--border-card);">
                <div style="font-size:13px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Humid ↔ AQI Corr.</div>
                <div style="font-family:'Outfit';font-size:26px;font-weight:700;color:{'#10b981' if abs(hum_corr) < 0.3 else '#f59e0b' if abs(hum_corr) < 0.6 else '#ef4444'};">{hum_corr}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=env_data["date"], y=env_data["temperature"],
            mode="lines+markers",
            name="Temperature (°C)",
            line=dict(width=3, color="#f59e0b", shape="spline"),
            marker=dict(size=6, color="#f59e0b", line=dict(width=1, color="rgba(245,158,11,0.3)")),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.06)",
            hovertemplate="<b>Temp:</b> %{y}°C<extra></extra>"
        ))

        fig3.add_trace(go.Scatter(
            x=env_data["date"], y=env_data["humidity"],
            mode="lines+markers",
            name="Humidity (%)",
            line=dict(width=3, color="#22d3ee", shape="spline"),
            marker=dict(size=6, color="#22d3ee", line=dict(width=1, color="rgba(34,211,238,0.3)")),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.04)",
            yaxis="y2",
            hovertemplate="<b>Humidity:</b> %{y}%<extra></extra>"
        ))

        fig3.update_layout(
            height=460,
            margin=dict(l=0, r=50, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(160,174,221,0.5)'),
            xaxis=dict(showgrid=False, color="rgba(160,174,221,0.3)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(139,92,246,0.06)", title="Temperature (°C)", color="rgba(245,158,11,0.4)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False, color="rgba(34,211,238,0.4)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(size=13, color="#a0aedd")),
            hovermode="x unified"
        )

        st.plotly_chart(fig3, use_container_width=True)


    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ──── REPORT SECTION ────
    st.markdown("""
    <div class="section-header">
        <div class="section-tag"><i class="fas fa-file-lines"></i>&nbsp; Report</div>
        <div class="section-title">AQI Summary Report</div>
        <div class="section-subtitle">Download a detailed report of the current analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col_rep, col_dl = st.columns([2, 1], gap="large")

    with col_rep:
        st.markdown(f"""
        <div class="report-card fade-in">
            <div style="font-family:'Outfit'; font-size:22px; font-weight:700; margin-bottom:4px;">
                {city} — Air Quality Report
            </div>
            <div style="font-size:14px; color:var(--text-muted); margin-bottom:16px;">
                Generated on {datetime.now().strftime("%d %B %Y at %I:%M %p")}
            </div>
            <div class="report-grid">
                <div class="report-item">
                    <div class="report-item-label">City</div>
                    <div class="report-item-value">{city}</div>
                </div>
                <div class="report-item">
                    <div class="report-item-label">Predicted AQI</div>
                    <div class="report-item-value" style="color:{color};">{prediction}</div>
                </div>
                <div class="report-item">
                    <div class="report-item-label">Status</div>
                    <div class="report-item-value">{status}</div>
                </div>
                <div class="report-item">
                    <div class="report-item-label">Date</div>
                    <div class="report-item-value">{datetime.now().strftime("%d %b %Y")}</div>
                </div>
            </div>
            <div style="padding:16px 20px; background:{'rgba(16,185,129,0.06)' if prediction <= 100 else 'rgba(245,158,11,0.06)' if prediction <= 200 else 'rgba(239,68,68,0.06)'}; border-radius:10px; border:1px solid {'rgba(16,185,129,0.15)' if prediction <= 100 else 'rgba(245,158,11,0.15)' if prediction <= 200 else 'rgba(239,68,68,0.15)'}; margin-top:4px;">
                <div style="font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted); margin-bottom:6px;">Recommendation</div>
                <div style="font-size:15px; color:var(--text-primary); line-height:1.6;">{advice}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_dl:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        report_text = f"""
╔══════════════════════════════════════════╗
║     AI Air Health Assistant — Report     ║
╚══════════════════════════════════════════╝

City:           {city}
Date:           {datetime.now().strftime("%d %B %Y")}
Predicted AQI:  {prediction}
Status:         {status}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POLLUTANT LEVELS (Latest Reading)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PM2.5:        {latest['pm25']}
  PM10:         {latest['pm10']}
  NO2:          {latest['no2']}
  SO2:          {latest['so2']}
  CO:           {latest['co']}
  O3:           {latest['o3']}
  Temperature:  {latest['temperature']}°C
  Humidity:     {latest['humidity']}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HEALTH RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {advice}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Powered by AI Air Health Assistant v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        st.download_button(
            "Download Full Report",
            report_text,
            f"AQI_Report_{city}_{datetime.now().strftime('%Y%m%d')}.txt",
            use_container_width=True
        )

        st.markdown(f"""
        <div style="margin-top:16px; padding:20px; border-radius:12px; background:var(--bg-card); border:1px solid var(--border-card);">
            <div style="font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--text-muted); margin-bottom:12px;">Quick Stats</div>
            <div style="font-size:15px; color:var(--text-secondary); line-height:2.2;">
                <i class="fas fa-brain" style="width:20px; color:var(--accent-violet);"></i> Model: Random Forest<br>
                <i class="fas fa-bullseye" style="width:20px; color:var(--accent-cyan);"></i> R² Score: 99.2%<br>
                <i class="fas fa-layer-group" style="width:20px; color:var(--accent-emerald);"></i> Features: 15<br>
                <i class="fas fa-city" style="width:20px; color:var(--accent-pink);"></i> Cities: {len(cities)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ──── ALERT BANNER ────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    if prediction > 300:
        st.error("**CRITICAL ALERT** — Air quality has reached hazardous levels. Stay indoors immediately.")
    elif prediction > 200:
        st.error("**HEALTH ALERT** — Air quality is dangerous. Avoid outdoor activities and wear N95 masks.")
    elif prediction > 100:
        st.warning("**ADVISORY** — Moderate pollution detected. Sensitive groups should take precautions.")
    else:
        st.success("**All Clear** — Air quality is within safe limits. Enjoy your day.")


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-logo"><i class="fas fa-wind"></i> AirHealth AI</div>
    <p>Built by <strong>Varun Kumar</strong> — Machine Learning & Data Science</p>
    <p>Predicting air quality, protecting health.</p>
    <div class="footer-tech">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">Random Forest</span>
    </div>
    <p style="margin-top:16px; font-size:11px;">© 2026 AI Air Health Assistant. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)