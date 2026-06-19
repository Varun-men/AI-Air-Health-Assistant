import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Air Health Assistant", layout="wide")

# ---------- LOAD MODEL ----------
model = joblib.load("best_aqi_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
# ---------- LOAD DATA ----------
df = pd.read_csv("air_quality.csv")
cities = sorted(df["city"].unique())

st.markdown("""
<style>

/* ===== Animated Purple–Black Background ===== */
.stApp {
    background: linear-gradient(
        -45deg,
        #0b0014,
        #1a0033,
        #2d004d,
        #000000
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

/* Background Animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


/* ===== Title ===== */
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 25px #a855f7;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 22px;
    opacity: 0.85;
}

/* ===== Selectbox Fix (IMPORTANT) ===== */
div[data-baseweb="select"] > div {
    background-color: #12001f !important;
    border-radius: 12px;
    border: 1px solid #6d28d9;
}

div[data-baseweb="select"] span {
    color: white !important;
    font-size: 16px;
}

/* Dropdown options */
div[data-baseweb="menu"] {
    background-color: #12001f !important;
}

div[data-baseweb="menu"] span {
    color: white !important;
}

/* ===== Neon Button ===== */
.stButton>button {
    width: 100%;
    height: 65px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    background: linear-gradient(90deg,#9333ea,#22d3ee);
    color: black;
    box-shadow: 0 0 30px #9333ea;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 55px #9333ea;
}

/* Reduce top padding */
.block-container {
    padding-top: 2rem;
}
            
div.element-container,
div.stMarkdown,
div.stText,
div.stSelectbox,
div.stButton {
    background: transparent !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="main-title">🌍 AI Air Health Assistant</div>
<div class="subtitle">Your Personal AI Doctor for the Air You Breathe</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- CITY SELECTION PANEL ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns([3,2], gap="small")

with col1:
    city = st.selectbox("Select Your City", cities)

with col2:
    st.markdown(
        "<div style='margin-top:40px; font-size:16px; opacity:0.9;'>"
        "AI analyzes pollution, weather & historical trends to predict tomorrow's AQI."
        "</div>",
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREPARE DATA ----------
city_df = df[df["city"] == city].sort_values("date")

if len(city_df) < 4:
    st.error("Not enough historical data.")
    st.stop()

last3 = city_df["aqi"].tail(3).values
latest = city_df.iloc[-1]
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
    "city_Jaipur": 1 if city=="Jaipur" else 0,
    "city_Lucknow": 1 if city=="Lucknow" else 0
}

features = pd.DataFrame([input_data])
features = features[feature_columns]

features = np.array([[ 
    latest["pm25"],
    latest["pm10"],
    latest["no2"],
    latest["nh3"],              
    latest["so2"],
    latest["co"],
    latest["o3"],
    latest["temperature"],
    latest["humidity"],
    last3[-1],                  
    last3[-2],                  
    last3[-3],                  
    rolling_mean,               
    latest.get("city_Jaipur",0),
    latest.get("city_Lucknow",0)
]])

st.markdown("<br>", unsafe_allow_html=True)


# ---------- AI PREDICTION BUTTON ----------
prediction = None

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("### 🤖 AI Prediction Engine")

if st.button("🚀 Predict Tomorrow AQI"):
    with st.spinner("AI is analyzing millions of air patterns..."):
        time.sleep(2)
        prediction = int(model.predict(features)[0])

st.markdown('</div>', unsafe_allow_html=True)


# ---------- ONLY SHOW RESULTS AFTER CLICK ----------
if prediction:

    # ---------- AQI CATEGORY ----------
    def aqi_info(aqi):
        if aqi <= 50: return "GOOD 😊", "#00e676"
        elif aqi <= 100: return "MODERATE 🙂", "#ffd54f"
        elif aqi <= 200: return "UNHEALTHY ⚠️", "#ff9800"
        else: return "VERY DANGEROUS 🚨", "#ff4b4b"

    status, color = aqi_info(prediction)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- HERO ----------
    col1, col2 = st.columns([2,1])

# LEFT PANEL
    with col1:
            st.markdown(f"""
            <div style="
            padding:30px;
            border-radius:18px;
            background:linear-gradient(145deg,#161b22,#0e1117);
            box-shadow:0 0 25px rgba(0,0,0,0.6);
            border-left:8px solid {color};
            ">
            
            <h1 style="
            font-size:44px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:6px;
            background:linear-gradient(90deg,#ffffff,#8be9fd);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            ">
            {city}
            </h1>

            <p style="
            font-size:14px;
            opacity:0.65;
            letter-spacing:0.5px;
            margin-bottom:10px;
            ">
            {datetime.now().strftime("%d %B %Y")}
            </p>
            
            <h1 style="
            color:{color};
            margin-top:10px;
            font-size:42px;
            font-weight:800;
            letter-spacing:1px;
            ">
            {status}
            </h1>
            
            <p style="
            opacity:0.8;
            font-size:15px;
            ">
            Real-time AI air quality assessment
            </p>

            </div>
            """, unsafe_allow_html=True)


# RIGHT PANEL — SMOOTH COUNT ANIMATION
    with col2:
        placeholder = st.empty()

        for i in range(0, prediction + 1, max(1, prediction//60)):
            placeholder.markdown(f"""
            <div style="
            padding:30px;
            border-radius:20px;
            text-align:center;
            background:linear-gradient(145deg,#0e1117,#161b22);
            box-shadow:0 0 30px {color}40;">
            
            <div style="
                font-size:75px;
                font-weight:bold;
                color:{color};
                text-shadow:0 0 15px {color};">
                {i}
            </div>

            <h4 style="opacity:0.7;">AQI INDEX</h4>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.015)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- AQI BAR ----------
    st.markdown("### 🌡 AQI Severity Meter")

    fig_meter = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        number={'font': {'size': 50}},
        title={'text': "Air Quality Index", 'font': {'size': 22}},
    
        gauge={
            'axis': {'range': [0, 500]},
        
            'bar': {'color': "white"},
        
            'steps': [
            {'range': [0, 50], 'color': "#00E400"},
            {'range': [50, 100], 'color': "#FFFF00"},
            {'range': [100, 200], 'color': "#FF7E00"},
            {'range': [200, 300], 'color': "#FF0000"},
            {'range': [300, 500], 'color': "#8F3F97"}
        ],
        
            'threshold': {
            'line': {'color': "cyan", 'width': 4},
            'thickness': 0.75,
            'value': prediction
        }
        }
    ))

    fig_meter.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=30, r=30, t=60, b=20)
    )

    st.plotly_chart(fig_meter, use_container_width=True)

    # ---------- BEAUTIFUL HEALTH IMPACT ----------
    st.subheader("👨‍👩‍👧‍👦 Health Risk Analysis")

    def risk_card(title, risk, color):
        st.markdown(f"""
    <div style="
        padding:18px;
        border-radius:15px;
        background:linear-gradient(145deg,#161b22,#0e1117);
        border-left:6px solid {color};
        box-shadow:0 0 15px rgba(0,0,0,0.5);
        text-align:center;">
        <h4>{title}</h4>
        <h2 style="color:{color};">{risk}</h2>
    </div>
    """, unsafe_allow_html=True)


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        risk_card("😷 Asthma Patients",
              "HIGH" if prediction > 100 else "LOW",
              "#ff4b4b" if prediction > 100 else "#00e676")

    with c2:
        risk_card("👶 Children",
              "MODERATE" if prediction > 80 else "LOW",
              "#ff9800" if prediction > 80 else "#00e676")

    with c3:
        risk_card("👴 Elderly",
              "HIGH" if prediction > 120 else "LOW",
              "#ff4b4b" if prediction > 120 else "#00e676")

    with c4:
        risk_card("🏃 Healthy Adults",
              "MODERATE" if prediction > 150 else "LOW",
              "#ff9800" if prediction > 150 else "#00e676")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------AI ADVICE CARD----------
    st.subheader("🤖 AI Health Advice")

    if prediction < 100:
        advice = "🌿 Great day! Enjoy outdoor activities safely."
        bg = "#0f5132"
        border = "#00ff9d"

    elif prediction < 200:
        advice = "😷 Air quality moderate. Limit outdoor exercise & wear a mask."
        bg = "#664d03"
        border = "#ffd43b"

    else:
        advice = "🚨 Dangerous air! Stay indoors & use N95 mask."
        bg = "#58151c"
        border = "#ff4d4d"


    st.markdown(f"""
    <div style="
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, {bg}, #0e1117);
    border-left: 6px solid {border};
    box-shadow: 0 0 30px rgba(0,0,0,0.5);
    font-size: 20px;
    font-weight: 500;
    ">
    {advice}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

   # ---------- BEAUTIFUL TREND GRAPH ----------
    st.subheader("📊 AQI Trend of Last Year 2025")

    past = city_df.tail(30)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=past["date"],
        y=past["aqi"],
        mode="lines+markers",
        line=dict(width=4, color="#00e5ff"),
        marker=dict(size=8, color="#00e5ff"),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.12)",
        hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>"
))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),

    title=dict(
        text="Air Quality Trend",
        font=dict(size=22)
    ),

    xaxis=dict(
        showgrid=False,
        title="Date"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        title="AQI Level"
    ),

    plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL POLLUTANT BREAKDOWN ----------
    st.subheader("🧪 Pollution Breakdown")

    pollutants = {
    "PM2.5": latest["pm25"],
    "PM10": latest["pm10"],
    "NO2": latest["no2"],
    "SO2": latest["so2"],
    "CO": latest["co"],
    "O3": latest["o3"]
}

    colors = ["#00e5ff","#00c853","#ffeb3b","#ff9800","#ff5722","#e91e63"]

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        text=[round(v,1) for v in pollutants.values()],
        textposition="outside",
        marker=dict(
        color=colors,
        line=dict(width=1, color="white")
    ),
    hovertemplate="<b>%{x}</b><br>Level: %{y}<extra></extra>"
))

    fig2.update_layout(
        template="plotly_dark",
        height=420,
        title="Major Pollutant Levels",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_title="Pollutants",
        yaxis_title="Concentration",
        showlegend=False
)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL DOWNLOAD REPORT ----------
    st.subheader("📄 AQI Report")

    st.markdown(f"""
        <div style="
        padding:20px;
        border-radius:15px;
        background:linear-gradient(135deg,#0e1117,#161b22);
        border:1px solid #30363d;">
        <h3 style="color:#00e5ff;">📊 {city} Air Quality Summary</h3>
        <p><b>Date:</b> {datetime.now().strftime("%d %B %Y")}</p>
        <p><b>Predicted AQI:</b> {prediction}</p>
        <p><b>Status:</b> {status}</p>
        <p><b>Advice:</b> {advice}</p>
        </div>
""", unsafe_allow_html=True)

    report = f"""
    City: {city}
    Date: {datetime.now().strftime("%d-%m-%Y")}
    Predicted AQI: {prediction}
    Status: {status}

    Advice:
    {advice}
    """

    st.download_button(
    "⬇️ Download Detailed Report",
    report,
    "AQI_Report.txt",
    use_container_width=True
)

    st.markdown("<br>", unsafe_allow_html=True)
    # ---------- BEAUTIFUL ALERT ----------
    if prediction > 200:
        st.error("🚨 HEALTH ALERT: Air quality is dangerous today!")
    elif prediction > 100:
        st.warning("⚠️ Moderate pollution detected. Take precautions.")
    else:
        st.success("✅ Air quality is safe today!")