"""
SQLAlchemy database models for AURA.
Defines all tables: users, air_quality, predictions,
exposure_predictions, alerts, agent_logs.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Boolean, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    outdoor_hours = Column(Float, default=1.0)
    primary_activity = Column(String(50), default="Walking")   # Walking/Running/Cycling
    usual_outdoor_start = Column(String(10), default="18:00")  # HH:MM
    usual_outdoor_end = Column(String(10), default="19:00")    # HH:MM
    travel_mode = Column(String(50), default="Metro")
    mask_usage = Column(Boolean, default=False)
    indoor_outdoor_pattern = Column(String(100), default="Mostly Indoor")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AirQuality(Base):
    __tablename__ = "air_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(100), nullable=False, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime, nullable=False, index=True)
    aqi = Column(Float)
    pm25 = Column(Float)
    pm10 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)
    o3 = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    source = Column(String(100), default="OpenAQ")
    source_timestamp = Column(DateTime)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(100), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction_horizon = Column(String(20))   # e.g. "+1h", "+3h", "+6h", "+12h"
    predicted_aqi = Column(Float)
    predicted_pm25 = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String(20))            # LOW / MODERATE / HIGH / VERY HIGH
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class ExposurePrediction(Base):
    __tablename__ = "exposure_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    activity = Column(String(50))
    duration_minutes = Column(Integer)
    planned_time = Column(String(10))          # HH:MM
    estimated_exposure = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String(20))
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50))            # POLLUTION_SPIKE / EXPOSURE_WARNING / FORECAST_HIGH
    severity = Column(String(20))              # INFO / WARNING / CRITICAL
    message = Column(Text)
    trigger_data = Column(Text)                # JSON string of raw trigger values
    status = Column(String(20), default="ACTIVE")   # ACTIVE / DISMISSED / RESOLVED


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent_name = Column(String(50))            # DATA / FORECAST / EXPOSURE / DECISION
    action = Column(String(255))
    tool_name = Column(String(100))
    reason = Column(Text)
    result_summary = Column(Text)
    status = Column(String(20))                # SUCCESS / FAILED / SKIPPED
    execution_time = Column(Float)             # seconds
