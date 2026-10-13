"""
CRUD operations for AURA database models.
All database reads and writes happen through these functions.
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import Optional, List
import json

from src.database.models import (
    AirQuality, User, Prediction,
    ExposurePrediction, Alert, AgentLog
)


# ─── AirQuality ────────────────────────────────────────────────────────────────

def insert_air_quality(db: Session, record: dict) -> AirQuality:
    """Insert a new air quality reading. Returns the inserted row."""
    row = AirQuality(**record)
    db.add(row)
    db.flush()
    return row


def get_latest_air_quality(db: Session, city: str) -> Optional[AirQuality]:
    """Get the most recent reading for a city."""
    return (
        db.query(AirQuality)
        .filter(AirQuality.city.ilike(city.strip()))
        .order_by(desc(AirQuality.timestamp))
        .first()
    )


def get_recent_air_quality(
    db: Session, city: str, hours: int = 24
) -> List[AirQuality]:
    """Get readings for the last N hours for a city (ordered oldest→newest)."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return (
        db.query(AirQuality)
        .filter(AirQuality.city.ilike(city.strip()), AirQuality.timestamp >= cutoff)
        .order_by(AirQuality.timestamp)
        .all()
    )


def is_duplicate(db: Session, city: str, source_timestamp: datetime) -> bool:
    """Check if a reading with the same source_timestamp already exists."""
    return (
        db.query(AirQuality)
        .filter(
            AirQuality.city == city,
            AirQuality.source_timestamp == source_timestamp,
        )
        .first()
        is not None
    )


# ─── Predictions ───────────────────────────────────────────────────────────────

def insert_prediction(db: Session, record: dict) -> Prediction:
    row = Prediction(**record)
    db.add(row)
    db.flush()
    return row


def get_latest_predictions(db: Session, city: str) -> List[Prediction]:
    """Get the most recent predictions for each horizon for a city."""
    cutoff = datetime.utcnow() - timedelta(hours=1)
    return (
        db.query(Prediction)
        .filter(Prediction.city == city, Prediction.created_at >= cutoff)
        .order_by(Prediction.created_at.desc())
        .limit(8)
        .all()
    )


# ─── Users ─────────────────────────────────────────────────────────────────────

def upsert_user(db: Session, profile: dict) -> User:
    """Insert or update a user profile by name + city."""
    existing = (
        db.query(User)
        .filter(User.name == profile["name"], User.city == profile["city"])
        .first()
    )
    if existing:
        for k, v in profile.items():
            setattr(existing, k, v)
        existing.updated_at = datetime.utcnow()
        db.flush()
        return existing
    else:
        row = User(**profile)
        db.add(row)
        db.flush()
        return row


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


# ─── Exposure Predictions ──────────────────────────────────────────────────────

def insert_exposure(db: Session, record: dict) -> ExposurePrediction:
    row = ExposurePrediction(**record)
    db.add(row)
    db.flush()
    return row


# ─── Alerts ────────────────────────────────────────────────────────────────────

def insert_alert(db: Session, record: dict) -> Alert:
    row = Alert(**record)
    db.add(row)
    db.flush()
    return row


def get_active_alerts(db: Session, user_id: Optional[int] = None) -> List[Alert]:
    q = db.query(Alert).filter(Alert.status == "ACTIVE")
    if user_id:
        q = q.filter(Alert.user_id == user_id)
    return q.order_by(desc(Alert.timestamp)).limit(20).all()


# ─── Agent Logs ────────────────────────────────────────────────────────────────

def log_agent_action(
    db: Session,
    agent_name: str,
    action: str,
    tool_name: str = "",
    reason: str = "",
    result_summary: str = "",
    status: str = "SUCCESS",
    execution_time: float = 0.0,
) -> AgentLog:
    row = AgentLog(
        agent_name=agent_name,
        action=action,
        tool_name=tool_name,
        reason=reason,
        result_summary=result_summary,
        status=status,
        execution_time=execution_time,
    )
    db.add(row)
    db.flush()
    return row


def get_recent_agent_logs(db: Session, limit: int = 50) -> List[AgentLog]:
    return (
        db.query(AgentLog)
        .order_by(desc(AgentLog.timestamp))
        .limit(limit)
        .all()
    )
