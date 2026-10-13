from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class AirQuality(Base):
    __tablename__ = 'air_quality'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
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
    source = Column(String(50))

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    city = Column(String(100), nullable=False)
    outdoor_hours = Column(Float)
    activity = Column(String(50))
    usual_outdoor_start = Column(String(20)) # e.g., "18:00"
    usual_outdoor_end = Column(String(20))   # e.g., "19:00"
    travel_mode = Column(String(50))
    mask_usage = Column(Boolean, default=False)

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(String(100), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_for = Column(DateTime, index=True)
    predicted_aqi = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String(20)) # e.g., LOW, MODERATE, HIGH
    model_version = Column(String(50))

class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String(50)) # e.g., POLLUTION_SPIKE, EXPOSURE_WARNING
    severity = Column(String(20)) # e.g., INFO, WARNING, CRITICAL
    message = Column(String(500))

class AgentLog(Base):
    __tablename__ = 'agent_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent = Column(String(50)) # e.g., PERCEPTION, REASONING, ACTION
    action = Column(String(255))
    reason = Column(String(500))
    status = Column(String(20)) # e.g., SUCCESS, FAILED

# Setup Database Connection
DATABASE_URL = "sqlite:///aura.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
