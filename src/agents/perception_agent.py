import time
import logging
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, AirQuality, AgentLog
from src.data_ingestion.openaq_client import OpenAQClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerceptionAgent:
    """
    The Perception Agent (previously Data Agent).
    Responsible for observing the real world, validating data, and storing it.
    """
    def __init__(self):
        self.client = OpenAQClient()
        self.db: Session = SessionLocal()

    def log_action(self, action: str, reason: str, status: str = "SUCCESS"):
        log_entry = AgentLog(
            agent="PERCEPTION",
            action=action,
            reason=reason,
            status=status
        )
        self.db.add(log_entry)
        self.db.commit()

    def observe(self, city="Delhi"):
        """
        Main loop action: Fetch -> Validate -> Store -> Log
        """
        logger.info(f"Perception Agent: Observing environment for {city}...")
        
        # 1. Fetch Data
        results = self.client.fetch_latest_measurements(city=city)
        if not results:
            self.log_action("FETCH_DATA", "No data returned from API", "FAILED")
            return
            
        record = self.client.parse_measurements_to_db_format(results, city=city)
        
        # 2. Validate Data (Simple validation for now)
        if not record or record['pm25'] is None:
            self.log_action("VALIDATE_DATA", "Missing critical PM2.5 reading", "FAILED")
            return
            
        # 3. Store Data
        db_record = AirQuality(
            city=record['city'],
            timestamp=record['timestamp'],
            aqi=record.get('aqi'),
            pm25=record['pm25'],
            pm10=record['pm10'],
            no2=record['no2'],
            so2=record['so2'],
            co=record['co'],
            o3=record['o3'],
            source=record['source']
        )
        
        self.db.add(db_record)
        self.db.commit()
        
        self.log_action(
            "ENVIRONMENT_UPDATE", 
            f"Successfully updated environmental state for {city}. PM2.5: {record['pm25']}"
        )
        
        logger.info(f"Perception Agent: Successfully stored new data for {city}.")
        return db_record

if __name__ == "__main__":
    agent = PerceptionAgent()
    # Trigger a single observation
    agent.observe("Delhi")
