import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from src.repositories.event_repo import AttendanceRepository, SecurityEventRepository
import logging

logger = logging.getLogger(__name__)

class AttendanceEngine:
    """
    Core business logic for processing attendance and security events based on
    face recognition results. Implements cooldown rules and persistence.
    """
    
    def __init__(self, db_session: Session, cooldown_minutes: int = 5):
        self.db = db_session
        self.attendance_repo = AttendanceRepository(db_session)
        self.security_repo = SecurityEventRepository(db_session)
        self.cooldown_minutes = cooldown_minutes

    def process_recognition(
        self, 
        employee_id: int, 
        similarity_score: float, 
        liveness_score: float, 
        source: str = "camera_1"
    ) -> Dict[str, Any]:
        """
        Processes a recognized face. Applies cooldown rules to avoid spamming check-ins.
        """
        # 1. Check for recent attendance to apply cooldown
        latest_event = self.attendance_repo.get_latest_for_employee(employee_id)
        
        status = "recorded"
        now = datetime.datetime.utcnow()
        
        if latest_event and latest_event.timestamp:
            delta = now - latest_event.timestamp
            if delta.total_seconds() < self.cooldown_minutes * 60:
                status = "cooldown_ignored"
                logger.info(f"Attendance ignored due to cooldown for employee {employee_id}")
        
        # 2. Record the attendance event
        event = self.attendance_repo.create(
            employee_id=employee_id,
            event_type="check-in",
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            source=source,
            status=status
        )
        
        return {
            "status": status,
            "event_id": event.id,
            "employee_id": employee_id,
            "timestamp": event.timestamp,
            "message": "Check-in successful" if status == "recorded" else "Check-in ignored (cooldown)"
        }
        
    def process_unknown(self, source: str = "camera_1", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Logs an event when an unknown person is detected.
        """
        event = self.security_repo.create(
            event_type="unknown_person",
            source=source,
            metadata=metadata
        )
        return {
            "status": "security_alert", 
            "event_id": event.id,
            "message": "Unknown person detected"
        }
        
    def process_spoof_attempt(
        self, 
        liveness_score: float, 
        employee_id: Optional[int] = None, 
        source: str = "camera_1"
    ) -> Dict[str, Any]:
        """
        Logs an event when a spoof attempt (fake face) is detected.
        """
        event = self.security_repo.create(
            event_type="spoof_attempt",
            employee_id=employee_id,
            confidence=1.0 - liveness_score,
            source=source,
            metadata={"liveness_score": liveness_score}
        )
        return {
            "status": "security_alert", 
            "event_id": event.id,
            "message": "Spoof attempt detected"
        }
