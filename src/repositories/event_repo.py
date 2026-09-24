"""
Event repository — database CRUD for attendance, recognition, and security events.
"""

import datetime
import json
import logging
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.models import AttendanceEvent, SecurityEvent

logger = logging.getLogger(__name__)


class AttendanceRepository:
    """CRUD operations for AttendanceEvent records."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        employee_id: int,
        event_type: str = "check-in",
        similarity_score: Optional[float] = None,
        liveness_score: Optional[float] = None,
        source: Optional[str] = None,
        status: str = "recorded",
    ) -> AttendanceEvent:
        event = AttendanceEvent(
            employee_id=employee_id,
            event_type=event_type,
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            source=source,
            status=status,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        logger.info(
            "Attendance event created: emp_id=%d type=%s status=%s",
            employee_id, event_type, status,
        )
        return event

    def get_latest_for_employee(
        self, employee_id: int, event_type: str = "check-in"
    ) -> Optional[AttendanceEvent]:
        """Get the most recent attendance event for an employee."""
        return (
            self.db.query(AttendanceEvent)
            .filter(
                AttendanceEvent.employee_id == employee_id,
                AttendanceEvent.event_type == event_type,
            )
            .order_by(AttendanceEvent.timestamp.desc())
            .first()
        )

    def get_today(self) -> list[AttendanceEvent]:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (
            self.db.query(AttendanceEvent)
            .filter(AttendanceEvent.timestamp >= today_start)
            .order_by(AttendanceEvent.timestamp.desc())
            .all()
        )

    def get_by_date_range(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        employee_id: Optional[int] = None,
    ) -> list[AttendanceEvent]:
        query = self.db.query(AttendanceEvent).filter(
            AttendanceEvent.timestamp >= start,
            AttendanceEvent.timestamp <= end,
        )
        if employee_id:
            query = query.filter(AttendanceEvent.employee_id == employee_id)
        return query.order_by(AttendanceEvent.timestamp.desc()).all()

    def count_today(self) -> int:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (
            self.db.query(AttendanceEvent)
            .filter(
                AttendanceEvent.timestamp >= today_start,
                AttendanceEvent.status == "recorded",
            )
            .count()
        )

    def count(self) -> int:
        """Total number of attendance events."""
        return self.db.query(AttendanceEvent).count()

    def get_recent(self, limit: int = 50) -> list[AttendanceEvent]:
        """Get most recent attendance events."""
        return (
            self.db.query(AttendanceEvent)
            .order_by(AttendanceEvent.timestamp.desc())
            .limit(limit)
            .all()
        )


class SecurityEventRepository:
    """CRUD operations for SecurityEvent records (unknown persons, spoof attempts)."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        event_type: str,
        employee_id: Optional[int] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SecurityEvent:
        event = SecurityEvent(
            event_type=event_type,
            employee_id=employee_id,
            confidence=confidence,
            source=source,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        logger.info("Security event created: type=%s", event_type)
        return event

    def get_today(self, event_type: Optional[str] = None) -> list[SecurityEvent]:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = self.db.query(SecurityEvent).filter(
            SecurityEvent.timestamp >= today_start
        )
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)
        return query.order_by(SecurityEvent.timestamp.desc()).all()

    def get_by_date_range(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        event_type: Optional[str] = None,
    ) -> list[SecurityEvent]:
        query = self.db.query(SecurityEvent).filter(
            SecurityEvent.timestamp >= start,
            SecurityEvent.timestamp <= end,
        )
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)
        return query.order_by(SecurityEvent.timestamp.desc()).all()

    def count_today(self, event_type: Optional[str] = None) -> int:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = self.db.query(SecurityEvent).filter(
            SecurityEvent.timestamp >= today_start
        )
        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)
        return query.count()

    def get_recent(self, limit: int = 50) -> list[SecurityEvent]:
        """Get most recent security events."""
        return (
            self.db.query(SecurityEvent)
            .order_by(SecurityEvent.timestamp.desc())
            .limit(limit)
            .all()
        )

