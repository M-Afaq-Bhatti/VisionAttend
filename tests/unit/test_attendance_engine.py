import pytest
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Employee
from src.attendance.engine import AttendanceEngine

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()

@pytest.fixture
def employee(db_session):
    emp = Employee(
        employee_code="E001",
        full_name="John Doe",
        department="Engineering",
        position="Developer"
    )
    db_session.add(emp)
    db_session.commit()
    db_session.refresh(emp)
    return emp

def test_process_recognition_success(db_session, employee):
    engine = AttendanceEngine(db_session=db_session, cooldown_minutes=5)
    
    result = engine.process_recognition(
        employee_id=employee.id,
        similarity_score=0.95,
        liveness_score=0.99
    )
    
    assert result["status"] == "recorded"
    assert result["employee_id"] == employee.id
    assert result["event_id"] is not None

def test_process_recognition_cooldown(db_session, employee):
    engine = AttendanceEngine(db_session=db_session, cooldown_minutes=5)
    
    # First check-in
    res1 = engine.process_recognition(
        employee_id=employee.id,
        similarity_score=0.95,
        liveness_score=0.99
    )
    assert res1["status"] == "recorded"
    
    # Second check-in immediately after should trigger cooldown
    res2 = engine.process_recognition(
        employee_id=employee.id,
        similarity_score=0.92,
        liveness_score=0.98
    )
    assert res2["status"] == "cooldown_ignored"

def test_process_unknown(db_session):
    engine = AttendanceEngine(db_session=db_session)
    result = engine.process_unknown(metadata={"reason": "No match found"})
    
    assert result["status"] == "security_alert"
    assert result["event_id"] is not None

def test_process_spoof_attempt(db_session, employee):
    engine = AttendanceEngine(db_session=db_session)
    result = engine.process_spoof_attempt(
        liveness_score=0.1, 
        employee_id=employee.id
    )
    
    assert result["status"] == "security_alert"
    assert result["event_id"] is not None
