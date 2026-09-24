"""
Unit tests for database models and repositories.
"""

import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Employee, FaceProfile, AttendanceEvent, SecurityEvent
from src.repositories.employee_repo import EmployeeRepository, FaceProfileRepository
from src.repositories.event_repo import AttendanceRepository, SecurityEventRepository


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestEmployeeRepository:
    def test_create_employee(self, db_session):
        repo = EmployeeRepository(db_session)
        emp = repo.create(
            employee_code="EMP001",
            full_name="John Doe",
            department="Engineering",
            position="Developer",
            email="john@example.com",
        )
        assert emp.id is not None
        assert emp.employee_code == "EMP001"
        assert emp.full_name == "John Doe"
        assert emp.status == "active"

    def test_get_by_code(self, db_session):
        repo = EmployeeRepository(db_session)
        repo.create(employee_code="EMP002", full_name="Jane Smith")
        found = repo.get_by_code("EMP002")
        assert found is not None
        assert found.full_name == "Jane Smith"

    def test_get_by_code_not_found(self, db_session):
        repo = EmployeeRepository(db_session)
        found = repo.get_by_code("NONEXISTENT")
        assert found is None

    def test_get_all(self, db_session):
        repo = EmployeeRepository(db_session)
        repo.create(employee_code="E1", full_name="A")
        repo.create(employee_code="E2", full_name="B")
        all_emps = repo.get_all()
        assert len(all_emps) == 2

    def test_deactivate(self, db_session):
        repo = EmployeeRepository(db_session)
        emp = repo.create(employee_code="E1", full_name="Test")
        updated = repo.deactivate(emp.id)
        assert updated.status == "inactive"

    def test_count(self, db_session):
        repo = EmployeeRepository(db_session)
        assert repo.count() == 0
        repo.create(employee_code="E1", full_name="A")
        assert repo.count() == 1
        repo.create(employee_code="E2", full_name="B")
        assert repo.count() == 2

    def test_update_employee(self, db_session):
        repo = EmployeeRepository(db_session)
        emp = repo.create(employee_code="E1", full_name="Original")
        updated = repo.update(emp.id, full_name="Updated Name", department="HR")
        assert updated.full_name == "Updated Name"
        assert updated.department == "HR"


class TestFaceProfileRepository:
    def test_create_profile(self, db_session):
        emp_repo = EmployeeRepository(db_session)
        emp = emp_repo.create(employee_code="E1", full_name="Test")

        profile_repo = FaceProfileRepository(db_session)
        profile = profile_repo.create(
            employee_id=emp.id,
            embedding_reference="data/enrollment/E1/embeddings.npz",
            model_name="buffalo_l",
            embedding_dimension=512,
        )
        assert profile.id is not None
        assert profile.employee_id == emp.id

    def test_get_by_employee(self, db_session):
        emp_repo = EmployeeRepository(db_session)
        emp = emp_repo.create(employee_code="E1", full_name="Test")
        profile_repo = FaceProfileRepository(db_session)
        profile_repo.create(emp.id, "path1", "model", 512)
        profile_repo.create(emp.id, "path2", "model", 512)
        profiles = profile_repo.get_by_employee(emp.id)
        assert len(profiles) == 2

    def test_delete_by_employee(self, db_session):
        emp_repo = EmployeeRepository(db_session)
        emp = emp_repo.create(employee_code="E1", full_name="Test")
        profile_repo = FaceProfileRepository(db_session)
        profile_repo.create(emp.id, "path1", "model", 512)
        deleted = profile_repo.delete_by_employee(emp.id)
        assert deleted == 1
        assert profile_repo.count() == 0


class TestAttendanceRepository:
    def test_create_attendance(self, db_session):
        emp_repo = EmployeeRepository(db_session)
        emp = emp_repo.create(employee_code="E1", full_name="Test")

        att_repo = AttendanceRepository(db_session)
        event = att_repo.create(
            employee_id=emp.id,
            similarity_score=0.85,
            liveness_score=0.95,
            source="webcam",
            status="recorded",
        )
        assert event.id is not None
        assert event.event_type == "check-in"
        assert event.status == "recorded"

    def test_get_latest(self, db_session):
        emp_repo = EmployeeRepository(db_session)
        emp = emp_repo.create(employee_code="E1", full_name="Test")
        att_repo = AttendanceRepository(db_session)
        att_repo.create(emp.id, status="recorded")
        latest = att_repo.get_latest_for_employee(emp.id)
        assert latest is not None


class TestSecurityEventRepository:
    def test_create_unknown_event(self, db_session):
        repo = SecurityEventRepository(db_session)
        event = repo.create(
            event_type="unknown_person",
            confidence=0.32,
            source="webcam",
            metadata={"best_match": "EMP001", "similarity": 0.32},
        )
        assert event.id is not None
        assert event.event_type == "unknown_person"

    def test_create_spoof_event(self, db_session):
        repo = SecurityEventRepository(db_session)
        event = repo.create(
            event_type="spoof_attempt",
            employee_id=None,
            confidence=0.12,
            source="webcam",
        )
        assert event.event_type == "spoof_attempt"
