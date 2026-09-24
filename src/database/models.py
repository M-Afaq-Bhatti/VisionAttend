import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    department = Column(String, nullable=True)
    position = Column(String, nullable=True)
    email = Column(String, nullable=True)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    face_profiles = relationship("FaceProfile", back_populates="employee")
    attendance_events = relationship("AttendanceEvent", back_populates="employee")

class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    embedding_reference = Column(String, nullable=False) # Path to stored embedding (e.g. numpy array)
    model_name = Column(String, nullable=False)
    embedding_dimension = Column(Integer, nullable=False, default=512)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    employee = relationship("Employee", back_populates="face_profiles")

class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    event_type = Column(String, nullable=False, default="check-in")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    similarity_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    source = Column(String, nullable=True)
    status = Column(String, nullable=False) # e.g. "recorded", "duplicate"

    employee = relationship("Employee", back_populates="attendance_events")

class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    event_type = Column(String, nullable=False) # e.g. "unknown_person", "spoof_attempt"
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    confidence = Column(Float, nullable=True)
    source = Column(String, nullable=True)
    metadata_json = Column(String, nullable=True) # JSON string for extra info
