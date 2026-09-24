"""
Employee repository — database CRUD operations for employees and face profiles.

Provides a clean data access layer so business logic and UI code
never interact with SQLAlchemy sessions directly.
"""

import datetime
import json
import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from src.database.models import Employee, FaceProfile

logger = logging.getLogger(__name__)


class EmployeeRepository:
    """CRUD operations for Employee records."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        employee_code: str,
        full_name: str,
        department: Optional[str] = None,
        position: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Employee:
        emp = Employee(
            employee_code=employee_code,
            full_name=full_name,
            department=department,
            position=position,
            email=email,
            status="active",
        )
        self.db.add(emp)
        self.db.commit()
        self.db.refresh(emp)
        logger.info("Created employee %s (id=%d)", employee_code, emp.id)
        return emp

    def get_by_id(self, employee_id: int) -> Optional[Employee]:
        return self.db.query(Employee).filter(Employee.id == employee_id).first()

    def get_by_code(self, employee_code: str) -> Optional[Employee]:
        return self.db.query(Employee).filter(
            Employee.employee_code == employee_code
        ).first()

    def get_all(self, status: Optional[str] = None) -> list[Employee]:
        query = self.db.query(Employee)
        if status:
            query = query.filter(Employee.status == status)
        return query.order_by(Employee.created_at.desc()).all()

    def update(self, employee_id: int, **kwargs) -> Optional[Employee]:
        emp = self.get_by_id(employee_id)
        if emp is None:
            return None
        for key, value in kwargs.items():
            if hasattr(emp, key):
                setattr(emp, key, value)
        emp.updated_at = datetime.datetime.utcnow()
        self.db.commit()
        self.db.refresh(emp)
        return emp

    def deactivate(self, employee_id: int) -> Optional[Employee]:
        return self.update(employee_id, status="inactive")

    def count(self, status: Optional[str] = "active") -> int:
        query = self.db.query(Employee)
        if status:
            query = query.filter(Employee.status == status)
        return query.count()


class FaceProfileRepository:
    """CRUD operations for FaceProfile records (embedding references)."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        employee_id: int,
        embedding_reference: str,
        model_name: str,
        embedding_dimension: int = 512,
    ) -> FaceProfile:
        profile = FaceProfile(
            employee_id=employee_id,
            embedding_reference=embedding_reference,
            model_name=model_name,
            embedding_dimension=embedding_dimension,
        )
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def get_by_employee(self, employee_id: int) -> list[FaceProfile]:
        return self.db.query(FaceProfile).filter(
            FaceProfile.employee_id == employee_id
        ).all()

    def delete_by_employee(self, employee_id: int) -> int:
        count = self.db.query(FaceProfile).filter(
            FaceProfile.employee_id == employee_id
        ).delete()
        self.db.commit()
        return count

    def get_all(self) -> list[FaceProfile]:
        return self.db.query(FaceProfile).all()

    def count(self) -> int:
        return self.db.query(FaceProfile).count()
