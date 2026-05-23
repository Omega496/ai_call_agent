"""
Mock appointments router.
Connects to SQLite database and handles appointment management.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from pydantic import BaseModel
from core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy database setup
engine = None
SessionLocal = None

class Base(DeclarativeBase):
    pass

class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    appointment_id = Column(String, unique=True, default=lambda: f"APT-{uuid.uuid4().hex[:6].upper()}")
    patient_name = Column(String, nullable=False)
    department = Column(String, nullable=False)
    preferred_date = Column(String, nullable=False)   # stored as YYYY-MM-DD string
    scheduled_date = Column(String, nullable=True)    # confirmed date
    status = Column(String, default="pending")        # pending | confirmed | cancelled
    confirmation_number = Column(String, default=lambda: f"CONF-{uuid.uuid4().hex[:8].upper()}")
    created_at = Column(String, default=lambda: datetime.utcnow().isoformat())

def init_db():
    """Initialize SQLite database engine and create tables."""
    global engine, SessionLocal
    db_path = settings.appointments_db_path
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info(f"SQLite Database initialized and verified at {db_path}")

def get_db():
    """FastAPI dependency for database session access."""
    if SessionLocal is None:
        init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class AppointmentCreate(BaseModel):
    patient_name: str
    department: str
    preferred_date: str
    status: Optional[str] = "pending"

class AppointmentResponse(BaseModel):
    id: int
    appointment_id: str
    patient_name: str
    department: str
    preferred_date: str
    scheduled_date: Optional[str] = None
    status: str
    confirmation_number: str
    created_at: str

    class Config:
        from_attributes = True

class AppointmentErrorResponse(BaseModel):
    detail: str
    appointment_id: Optional[str] = None

router = APIRouter()

@router.post("/", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED)
def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    """
    Create a new appointment.
    Auto-generates appointment_id and confirmation_number.
    Sets scheduled_date = preferred_date (mocking availability) and status = "confirmed".
    """
    logger.debug(f"Creating appointment for patient: {appointment.patient_name}")
    db_appointment = Appointment(
        patient_name=appointment.patient_name,
        department=appointment.department,
        preferred_date=appointment.preferred_date,
        scheduled_date=appointment.preferred_date,  # Mock confirmation date
        status="confirmed"
    )
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    logger.info(f"Appointment booked successfully: {db_appointment.appointment_id}")
    return db_appointment

@router.get("/{appointment_id}", response_model=AppointmentResponse, responses={404: {"model": AppointmentErrorResponse}})
def get_appointment(appointment_id: str, db: Session = Depends(get_db)):
    """
    Retrieve details for a specific appointment by appointment_id.
    """
    app = db.query(Appointment).filter(Appointment.appointment_id == appointment_id).first()
    if not app:
        return JSONResponse(
            status_code=404,
            content={"detail": "Appointment not found", "appointment_id": appointment_id}
        )
    return app

@router.get("/", response_model=List[AppointmentResponse])
def get_appointments(patient_name: Optional[str] = None, db: Session = Depends(get_db)):
    """
    List all appointments, optionally filtered by patient name.
    """
    query = db.query(Appointment)
    if patient_name:
        query = query.filter(Appointment.patient_name.ilike(f"%{patient_name}%"))
    return query.all()

@router.delete("/{appointment_id}", response_model=AppointmentResponse, responses={404: {"model": AppointmentErrorResponse}})
def cancel_appointment(appointment_id: str, db: Session = Depends(get_db)):
    """
    Cancel an appointment (sets status="cancelled").
    """
    app = db.query(Appointment).filter(Appointment.appointment_id == appointment_id).first()
    if not app:
        return JSONResponse(
            status_code=404,
            content={"detail": "Appointment not found", "appointment_id": appointment_id}
        )
    app.status = "cancelled"
    db.commit()
    db.refresh(app)
    logger.info(f"Cancelled appointment: {appointment_id}")
    return app
