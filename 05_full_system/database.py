from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

from config import DB_PATH


engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    video_source = Column(String, default="webcam")
    total_persons = Column(Integer, default=0)

    detections = relationship("Detection", back_populates="session", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="session", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    dwell_seconds = Column(Float, default=0.0)
    avg_x = Column(Float, default=0.0)
    avg_y = Column(Float, default=0.0)

    session = relationship("Session", back_populates="detections")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    alert_type = Column(String, nullable=False)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON, nullable=True)

    session = relationship("Session", back_populates="alerts")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
