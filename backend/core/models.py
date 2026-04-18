from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import uuid

from sqlalchemy import String, Float, JSON, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


# Enums

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Pydantic API models
class LogEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # DOOR_FORCED, FIRE_ALARM, ACCESS_DENIED, PANIC_BUTTON, LIFT_ALARM…
    location: str
    device_id: Optional[str] = None
    details: dict = {}


class VisionAssessment(BaseModel):
    threat_detected: bool
    threat_type: Optional[str] = None
    confidence: float
    description: str
    evidence: List[str] = []
    severity: Optional[SeverityLevel] = None
    location: Optional[str] = None


class AudioAssessment(BaseModel):
    threat_detected: bool
    threat_type: Optional[str] = None
    confidence: float
    description: str
    evidence: List[str] = []
    severity: Optional[SeverityLevel] = None
    location: Optional[str] = None
    transcript: Optional[str] = None


class LogAssessment(BaseModel):
    threat_detected: bool
    threat_type: Optional[str] = None
    confidence: float
    description: str
    evidence: List[str] = []
    severity: Optional[SeverityLevel] = None
    location: Optional[str] = None
    triggered_rules: List[str] = []


class Alert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: SeverityLevel
    category: str
    title: str
    description: str
    evidence: List[str]
    recommended_actions: List[str]
    contributing_agents: List[str]
    confidence: float
    location: str
    status: Literal["active", "confirmed", "dismissed"] = "active"
    camera_id: Optional[str] = None
    frame_snapshot: Optional[str] = None  # base64 JPEG


class AlertFeedback(BaseModel):
    alert_id: str
    outcome: Literal["confirmed", "dismissed"]
    officer_note: Optional[str] = None


class AudioInput(BaseModel):
    transcript: str
    source: str = "intercom"
    location: Optional[str] = None


# SQLAlchemy ORM models

class Base(DeclarativeBase):
    pass


class AlertRecord(Base):
    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    severity: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON)
    recommended_actions: Mapped[list] = mapped_column(JSON)
    contributing_agents: Mapped[list] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    location: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="active")
    camera_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    frame_snapshot: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String)
    outcome: Mapped[str] = mapped_column(String)
    officer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# Database lifecycle

_engine = None
_session_factory = None


async def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_session() -> AsyncSession:
    if _session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _session_factory()
