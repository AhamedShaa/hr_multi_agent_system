"""SQLAlchemy ORM models for the HR automation system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class STMMemory(Base):
    """Short-term memory entry stored as JSON text."""

    __tablename__ = "stm_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LTMMemory(Base):
    """Long-term memory entry stored as JSON text with a significance score."""

    __tablename__ = "ltm_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    significance_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AuditLog(Base):
    """Append-only audit log entry for HR request processing."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    raw_request: Mapped[str] = mapped_column(Text, nullable=False)
    classified_intent: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    agent_routed_to: Mapped[str] = mapped_column(String(255), nullable=False)
    agent_response: Mapped[str] = mapped_column(Text, nullable=False)
    memory_context_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(100), nullable=False)


class Session(Base):
    """User session tracked for request continuity."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_active: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


def prevent_audit_log_update(mapper: object, connection: object, target: AuditLog) -> None:
    """Prevent application-level updates to audit log rows so the table remains append-only."""
    raise ValueError("audit_log is append-only; updates are not allowed.")


def prevent_audit_log_delete(mapper: object, connection: object, target: AuditLog) -> None:
    """Prevent application-level deletes from audit log rows so the table remains append-only."""
    raise ValueError("audit_log is append-only; deletes are not allowed.")


event.listen(AuditLog, "before_update", prevent_audit_log_update)
event.listen(AuditLog, "before_delete", prevent_audit_log_delete)
