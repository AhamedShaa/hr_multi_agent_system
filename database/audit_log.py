"""Append-only audit logging for HR request processing."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from database.models import AuditLog
from schemas.models import AuditEntry

logger = logging.getLogger("hr_agent_system.audit")

_FALLBACK_PATH = Path(__file__).parent.parent / settings.audit_fallback_file


class AuditLogger:
    """Manage audit log inserts and reads.

    This class must never call UPDATE or DELETE on audit_log table. Append-only by design.
    """

    def __init__(self, db: Session) -> None:
        """Initialize the audit logger with an injected database session."""
        self.db = db

    async def log_request(self, entry: AuditEntry) -> AuditEntry:
        """Insert a validated audit log entry, retrying before falling back to a file."""
        return await asyncio.to_thread(self._log_request_sync, entry)

    def _log_request_sync(self, entry: AuditEntry) -> AuditEntry:
        """Synchronously insert a validated audit log entry, retrying before falling back to a file."""
        validated_entry = AuditEntry.model_validate(entry)
        entry_data = validated_entry.model_dump()
        entry_data["id"] = str(uuid.uuid4())

        last_exc: Exception | None = None
        for attempt in range(settings.agent_max_retries):
            try:
                audit_row = AuditLog(**entry_data)
                self.db.add(audit_row)
                self.db.commit()
                self.db.refresh(audit_row)
                return AuditEntry.model_validate(audit_row)
            except Exception as exc:
                self.db.rollback()
                last_exc = exc
                logger.warning(
                    "Audit DB write attempt %d/%d failed: %s", attempt + 1, settings.agent_max_retries, exc
                )
                if attempt < settings.agent_max_retries - 1:
                    time.sleep(settings.agent_retry_delay)

        logger.warning("Audit DB write failed after %d attempts, writing to fallback file", settings.agent_max_retries)
        self._write_fallback(entry_data)
        return validated_entry

    def _write_fallback(self, entry_data: dict) -> None:
        """Append an audit entry as JSON to the fallback file."""
        try:
            serializable = {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in entry_data.items()
            }
            with _FALLBACK_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(serializable) + "\n")
        except Exception as exc:
            logger.error("Audit fallback write also failed: %s", exc)

    def get_logs(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        intent_filter: str | None = None,
    ) -> list[AuditEntry]:
        """Return audit log entries with optional user and intent filters."""
        query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)

        if user_id is not None:
            query = query.where(AuditLog.user_id == user_id)
        if intent_filter is not None:
            query = query.where(AuditLog.classified_intent == intent_filter)

        rows = self.db.scalars(query).all()
        return [AuditEntry.model_validate(row) for row in rows]
