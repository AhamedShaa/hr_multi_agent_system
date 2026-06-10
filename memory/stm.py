"""Short-term memory storage for the HR automation system."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from config import settings
from database.models import STMMemory, utc_now


class STMManager:
    """Manage short-term memory entries persisted in the stm_memory table."""

    def __init__(self, db: Session) -> None:
        """Initialize the short-term memory manager with a database session."""
        self.db = db

    def add_entry(self, user_id: str, session_id: str, content: dict[str, Any]) -> STMMemory:
        """Store a JSON-serialized short-term memory entry for a user session."""
        now = utc_now()
        normalized_content = {
            "role": content.get("role", ""),
            "message": content.get("message", ""),
            "intent": content.get("intent", ""),
            "timestamp": content.get("timestamp", now.isoformat()),
        }
        entry = STMMemory(
            user_id=user_id,
            session_id=session_id,
            content=json.dumps(normalized_content),
            timestamp=now,
            expires_at=now + timedelta(hours=settings.stm_ttl_hours),
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_recent(self, user_id: str, session_id: str, limit: int = settings.stm_max_entries) -> list[dict[str, Any]]:
        """Return the most recent short-term memory entries for a user session."""
        query = (
            select(STMMemory)
            .where(STMMemory.user_id == user_id, STMMemory.session_id == session_id)
            .order_by(STMMemory.timestamp.desc())
            .limit(limit)
        )
        entries = self.db.scalars(query).all()
        return [self._deserialize_entry(entry) for entry in entries]

    def expire_old_entries(self, hours: int = settings.stm_ttl_hours) -> int:
        """Delete short-term memory entries older than the provided number of hours."""
        cutoff = utc_now() - timedelta(hours=hours)
        statement = delete(STMMemory).where(STMMemory.timestamp < cutoff)
        result = self.db.execute(statement)
        self.db.commit()
        return int(result.rowcount or 0)

    def _deserialize_entry(self, entry: STMMemory) -> dict[str, Any]:
        """Deserialize an ORM short-term memory entry into a dictionary."""
        data = json.loads(entry.content)
        data["id"] = entry.id
        data["user_id"] = entry.user_id
        data["session_id"] = entry.session_id
        return data
