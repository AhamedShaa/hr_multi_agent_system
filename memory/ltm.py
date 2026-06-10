"""Long-term memory storage and promotion logic for the HR automation system."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from database.models import LTMMemory, utc_now


class LTMManager:
    """Manage long-term memory entries persisted in the ltm_memory table."""

    def __init__(self, db: Session, significance_threshold: float | None = None) -> None:
        """Initialize the long-term memory manager with a database session."""
        self.db = db
        self.significance_threshold = (
            significance_threshold if significance_threshold is not None else settings.ltm_significance_threshold
        )

    def calculate_significance(
        self,
        intent: str,
        had_decision: bool,
        had_entities: bool,
        needed_clarification: bool,
    ) -> float:
        """Calculate significance using the exact configured scoring rubric."""
        score = 0.0
        if intent in ["leave_request", "compliance_query"]:
            score += 0.4
        if had_decision is True:
            score += 0.3
        if had_entities is True:
            score += 0.2
        if needed_clarification is True:
            score += 0.1
        return min(score, 1.0)

    def promote_from_stm(
        self,
        user_id: str,
        stm_entry: dict[str, Any],
        significance_score: float,
    ) -> LTMMemory | None:
        """Promote a short-term memory entry to long-term memory when it meets the threshold."""
        if significance_score < self.significance_threshold:
            return None

        now = utc_now()
        entry = LTMMemory(
            user_id=user_id,
            content=json.dumps(stm_entry),
            significance_score=significance_score,
            created_at=now,
            last_accessed=now,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_relevant(
        self, user_id: str, current_intent: str, limit: int = settings.ltm_retrieval_limit
    ) -> list[dict[str, Any]]:
        """Return relevant long-term memory entries ordered by significance score descending."""
        query = (
            select(LTMMemory)
            .where(LTMMemory.user_id == user_id)
            .order_by(LTMMemory.significance_score.desc())
        )
        entries = self.db.scalars(query).all()
        relevant_entries: list[dict[str, Any]] = []

        for entry in entries:
            data = self._deserialize_entry(entry)
            if data.get("intent") == current_intent:
                relevant_entries.append(data)
                if len(relevant_entries) >= limit:
                    break

        return relevant_entries

    def update_last_accessed(self, ltm_entry_id: int) -> LTMMemory | None:
        """Update the last accessed timestamp for a long-term memory entry."""
        entry = self.db.get(LTMMemory, ltm_entry_id)
        if entry is None:
            return None

        entry.last_accessed = utc_now()
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def _deserialize_entry(self, entry: LTMMemory) -> dict[str, Any]:
        """Deserialize an ORM long-term memory entry into a dictionary."""
        data = json.loads(entry.content)
        data["id"] = entry.id
        data["user_id"] = entry.user_id
        data["significance_score"] = entry.significance_score
        data["created_at"] = entry.created_at.isoformat()
        data["last_accessed"] = entry.last_accessed.isoformat()
        return data
