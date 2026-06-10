"""Unified memory manager for short-term and long-term memory operations."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger("hr_agent_system.memory")

from config import settings
from memory.ltm import LTMManager
from memory.stm import STMManager
from schemas.models import MemoryContext

T = TypeVar("T")


class MemoryManager:
    """Compose short-term and long-term memory managers behind one interface."""

    def __init__(self, db: Session) -> None:
        """Initialize the unified memory manager with an injected database session."""
        self.db = db
        self.stm = STMManager(db)
        self.ltm = LTMManager(db)

    async def get_context(self, user_id: str, session_id: str, current_intent: str) -> MemoryContext:
        """Return short-term and long-term memory context for the current interaction."""
        return await asyncio.to_thread(self._get_context_sync, user_id, session_id, current_intent)

    async def store_interaction(self, user_id: str, session_id: str, interaction: dict[str, Any]) -> None:
        """Store an interaction in STM and promote it to LTM when significance meets the threshold."""
        await asyncio.to_thread(self._store_interaction_sync, user_id, session_id, interaction)

    def _get_context_sync(self, user_id: str, session_id: str, current_intent: str) -> MemoryContext:
        """Synchronously return short-term and long-term memory context for the current interaction."""
        try:
            return self._with_retry(
                lambda: MemoryContext(
                    stm_entries=self.stm.get_recent(user_id, session_id),
                    ltm_entries=self.ltm.get_relevant(user_id, current_intent),
                )
            )
        except Exception as exc:
            logger.error("Memory get_context failed for user_id=%s: %s", user_id, exc)
            return MemoryContext(stm_entries=[], ltm_entries=[])

    def _store_interaction_sync(self, user_id: str, session_id: str, interaction: dict[str, Any]) -> None:
        """Synchronously store an interaction in STM and promote it to LTM when significance meets the threshold."""
        try:
            self._with_retry(lambda: self._store_interaction_once(user_id, session_id, interaction))
        except Exception as exc:
            logger.error("Memory store_interaction failed for user_id=%s: %s", user_id, exc)

    def _store_interaction_once(self, user_id: str, session_id: str, interaction: dict[str, Any]) -> None:
        """Store a single interaction without retry wrapping."""
        self.stm.add_entry(user_id, session_id, interaction)
        intent = str(interaction.get("intent", ""))
        entities = interaction.get("entities", {})
        significance_score = self.ltm.calculate_significance(
            intent=intent,
            had_decision=bool(interaction.get("had_decision", False)),
            had_entities=bool(entities),
            needed_clarification=bool(interaction.get("needed_clarification", False)),
        )
        self.ltm.promote_from_stm(user_id, interaction, significance_score)

    def _with_retry(self, operation: Callable[[], T]) -> T:
        """Run a database operation with retry attempts and a delay on DB errors."""
        last_error: SQLAlchemyError | None = None
        for attempt in range(settings.agent_max_retries):
            try:
                return operation()
            except SQLAlchemyError as error:
                self.db.rollback()
                last_error = error
                if attempt < settings.agent_max_retries - 1:
                    time.sleep(settings.agent_retry_delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry operation failed without a database error.")
