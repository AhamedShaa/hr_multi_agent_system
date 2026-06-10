"""Scheduling sub-agent for HR calendar and meeting requests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from config import settings
from prompts.scheduling import NO_DATE_TEMPLATE, SCHEDULED_TEMPLATE, TIMEOUT_RESPONSE
from schemas.models import AgentResponse, IntentResult, MemoryContext
from utils.retry import async_retry


class SchedulingAgent:
    """Handle mock HR scheduling, meeting, and interview booking requests."""

    def __init__(self, llm_client: Any | None = None, timeout_seconds: int = settings.agent_timeout_seconds) -> None:
        """Initialize the scheduling agent with optional LLM client and timeout."""
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        message: str,
        intent_result: IntentResult,
        memory_context: MemoryContext,
        employee_data: dict[str, Any],
    ) -> AgentResponse:
        """Execute a mock scheduling workflow and return an agent response."""
        try:
            return await asyncio.wait_for(
                self._execute(message, intent_result, memory_context, employee_data),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            return self._timeout_response()
        except Exception:
            return self._exception_response()

    async def _execute(
        self,
        message: str,
        intent_result: IntentResult,
        memory_context: MemoryContext,
        employee_data: dict[str, Any],
    ) -> AgentResponse:
        """Run the deterministic mock scheduling behavior."""
        scheduled_time = self._extract_scheduled_time(intent_result.entities)
        if scheduled_time is None:
            return AgentResponse(
                agent_name="SchedulingAgent",
                response=NO_DATE_TEMPLATE,
                status="clarification_needed",
                data={"scheduled_time": None, "calendar_id": "mock-cal-001"},
            )

        return AgentResponse(
            agent_name="SchedulingAgent",
            response=SCHEDULED_TEMPLATE.format(scheduled_time=scheduled_time, calendar_id="mock-cal-001"),
            status="success",
            data={"scheduled_time": scheduled_time, "calendar_id": "mock-cal-001"},
        )

    async def _call_llm(self, prompt: str) -> str:
        """Call an optional LLM client with timeout support and return text."""
        response = await asyncio.wait_for(self._invoke_llm(prompt), timeout=self.timeout_seconds)
        return self._extract_text(response)

    @async_retry(max_attempts=settings.agent_max_retries, delay=settings.agent_retry_delay)
    async def _invoke_llm(self, prompt: str) -> Any:
        """Invoke a LangChain-compatible LLM client."""
        if self.llm_client is None:
            raise TypeError("LLM client is not configured.")
        if hasattr(self.llm_client, "ainvoke"):
            return await self.llm_client.ainvoke(prompt)
        if callable(self.llm_client):
            response = self.llm_client(prompt)
            if inspect.isawaitable(response):
                return await response
            return response
        raise TypeError("LLM client must provide ainvoke or be callable.")

    def _extract_scheduled_time(self, entities: dict[str, Any]) -> str | None:
        """Extract date or time information from classified entities."""
        for key in ("scheduled_time", "datetime", "date_time", "date", "time"):
            value = entities.get(key)
            if value:
                return str(value)
        return None

    def _extract_text(self, response: Any) -> str:
        """Extract text from common LLM response shapes."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return str(response.content)
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        return str(response)

    def _timeout_response(self) -> AgentResponse:
        """Return a standard timeout failure response."""
        return AgentResponse(
            agent_name="SchedulingAgent",
            response=TIMEOUT_RESPONSE,
            status="failed",
            data={},
        )

    def _exception_response(self) -> AgentResponse:
        """Return a standard exception failure response."""
        return AgentResponse(
            agent_name="SchedulingAgent",
            response="Unable to process request. Please contact HR directly.",
            status="failed",
            data={},
        )
