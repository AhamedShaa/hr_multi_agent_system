"""Leave management sub-agent for HR absence requests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from config import settings
from prompts.leave import (
    APPROVAL_TEMPLATE,
    CLARIFICATION_TEMPLATE,
    DENIAL_TEMPLATE,
)
from schemas.models import AgentResponse, IntentResult, MemoryContext
from utils.retry import async_retry


class LeaveAgent:
    """Handle mock leave requests and leave balance checks."""

    def __init__(self, llm_client: Any | None = None, timeout_seconds: int = settings.agent_timeout_seconds) -> None:
        """Initialize the leave agent with optional LLM client and timeout."""
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        message: str,
        intent_result: IntentResult,
        memory_context: MemoryContext,
        employee_data: dict[str, Any],
    ) -> AgentResponse:
        """Execute a mock leave workflow and return an agent response."""
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
        """Run the deterministic mock leave behavior."""
        days_requested = self._extract_days_requested(intent_result.entities)
        if days_requested is None:
            return AgentResponse(
                agent_name="LeaveAgent",
                response=CLARIFICATION_TEMPLATE,
                status="clarification_needed",
                data={"approved": False, "days_requested": 0, "new_balance": int(employee_data.get("leave_balance", 0))},
            )

        leave_balance = int(employee_data.get("leave_balance", 0))
        if days_requested <= leave_balance:
            new_balance = leave_balance - days_requested
            return AgentResponse(
                agent_name="LeaveAgent",
                response=APPROVAL_TEMPLATE.format(days_requested=days_requested, new_balance=new_balance),
                status="success",
                data={
                    "approved": True,
                    "days_requested": days_requested,
                    "new_balance": new_balance,
                    "had_decision": True,
                },
            )

        return AgentResponse(
            agent_name="LeaveAgent",
            response=DENIAL_TEMPLATE.format(reason="Insufficient leave balance", current_balance=leave_balance),
            status="success",
            data={
                "approved": False,
                "days_requested": days_requested,
                "new_balance": leave_balance,
                "had_decision": True,
            },
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

    def _extract_days_requested(self, entities: dict[str, Any]) -> int | None:
        """Extract requested leave days from classified entities."""
        for key in ("days", "days_requested", "leave_days", "duration_days"):
            value = entities.get(key)
            if value is not None:
                return int(value)
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
            agent_name="LeaveAgent",
            response="Request timed out. Please try again.",
            status="failed",
            data={},
        )

    def _exception_response(self) -> AgentResponse:
        """Return a standard exception failure response."""
        return AgentResponse(
            agent_name="LeaveAgent",
            response="Unable to process request. Please contact HR directly.",
            status="failed",
            data={},
        )
