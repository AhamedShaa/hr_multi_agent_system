"""Clarification sub-agent for ambiguous HR requests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from config import settings
from prompts.clarification import PROMPT_TEMPLATE, SYSTEM_PROMPT
from schemas.models import AgentResponse, IntentResult, MemoryContext
from utils.retry import async_retry


class ClarificationAgent:
    """Ask clear follow-up questions for vague or ambiguous HR requests."""

    def __init__(self, llm_client: Any | None = None, timeout_seconds: int = settings.agent_timeout_seconds) -> None:
        """Initialize the clarification agent with optional LLM client and timeout."""
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        message: str,
        intent_result: IntentResult,
        memory_context: MemoryContext,
        employee_data: dict[str, Any],
    ) -> AgentResponse:
        """Execute clarification handling and return a clarifying question."""
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
        """Run the clarification behavior using the LLM when configured."""
        prompt = f"{SYSTEM_PROMPT}\n\n{PROMPT_TEMPLATE.format(original_message=message)}"
        if self.llm_client is None:
            clarification = "Could you share whether this is about scheduling, leave, or an HR policy?"
        else:
            clarification = await self._call_llm(prompt)

        return AgentResponse(
            agent_name="ClarificationAgent",
            response=clarification,
            status="clarification_needed",
            data={"original_message": message, "clarification_requested": True},
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
            agent_name="ClarificationAgent",
            response="Request timed out. Please try again.",
            status="failed",
            data={},
        )

    def _exception_response(self) -> AgentResponse:
        """Return a standard exception failure response."""
        return AgentResponse(
            agent_name="ClarificationAgent",
            response="Unable to process request. Please contact HR directly.",
            status="failed",
            data={},
        )
