"""Compliance sub-agent for HR policy and procedure questions."""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any

from config import settings
from mock_data.employees import get_policies
from prompts.compliance import RESPONSE_TEMPLATE
from schemas.models import AgentResponse, IntentResult, MemoryContext
from utils.retry import async_retry


class ComplianceAgent:
    """Handle mock HR compliance and policy information requests."""

    def __init__(self, llm_client: Any | None = None, timeout_seconds: int = settings.agent_timeout_seconds) -> None:
        """Initialize the compliance agent with optional LLM client and timeout."""
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        message: str,
        intent_result: IntentResult,
        memory_context: MemoryContext,
        employee_data: dict[str, Any],
    ) -> AgentResponse:
        """Execute a mock compliance lookup and return an agent response."""
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
        """Run the deterministic mock compliance behavior."""
        policy = self._find_policy(message)
        response = RESPONSE_TEMPLATE.format(policy_name=policy["name"], policy_excerpt=policy["content"])
        return AgentResponse(
            agent_name="ComplianceAgent",
            response=response,
            status="success",
            data={"policy_referenced": policy["name"], "source": "mock_policy_db"},
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

    def _find_policy(self, message: str) -> dict[str, str]:
        """Find the best matching mock policy for keywords in the message."""
        policies = get_policies()
        keywords = self._keywords(message)
        best_policy = policies[0]
        best_score = -1

        for policy in policies:
            haystack = f"{policy['name']} {policy['content']}".lower()
            score = sum(1 for keyword in keywords if keyword in haystack)
            if score > best_score:
                best_policy = policy
                best_score = score

        return best_policy

    def _keywords(self, message: str) -> set[str]:
        """Extract simple policy search keywords from a user message."""
        stop_words = {"a", "about", "and", "are", "for", "hr", "i", "is", "of", "the", "to", "what"}
        words = re.findall(r"[a-zA-Z_]+", message.lower())
        return {word for word in words if len(word) > 2 and word not in stop_words}

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
            agent_name="ComplianceAgent",
            response="Request timed out. Please try again.",
            status="failed",
            data={},
        )

    def _exception_response(self) -> AgentResponse:
        """Return a standard exception failure response."""
        return AgentResponse(
            agent_name="ComplianceAgent",
            response="Unable to process request. Please contact HR directly.",
            status="failed",
            data={},
        )
