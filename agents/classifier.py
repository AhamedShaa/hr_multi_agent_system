"""Intent classification engine for HR agent routing."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from config import settings
from prompts.classifier import (
    INTENT_DESCRIPTIONS,
    RESPONSE_FORMAT_INSTRUCTION,
    SYSTEM_PROMPT,
)
from schemas.models import IntentResult, MemoryContext
from utils.retry import async_retry


class IntentClassifier:
    """Classify HR user messages into supported intent categories."""

    def __init__(self, llm_client: Any) -> None:
        """Initialize the classifier with a LangChain-compatible LLM client."""
        self.llm_client = llm_client

    async def classify(self, message: str, memory_context: MemoryContext) -> IntentResult:
        """Classify a user message using recent memory context and a timeout-bound LLM call."""
        prompt = self._build_prompt(message, memory_context)
        try:
            retried_call = async_retry(
                max_attempts=settings.classifier_max_retries,
                delay=settings.classifier_retry_delay,
            )(self._call_llm_with_timeout)
            raw_response = await retried_call(prompt)
            result = self._parse_response(raw_response)
            if result.confidence < settings.classifier_confidence_threshold:
                return IntentResult(
                    intent="clarification_needed",
                    confidence=result.confidence,
                    entities=result.entities,
                    reasoning=result.reasoning,
                )
            return result
        except Exception:
            return self._fallback_result("Classification failure")

    async def _call_llm_with_timeout(self, prompt: str) -> str:
        """Call the LLM with a bounded timeout."""
        return await asyncio.wait_for(self._call_llm(prompt), timeout=settings.classifier_timeout_seconds)

    def _build_prompt(self, message: str, memory_context: MemoryContext) -> str:
        """Build the prompt used to classify a user message."""
        recent_entries = memory_context.stm_entries[-settings.stm_context_injection_limit :]
        recent_conversation = "\n".join(self._format_memory_entry(entry) for entry in recent_entries)
        if not recent_conversation:
            recent_conversation = "No recent conversation."

        intent_lines = "\n".join(f"- {name}: {desc}" for name, desc in INTENT_DESCRIPTIONS.items())

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Available intents:\n{intent_lines}\n\n"
            f"Recent conversation:\n{recent_conversation}\n\n"
            f"User message:\n{message}\n\n"
            f"{RESPONSE_FORMAT_INSTRUCTION}"
        )

    def _parse_response(self, raw: str) -> IntentResult:
        """Parse an LLM JSON response into an intent result with safe fallback on errors."""
        try:
            cleaned = self._strip_code_fences(raw)
            data = json.loads(cleaned)
            result = IntentResult.model_validate(data)
            return result
        except Exception:
            return self._fallback_result("Parse error")

    async def _call_llm(self, prompt: str) -> str:
        """Call a LangChain-compatible LLM client and normalize the response to text."""
        if hasattr(self.llm_client, "ainvoke"):
            response = await self.llm_client.ainvoke(prompt)
        elif callable(self.llm_client):
            response = self.llm_client(prompt)
            if inspect.isawaitable(response):
                response = await response
        else:
            raise TypeError("LLM client must provide ainvoke or be callable.")

        return self._extract_text(response)

    def _extract_text(self, response: Any) -> str:
        """Extract response text from common LangChain response shapes."""
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return str(response.content)
        if isinstance(response, dict) and "content" in response:
            return str(response["content"])
        return str(response)

    def _strip_code_fences(self, raw: str) -> str:
        """Strip markdown JSON code fences from a raw LLM response if present."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    def _format_memory_entry(self, entry: Any) -> str:
        """Format one memory entry for inclusion in the classifier prompt."""
        if isinstance(entry, dict):
            role = entry.get("role", "unknown")
            intent = entry.get("intent", "unknown")
            message = entry.get("message", "")
            timestamp = entry.get("timestamp", "")
            return f"- {timestamp} | {role} | intent={intent} | message={message}"
        return f"- {entry}"

    def _fallback_result(self, reasoning: str) -> IntentResult:
        """Return the safe fallback classification result."""
        return IntentResult(
            intent="clarification_needed",
            confidence=0.0,
            entities={},
            reasoning=reasoning,
        )
