"""End-to-end pipeline tests with mocked LLM responses."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError

from agents.classifier import IntentClassifier
from agents.sub_agents.clarification import ClarificationAgent
from main import app, lifespan
from memory.manager import MemoryManager
from schemas.models import IntentResult


class _FakeLLMResponse:
    """Minimal stand-in for a LangChain LLM response with a `.content` attribute."""

    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
async def client():
    """Start the app lifespan and yield a test HTTP client."""
    async with lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


async def test_valid_leave_request_routes_to_leave_agent(client, monkeypatch):
    """A leave request with sufficient balance should succeed."""

    async def mock_classify(self, message, memory_context):
        return IntentResult(
            intent="leave_request",
            confidence=0.9,
            entities={"days": 5},
            reasoning="test",
        )

    monkeypatch.setattr(IntentClassifier, "classify", mock_classify)

    response = await client.post(
        "/process",
        json={"user_id": "emp_001", "message": "I need 5 days off"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"]["status"] in ("success", "clarification_needed")
    assert data["response"]["agent_name"] == "LeaveAgent"


async def test_low_confidence_intent_routes_to_clarification(client, monkeypatch):
    """A low-confidence classification should route to the clarification agent."""

    async def mock_classify(self, message, memory_context):
        return IntentResult(
            intent="clarification_needed",
            confidence=0.4,
            entities={},
            reasoning="low confidence",
        )

    async def mock_invoke_llm(self, prompt):
        return _FakeLLMResponse("Could you clarify whether this is about scheduling, leave, or policy?")

    monkeypatch.setattr(IntentClassifier, "classify", mock_classify)
    monkeypatch.setattr(ClarificationAgent, "_invoke_llm", mock_invoke_llm)

    response = await client.post(
        "/process",
        json={"user_id": "emp_001", "message": "Help me with something"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"]["status"] == "clarification_needed"


async def test_db_failure_during_memory_retrieval_still_processes(client, monkeypatch):
    """A DB failure in get_context should not fail the request (graceful degradation)."""

    def mock_get_context(self, user_id, session_id, current_intent):
        raise SQLAlchemyError("simulated db down")

    async def mock_classify(self, message, memory_context):
        return IntentResult(
            intent="leave_request",
            confidence=0.9,
            entities={"days": 2},
            reasoning="test",
        )

    monkeypatch.setattr(MemoryManager, "get_context", mock_get_context)
    monkeypatch.setattr(IntentClassifier, "classify", mock_classify)

    response = await client.post(
        "/process",
        json={"user_id": "emp_001", "message": "I need 2 days off"},
    )
    assert response.status_code == 200


async def test_all_five_endpoints_return_correct_status_codes(client, monkeypatch):
    """All 5 endpoints must respond with expected HTTP status codes."""

    async def mock_classify(self, message, memory_context):
        return IntentResult(
            intent="leave_request",
            confidence=0.9,
            entities={"days": 1},
            reasoning="test",
        )

    monkeypatch.setattr(IntentClassifier, "classify", mock_classify)

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    process = await client.post(
        "/process",
        json={"user_id": "emp_001", "message": "I need a day off"},
    )
    assert process.status_code == 200

    audit = await client.get("/audit")
    assert audit.status_code == 200
    assert "entries" in audit.json()

    stm = await client.get("/memory/stm?user_id=nonexistent_user&session_id=test_session_xyz")
    assert stm.status_code in (200, 404)

    ltm = await client.get("/memory/ltm?user_id=nonexistent_user")
    assert ltm.status_code in (200, 404)
