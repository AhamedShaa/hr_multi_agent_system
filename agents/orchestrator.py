"""LangGraph orchestrator for the HR automation multi-agent pipeline."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

logger = logging.getLogger("hr_agent_system.orchestrator")

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents.classifier import IntentClassifier
from agents.sub_agents.clarification import ClarificationAgent
from agents.sub_agents.compliance import ComplianceAgent
from agents.sub_agents.leave import LeaveAgent
from agents.sub_agents.scheduling import SchedulingAgent
from config import settings
from database.audit_log import AuditLogger
from database.db import SessionLocal, create_all_tables
from memory.manager import MemoryManager
from mock_data.employees import get_employee
from schemas.models import AgentResponse, AuditEntry, IntentResult, MemoryContext, ProcessResponse
from utils.guardrails import validate_input


class OrchestrationState(TypedDict):
    """Shared state passed through the LangGraph orchestration pipeline."""

    user_id: str
    session_id: str
    message: str
    memory_context: Optional[MemoryContext]
    intent_result: Optional[IntentResult]
    agent_response: Optional[AgentResponse]
    employee_data: dict[str, Any]
    start_time: float
    error: Optional[str]


class Orchestrator:
    """Coordinate memory, intent classification, routing, agents, and audit logging."""

    def __init__(self) -> None:
        """Initialize the LLM, agents, memory manager, audit logger, and compiled graph."""
        create_all_tables()
        self.db = SessionLocal()
        self.llm = self._build_llm()
        self.memory_manager = MemoryManager(self.db)
        self.audit_logger = AuditLogger(self.db)
        self.intent_classifier = IntentClassifier(self.llm)
        self.scheduling_agent = SchedulingAgent(self.llm)
        self.leave_agent = LeaveAgent(self.llm)
        self.compliance_agent = ComplianceAgent(self.llm)
        self.clarification_agent = ClarificationAgent(self.llm)
        self.graph = self._build_graph()

    async def retrieve_memory_node(self, state: OrchestrationState) -> dict[str, MemoryContext]:
        """Retrieve memory context for the active user session."""
        try:
            memory_context = await self.memory_manager.get_context(
                state["user_id"],
                state["session_id"],
                "unknown",
            )
        except Exception:
            memory_context = MemoryContext(stm_entries=[], ltm_entries=[])
        return {"memory_context": memory_context}

    async def classify_intent_node(self, state: OrchestrationState) -> dict[str, IntentResult]:
        """Classify the user message into one supported HR intent."""
        memory_context = state.get("memory_context") or MemoryContext(stm_entries=[], ltm_entries=[])
        try:
            intent_result = await self.intent_classifier.classify(state["message"], memory_context)
        except Exception:
            intent_result = IntentResult(
                intent="clarification_needed",
                confidence=0.0,
                entities={},
                reasoning="Intent classification failed",
            )
        return {"intent_result": intent_result}

    def route_agent_node(self, state: OrchestrationState) -> str:
        """Route the graph to the sub-agent node matching the classified intent."""
        intent_result = state.get("intent_result")
        intent = intent_result.intent if intent_result is not None else "clarification_needed"

        if intent == "scheduling":
            return "scheduling_node"
        if intent == "leave_request":
            return "leave_node"
        if intent == "compliance_query":
            return "compliance_node"
        return "clarification_node"

    async def scheduling_node(self, state: OrchestrationState) -> dict[str, AgentResponse]:
        """Execute the scheduling sub-agent and store its response in graph state."""
        agent_response = await self._execute_agent(self.scheduling_agent, state)
        return {"agent_response": agent_response}

    async def leave_node(self, state: OrchestrationState) -> dict[str, AgentResponse]:
        """Execute the leave sub-agent and store its response in graph state."""
        agent_response = await self._execute_agent(self.leave_agent, state)
        return {"agent_response": agent_response}

    async def compliance_node(self, state: OrchestrationState) -> dict[str, AgentResponse]:
        """Execute the compliance sub-agent and store its response in graph state."""
        agent_response = await self._execute_agent(self.compliance_agent, state)
        return {"agent_response": agent_response}

    async def clarification_node(self, state: OrchestrationState) -> dict[str, AgentResponse]:
        """Execute the clarification sub-agent and store its response in graph state."""
        agent_response = await self._execute_agent(self.clarification_agent, state)
        return {"agent_response": agent_response}

    async def finalize_node(self, state: OrchestrationState) -> dict[str, Any]:
        """Persist memory and audit data for the completed interaction."""
        processing_time_ms = self._processing_time_ms(state["start_time"])
        intent_result = state.get("intent_result") or self._fallback_intent("Intent missing")
        agent_response = state.get("agent_response") or self._fallback_agent_response()
        memory_context = state.get("memory_context") or MemoryContext(stm_entries=[], ltm_entries=[])

        try:
            await self.memory_manager.store_interaction(
                state["user_id"],
                state["session_id"],
                self._build_interaction(state, intent_result, agent_response),
            )
        except Exception:
            state["error"] = "Memory storage failed"

        try:
            await self.audit_logger.log_request(
                AuditEntry(
                    id="pending",
                    timestamp=datetime.now(timezone.utc),
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    raw_request=state["message"],
                    classified_intent=intent_result.intent,
                    confidence_score=intent_result.confidence,
                    agent_routed_to=agent_response.agent_name,
                    agent_response=agent_response.response,
                    memory_context_used=bool(memory_context.stm_entries or memory_context.ltm_entries),
                    processing_time_ms=processing_time_ms,
                    status=agent_response.status,
                )
            )
        except Exception:
            state["error"] = "Audit logging failed"

        return {
            "intent_result": intent_result,
            "agent_response": agent_response,
            "error": state.get("error"),
        }

    async def run(self, user_id: str, session_id: str, message: str) -> ProcessResponse:
        """Run the HR orchestration graph and return a user-safe process response."""
        start_time = time.time()
        logger.info("Orchestrator.run start request_id=%s user_id=%s", session_id, user_id)

        is_valid, rejection_reason, rejection_message = validate_input(message)
        if not is_valid:
            return ProcessResponse(
                request_id=session_id,
                intent=IntentResult(
                    intent="out_of_scope",
                    confidence=1.0,
                    entities={},
                    reasoning=rejection_reason,
                ),
                response=AgentResponse(
                    agent_name="Guardrail",
                    response=rejection_message,
                    status="clarification_needed",
                    data={"rejection_reason": rejection_reason},
                ),
                processing_time_ms=0,
            )

        initial_state: OrchestrationState = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "memory_context": None,
            "intent_result": None,
            "agent_response": None,
            "employee_data": get_employee(user_id) or {},
            "start_time": start_time,
            "error": None,
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)
            intent_result = final_state.get("intent_result") or self._fallback_intent("Intent missing")
            agent_response = final_state.get("agent_response") or self._fallback_agent_response()
            return ProcessResponse(
                request_id=session_id,
                intent=intent_result,
                response=agent_response,
                processing_time_ms=self._processing_time_ms(start_time),
            )
        except Exception as exc:
            logger.error("Orchestrator.run failed request_id=%s: %s", session_id, exc)
            return ProcessResponse(
                request_id=session_id,
                intent=self._fallback_intent("Unable to classify request"),
                response=self._fallback_agent_response(),
                processing_time_ms=self._processing_time_ms(start_time),
            )

    def close(self) -> None:
        """Close the orchestrator database session."""
        self.db.close()

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph state machine."""
        graph = StateGraph(OrchestrationState)
        graph.add_node("retrieve_memory_node", self.retrieve_memory_node)
        graph.add_node("classify_intent_node", self.classify_intent_node)
        graph.add_node("scheduling_node", self.scheduling_node)
        graph.add_node("leave_node", self.leave_node)
        graph.add_node("compliance_node", self.compliance_node)
        graph.add_node("clarification_node", self.clarification_node)
        graph.add_node("finalize_node", self.finalize_node)

        graph.set_entry_point("retrieve_memory_node")
        graph.add_edge("retrieve_memory_node", "classify_intent_node")
        graph.add_conditional_edges(
            "classify_intent_node",
            self.route_agent_node,
            {
                "scheduling_node": "scheduling_node",
                "leave_node": "leave_node",
                "compliance_node": "compliance_node",
                "clarification_node": "clarification_node",
            },
        )
        graph.add_edge("scheduling_node", "finalize_node")
        graph.add_edge("leave_node", "finalize_node")
        graph.add_edge("compliance_node", "finalize_node")
        graph.add_edge("clarification_node", "finalize_node")
        graph.add_edge("finalize_node", END)
        return graph.compile()

    def _build_llm(self) -> ChatOpenAI:
        """Build the OpenAI chat model from environment settings."""
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.openrouter_api_key or "missing-openrouter-api-key",
            base_url="https://openrouter.ai/api/v1",
        )

    async def _execute_agent(self, agent: Any, state: OrchestrationState) -> AgentResponse:
        """Execute a sub-agent with safe default inputs."""
        intent_result = state.get("intent_result") or self._fallback_intent("Intent missing")
        memory_context = state.get("memory_context") or MemoryContext(stm_entries=[], ltm_entries=[])
        return await agent.execute(
            state["message"],
            intent_result,
            memory_context,
            state["employee_data"],
        )

    def _build_interaction(
        self,
        state: OrchestrationState,
        intent_result: IntentResult,
        agent_response: AgentResponse,
    ) -> dict[str, Any]:
        """Build the interaction payload stored in memory."""
        return {
            "role": "user",
            "message": state["message"],
            "intent": intent_result.intent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entities": intent_result.entities,
            "agent_name": agent_response.agent_name,
            "agent_response": agent_response.response,
            "status": agent_response.status,
            "had_decision": bool(agent_response.data.get("had_decision", False)),
            "needed_clarification": agent_response.status == "clarification_needed",
        }

    def _processing_time_ms(self, start_time: float) -> int:
        """Calculate elapsed processing time in milliseconds."""
        return int((time.time() - start_time) * 1000)

    def _fallback_intent(self, reasoning: str) -> IntentResult:
        """Return a safe fallback intent result."""
        return IntentResult(
            intent="clarification_needed",
            confidence=0.0,
            entities={},
            reasoning=reasoning,
        )

    def _fallback_agent_response(self) -> AgentResponse:
        """Return a safe fallback agent response without exposing internal errors."""
        return AgentResponse(
            agent_name="Orchestrator",
            response="I'm unable to process your request right now. Please try again or contact HR directly.",
            status="failed",
            data={},
        )
