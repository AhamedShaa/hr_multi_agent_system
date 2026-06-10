# HR Multi-Agent System

A FastAPI-based multi-agent automation platform that processes natural-language HR
requests from employees. A [LangGraph](https://github.com/langchain-ai/langgraph)
state machine classifies the intent of each request with an LLM and routes it to a
specialized agent (leave, scheduling, compliance, or clarification), while a two-tier
memory store and an append-only audit log track every interaction.

## Features

- **Single `/process` endpoint** — submit a free-text HR request and get back a
  structured response with the classified intent, routed agent, and result.
- **LLM-based intent classification** with a confidence threshold; low-confidence
  requests are automatically routed to a clarification agent.
- **Four specialized sub-agents**: `LeaveAgent`, `SchedulingAgent`,
  `ComplianceAgent`, `ClarificationAgent`.
- **Two-tier memory**: short-term memory (STM) per session and long-term memory
  (LTM) promoted based on a significance score, both backed by SQLite.
- **Append-only audit log** with automatic retry and a JSONL fallback file if the
  database is unavailable.
- **Resilient by design**: every LLM call is wrapped with a timeout and retry
  decorator; memory/audit failures degrade gracefully instead of failing the request.
- **Centralized configuration** via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
  (`config.py`) and externalized prompt templates (`prompts/`).

## Architecture

```
HTTP Client ──POST /process──► FastAPI ──► Orchestrator (LangGraph)
                                                  │
                          ┌───────────────────────┴────────────────────────┐
                          ▼                                                 │
                 retrieve_memory_node ──► STMManager + LTMManager (SQLite)  │
                          │                                                 │
                          ▼                                                 │
                 classify_intent_node ──► IntentClassifier (LLM)            │
                          │                                                 │
            ┌─────────────┼─────────────────┬──────────────────┐           │
            ▼              ▼                ▼                  ▼           │
      LeaveAgent   SchedulingAgent   ComplianceAgent   ClarificationAgent   │
            └─────────────┴────────────────┴──────────────────┘            │
                          │                                                 │
                          ▼                                                 │
                   finalize_node ──► STM store / LTM promote + AuditLogger ─┘
                          │
                  ProcessResponse
```

See [REPORT.md](REPORT.md) for the full design rationale and trade-offs.

## Project Structure

```
hr_agent_system/
├── main.py                  # FastAPI app, routes, middleware
├── config.py                 # Centralized settings (pydantic-settings)
├── agents/
│   ├── orchestrator.py       # LangGraph pipeline
│   ├── classifier.py         # LLM intent classifier
│   └── sub_agents/
│       ├── leave.py
│       ├── scheduling.py
│       ├── compliance.py
│       └── clarification.py
├── memory/
│   ├── manager.py            # Unified memory interface (async)
│   ├── stm.py                # Short-term memory (SQLite)
│   └── ltm.py                # Long-term memory (SQLite)
├── database/
│   ├── db.py                 # SQLAlchemy engine/session
│   ├── models.py             # ORM models
│   └── audit_log.py          # Append-only audit logging
├── prompts/                   # Externalized prompt/response templates
├── schemas/models.py          # Pydantic request/response schemas
├── mock_data/employees.py     # Mock employee directory
├── utils/retry.py             # Async retry decorator
└── tests/test_pipeline.py     # End-to-end pipeline tests
```

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (for intent classification and the clarification agent)

### Installation

```bash
git clone https://github.com/AhamedShaa/hr_multi_agent_system.git
cd hr_multi_agent_system
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

At minimum, set `OPENAI_API_KEY`. All other settings have sensible defaults — see
`.env.example` for the full list of configurable values (timeouts, retry counts,
memory limits, CORS origins, etc.). Configuration is loaded once at startup via
`config.py` (`pydantic-settings`); no other module reads environment variables
directly.

## Running the App

```bash
uvicorn main:app --reload
```

The API is available at `http://localhost:8000`, with interactive docs at
`http://localhost:8000/docs`.

## API Endpoints

| Method | Path           | Description                                      |
|--------|----------------|---------------------------------------------------|
| POST   | `/process`     | Submit an HR request for classification & routing |
| GET    | `/audit`       | Paginated audit log entries (filterable)           |
| GET    | `/memory/stm`  | Short-term memory entries for a user/session       |
| GET    | `/memory/ltm`  | Long-term memory entries for a user                |
| GET    | `/health`      | Service and database health check                  |

### Example: `/process`

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"user_id": "emp_001", "message": "I need 2 days off next week"}'
```

```json
{
  "request_id": "...",
  "intent": {"intent": "leave_request", "confidence": 0.92, "entities": {"days": 2}, "reasoning": "..."},
  "response": {"agent_name": "LeaveAgent", "response": "...", "status": "success", "data": {...}},
  "processing_time_ms": 842
}
```

## Testing

```bash
pytest tests/test_pipeline.py -v
```

The test suite mocks LLM calls, so it runs without requiring a live OpenAI quota.

## Tech Stack

- **FastAPI** + **Uvicorn** — HTTP API
- **LangGraph** + **LangChain (OpenAI)** — agent orchestration and LLM calls
- **SQLAlchemy** + **SQLite** — persistence (memory, audit log)
- **Pydantic v2 / pydantic-settings** — schemas and configuration
- **pytest / pytest-asyncio** — testing
