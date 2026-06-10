# Technical Report: HR Multi-Agent Automation System

## 1. System Overview

This system is a FastAPI-based multi-agent automation platform that processes natural-language HR requests from employees. When a request arrives at the `/process` endpoint, a LangGraph state machine orchestrates the full pipeline: it retrieves the user's conversational memory context, classifies the intent using an LLM (GPT-4o-mini), routes to one of four specialized agents (leave, scheduling, compliance, or clarification), persists the interaction in a two-tier memory store, and writes an immutable audit record — all before returning a structured `ProcessResponse`. The system is designed for resilience: every LLM call is wrapped in retry logic and a timeout, memory failures degrade gracefully to empty context, and audit records fall back to a local JSONL file if the database is unavailable.

### Architecture Diagram

```
                         ┌────────────────────────────────────────────────┐
                         │              FastAPI Application                │
                         │                                                  │
  HTTP Client ──POST /process──► RequestLoggingMiddleware                  │
                         │              │                                  │
                         │              ▼                                  │
                         │      Input Validation                           │
                         │    (strip, length 1–2000)                       │
                         │              │                                  │
                         │              ▼                                  │
                         │    ┌─────────────────────┐                     │
                         │    │    Orchestrator       │                     │
                         │    │   (LangGraph graph)  │                     │
                         │    └─────────┬───────────┘                     │
                         │              │                                  │
                         └─────────────┼──────────────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────────┐
              │                   LangGraph State Machine               │
              │                                                          │
              │   START                                                  │
              │     │                                                    │
              │     ▼                                                    │
              │  retrieve_memory_node ──► STMManager + LTMManager       │
              │     │                         (SQLite)                  │
              │     ▼                                                    │
              │  classify_intent_node ──► IntentClassifier               │
              │     │                    (GPT-4o-mini, 10s timeout,      │
              │     │                     3x retry, confidence ≥ 0.6)   │
              │     │                                                    │
              │     ▼  conditional routing                               │
              │  ┌──┴──────────────────────────────────┐                │
              │  │                                      │                │
              │  ▼             ▼             ▼          ▼                │
              │ LeaveAgent  SchedulingAgent  ComplianceAgent  ClarificationAgent
              │  (mock)       (mock)          (mock)          (LLM / fallback)
              │  │             │               │               │         │
              │  └──────────────┴───────────────┴───────────────┘        │
              │                         │                                 │
              │                         ▼                                 │
              │                   finalize_node                           │
              │                  /             \                          │
              │           STM store        AuditLogger                   │
              │          + LTM promote      (3x retry + JSONL fallback)  │
              │                         │                                 │
              │                        END                                │
              └──────────────────────────────────────────────────────────┘
                                        │
                               ProcessResponse returned
```

---

## 2. Design Decisions

### LangGraph vs Plain Function Chains

**What:** The agent pipeline is implemented as a compiled LangGraph state machine rather than a plain sequence of function calls.

**Why:** LangGraph provides an explicit, typed state object (`OrchestrationState` TypedDict) that is passed through every node without side effects. Conditional routing is declared as graph edges rather than nested if/else blocks, which makes the branching logic inspectable and testable in isolation. Adding a new agent (e.g., a payroll agent) requires only a new node and one new edge — no modification to existing nodes.

**Trade-offs:** LangGraph adds a dependency and a compilation step. For a system with only four agents the overhead is non-trivial compared to a simple `match intent: ...` dispatch. The framework pays off at larger scales or when streaming, checkpointing, or parallel node execution is needed.

---

### SQLite vs Production Database

**What:** The system persists all state — STM, LTM, audit logs, sessions — in a single SQLite file (`hr_system.db`).

**Why:** SQLite requires zero infrastructure. Local development, CI, and demo runs need only the Python package; no Postgres service or Docker Compose setup is required. The entire database is a single file that can be deleted and recreated with one `create_all_tables()` call.

**Trade-offs:** SQLite allows only one writer at a time. Under concurrent load (multiple uvicorn workers or async writers hitting the same table) write operations serialize and may time out. There is no native JSON column type, so structured data (memory content, agent responses) is stored as serialized JSON text. SQLite is not suitable for multi-instance deployments.

---

### Confidence Threshold 0.6

**What:** If `IntentClassifier` returns a confidence score below 0.6, the request is unconditionally routed to `ClarificationAgent` rather than acting on the low-confidence intent.

**Why:** 0.6 is the midpoint between pure chance (0.25 for four categories) and high certainty (0.9+). Below this level, the model is closer to guessing than knowing; acting on such a classification risks routing a scheduling request to the leave agent, for example. The threshold was chosen to minimize wrong-agent routes while keeping clarification requests rare for clearly-worded inputs.

**Trade-offs:** A higher threshold (e.g., 0.8) would mean more clarification requests even for reasonable inputs, degrading user experience. A lower threshold (e.g., 0.4) would allow frequent misroutes. 0.6 is configurable — the `LTM_SIGNIFICANCE_THRESHOLD` env var exists as a precedent; a `CLASSIFIER_CONFIDENCE_THRESHOLD` env var could be added for the same reason without a code change.

---

### Two-Tier Memory Design (STM + LTM)

**What:** Short-Term Memory (STM) stores raw interaction entries per session with a 24-hour TTL. Long-Term Memory (LTM) stores significance-filtered entries indefinitely, ordered by score.

**Why:** This mirrors how human memory works. STM gives the classifier recent conversational context within a session (e.g., "I asked about leave 2 turns ago") without polluting it with every interaction from months ago. LTM retains only meaningful events — approved leave, compliance decisions — that are worth surfacing in future sessions for the same user. The two tiers have very different access patterns: STM is always read (low latency, keyed by session), LTM is only read when a user has prior history (less frequent, ordered by score).

**Trade-offs:** Two tables and two manager classes add complexity. A single "memory" table with a TTL column would be simpler but would require the retrieval query to distinguish session-recent entries from cross-session significant ones, conflating two different semantic concepts.

---

### Significance Scoring Formula

**What:** Each interaction is scored before deciding whether to promote it from STM to LTM:

```
score = 0.0
score += 0.4  if intent in {"leave_request", "compliance_query"}
score += 0.3  if had_decision
score += 0.2  if had_entities
score += 0.1  if needed_clarification
score = min(score, 1.0)
```

Promotion threshold: `score >= 0.6` (configurable via `LTM_SIGNIFICANCE_THRESHOLD`).

**Why each weight:**

| Factor | Weight | Reasoning |
|--------|--------|-----------|
| High-priority intent | 0.4 | Leave requests and compliance queries are the most consequential HR actions. An employee will likely want to reference "when did I take leave" or "what does the policy say" in future sessions. |
| Decision made | 0.3 | A request that produced an approval or denial is more important to recall than one that was left unanswered. Decisions have real-world consequences. |
| Entities present | 0.2 | Structured data (days, dates, policy names) makes the record reusable by future agents. A structured memory can seed default values in follow-up requests. |
| Clarification needed | 0.1 | An interaction that required clarification signals the user struggled to express their intent — worth a small signal to help future classification. |

**Trade-offs:** The weights are hand-tuned heuristics. A ML-based scoring model trained on HR interaction data would be more accurate. The additive formula also means a scheduling request with no decision and no entities scores 0.0 and is never promoted — which is intentional but means scheduling history is not retained unless entities are present.

---

## 3. Agent Architecture

### Orchestrator Flow

The `Orchestrator` class owns the compiled LangGraph graph and a single SQLite session shared across memory and audit operations. On each `run()` call, it invokes the graph synchronously via `graph.invoke()` inside an async task, passing an initial `OrchestrationState`.

```
OrchestrationState (TypedDict):
  user_id        str
  session_id     str
  message        str
  memory_context MemoryContext   ← injected by retrieve_memory_node
  intent_result  IntentResult    ← injected by classify_intent_node
  agent_response AgentResponse   ← injected by the routed agent node
  employee_data  dict            ← injected by orchestrator from mock_data
  start_time     float           ← set at graph entry
  error          str | None
```

The conditional edge after `classify_intent_node` reads `state["intent_result"].intent` and routes to one of four agent nodes. All four converge to `finalize_node`, which commits memory and audit records before the graph ends.

### Context Injection

`retrieve_memory_node` calls `MemoryManager.get_context(user_id, session_id, current_intent)`, which returns up to 3 recent STM entries and up to 5 LTM entries ordered by significance. The `IntentClassifier.classify()` method receives this `MemoryContext` and embeds the last 3 STM entries as conversation history in the classification prompt, giving the LLM prior-turn context for disambiguation.

Employee data is injected directly by the orchestrator from `mock_data.employees.get_employee(user_id)` before graph invocation. If the employee is not found, a default empty dict is passed and agents degrade gracefully.

### Retry and Timeout Logic

Every LLM call point uses two layers of protection:

```
Layer 1 — Timeout:   asyncio.wait_for(llm_call, timeout=N)
Layer 2 — Retry:     @async_retry(max_attempts=3, delay=1.0)

Classifier: timeout=10s, wrapped in async_retry via _call_llm_with_timeout()
Sub-agents: timeout=30s on the full _execute() call; @async_retry() on _invoke_llm()
Endpoint:   asyncio.wait_for(orchestrator.run(...), timeout=30) → HTTP 504
```

The `async_retry` decorator (`utils/retry.py`) logs a WARNING on each retry attempt with the attempt number, function name, and exception message. After `max_attempts` exhausted, it re-raises the last exception. The orchestrator wraps `graph.invoke()` in a broad try/except and returns a safe `AgentResponse(status="failed")` for any uncaught exception, ensuring the endpoint always receives a structured object rather than a raw exception.

---

## 4. Memory System

### Short-Term Memory (STM)

**What it stores:** One row per interaction, containing the intent, entities, agent response, and status, serialized as JSON in the `content` column.

**Schema fields:** `id`, `user_id`, `session_id`, `content` (JSON text), `timestamp`, `expires_at`.

**When it expires:** Each entry's `expires_at` is set to `utcnow() + timedelta(hours=24)`. The `expire_old_entries()` method deletes rows where `expires_at < utcnow()`. This method is called automatically on each `get_recent()` call, so expired rows are lazily pruned on access.

**Why 24 hours:** Within a single work day a user may revisit the same request (e.g., check leave balance, then apply). Beyond 24 hours the session context is stale. A shorter TTL (e.g., 1 hour) would lose same-day context; a longer TTL would grow the STM table without benefit.

---

### Long-Term Memory (LTM)

**What it stores:** Promoted interactions with a significance score ≥ 0.6, persisted indefinitely.

**Schema fields:** `id`, `user_id`, `content` (JSON text), `significance_score`, `created_at`, `last_accessed`.

**Retrieval:** `LTMManager.get_relevant(user_id, intent)` returns the top 5 entries for the user, ordered by `significance_score DESC`. The `last_accessed` timestamp is updated on each retrieval.

**Promotion criteria:** Called from `finalize_node` after each successful interaction. `LTMManager.calculate_significance()` computes the score; if `>= threshold`, `promote_from_stm()` inserts a new LTM row.

---

### Worked Example: Leave Approval

```
Input: user_id="emp_001", message="I need 5 days off next week"

1. classify_intent_node
   → intent="leave_request", confidence=0.91, entities={"days": 5}

2. leave_node
   → employee leave_balance=18, days_requested=5
   → 5 ≤ 18, approved; new_balance=13
   → AgentResponse(status="success", agent_name="LeaveAgent",
                   data={"days_approved": 5, "new_balance": 13})

3. finalize_node — significance calculation
   intent = "leave_request"    → +0.4
   had_decision = True          → +0.3
   had_entities = True          → +0.2
   needed_clarification = False → +0.0
   ─────────────────────────────────────
   score = 0.9  ≥  0.6 threshold → PROMOTED to LTM

4. STM entry written (24h TTL)
5. LTM entry written (significance_score=0.9)
6. Audit record appended (immutable)
```

---

### Worked Example: Ambiguous Request (Not Promoted)

```
Input: user_id="emp_001", message="I need some help"

1. classify_intent_node
   → intent="clarification_needed", confidence=0.38 (< 0.6 threshold)

2. clarification_node
   → "Could you share whether this is about scheduling, leave,
      or an HR policy?"
   → AgentResponse(status="clarification_needed")

3. finalize_node — significance calculation
   intent = "clarification_needed" → +0.0
   had_decision = False             → +0.0
   had_entities = False             → +0.0
   needed_clarification = True      → +0.1
   ─────────────────────────────────────
   score = 0.1  <  0.6 threshold → NOT promoted

4. STM entry written (24h TTL only)
```

---

## 5. Audit Log

### Why Append-Only

HR systems are subject to compliance and legal obligations. Every action taken on behalf of an employee — leave approvals, policy lookups, schedule changes — must be permanently traceable. If audit records could be modified or deleted, the audit trail would be meaningless: an incorrect routing decision or a system error could be silently erased. Append-only design ensures that even internal code cannot retroactively alter the record of what the system did.

### How It Is Enforced

Enforcement is applied at two independent layers so that neither a code change nor a direct ORM call can bypass it:

**Layer 1 — SQLAlchemy ORM event listeners** (`database/models.py`):
```python
event.listen(AuditLog, "before_update", prevent_audit_log_update)
event.listen(AuditLog, "before_delete", prevent_audit_log_delete)
```
Both handlers raise `ValueError` immediately, before the SQL is emitted. Any ORM-level `session.flush()` that would trigger an UPDATE or DELETE on `AuditLog` raises an exception.

**Layer 2 — Class design** (`database/audit_log.py`):
`AuditLogger` exposes only two public methods: `log_request()` (INSERT) and `get_logs()` (SELECT). There are no `update_*` or `delete_*` methods. A developer reading the class cannot accidentally call a mutation method that doesn't exist.

**Fallback — JSONL file:**
If all 3 DB retry attempts fail, `_write_fallback()` appends the entry as a JSON line to `audit_fallback.jsonl` in the project root (opened in `"a"` mode). Records are never silently dropped.

### Fields Captured

| Field | Type | Purpose |
|-------|------|---------|
| `id` | UUID string | Unique, immutable record identifier |
| `timestamp` | DateTime UTC | When the request was processed |
| `user_id` | String | Which employee made the request |
| `session_id` | String | Which session it belonged to |
| `raw_request` | Text | Original message text |
| `classified_intent` | String | What the LLM classified the request as |
| `confidence_score` | Float | LLM classification confidence (0–1) |
| `agent_routed_to` | String | Which agent handled the request |
| `agent_response` | Text | Full agent response text |
| `memory_context_used` | Boolean | Whether prior memory influenced routing |
| `processing_time_ms` | Integer | End-to-end latency |
| `status` | String | Outcome: success / failed / clarification_needed |

---

## 6. Known Bugs Fixed

### Bug 1 — Wrong FastAPI Middleware Import Path

**Location:** `main.py:18`

**Bug:** `from fastapi.middleware.base import BaseHTTPMiddleware`

FastAPI does not re-export `BaseHTTPMiddleware` from `fastapi.middleware.base`. This module path does not exist in any released version of FastAPI, causing an immediate `ModuleNotFoundError` at startup and preventing the server from starting or tests from being collected.

**Fix:** `from starlette.middleware.base import BaseHTTPMiddleware`

FastAPI is built on Starlette and `BaseHTTPMiddleware` lives in Starlette's namespace. FastAPI re-exports many Starlette components, but not this one.

---

### Bug 2 — `db.rollback()` Called Before First Insert Attempt

**Location:** `database/audit_log.py`, `log_request()` retry loop

**Bug:**
```python
for attempt in range(3):
    try:
        self.db.rollback()   # ← called BEFORE the insert, including on attempt 0
        audit_row = AuditLog(...)
        self.db.add(audit_row)
        self.db.commit()
```

The orchestrator's database session is shared across memory storage and audit logging within a single request. Calling `rollback()` unconditionally at the start of attempt 0 discards any pending (but not yet committed) transaction from that shared session — for example, a successful STM write that was staged just before `log_request()` was called. This silently cancels valid work on every single audit insert.

**Fix:** Move `self.db.rollback()` into the `except` block, so it only runs after a failure to clear the broken transaction state before retrying:
```python
for attempt in range(3):
    try:
        audit_row = AuditLog(...)
        self.db.add(audit_row)
        self.db.commit()
        ...
    except Exception as exc:
        self.db.rollback()   # ← only after failure
        ...
```

---

### Bug 3 — `pytest-asyncio` Not Installed

**Location:** `requirements.txt` / environment

**Bug:** `pytest-asyncio` was listed in `requirements.txt` but not installed in the active environment. As a result, `pytest.ini`'s `asyncio_mode = auto` was treated as an unknown configuration option (warning shown), async test functions were not collected as coroutines, and `collected 0 items / 1 error` was reported.

**Fix:** `python -m pip install pytest-asyncio` (version 1.4.0 installed).

---

### Bug 4 — Test Suite Used Fake API Key for Real LLM Calls

**Location:** `tests/test_pipeline.py`

**Bug:**
```python
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")  # runs at import time
...
async with lifespan(app):   # load_dotenv() runs here — but key is already set
```

`os.environ.setdefault` only sets the variable if it is not already present. When `lifespan()` later called `load_dotenv()`, the key was already set to `"test-key-for-tests"`, so `load_dotenv()` skipped it. The orchestrator then built `ChatOpenAI` with a fake key. While `IntentClassifier.classify` was monkeypatched, the sub-agents (LeaveAgent, etc.) still made real API calls via `_invoke_llm`, which failed with authentication errors. After 3 retry attempts (3 seconds of delay), the orchestrator returned `status="failed"`, breaking assertions that expected `"success"` or `"clarification_needed"`.

**Fix:** Call `load_dotenv()` at the top of the test module, before `setdefault`, so the real key from `.env` is loaded first:
```python
from dotenv import load_dotenv
load_dotenv()                                              # real key loaded first
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-tests")  # CI fallback only
```

---

## 7. Known Limitations & Trade-offs

| Limitation | Detail |
|------------|--------|
| **Mock sub-agents** | Leave balance decisions, scheduling confirmations, and compliance answers are implemented with mock data (`mock_data/employees.py`). No real HR system (Workday, BambooHR, Google Calendar) is integrated. Leave balance is not actually deducted from any database. |
| **SQLite scalability** | One writer at a time. Multi-worker deployments (e.g., `uvicorn --workers 4`) will serialize all DB writes, causing latency under load. Not suitable for > ~50 concurrent users. |
| **LLM latency** | Each request makes at least one LLM call (classification, ~500ms). Requests routed to the clarification agent make a second call. End-to-end latency is 1–4 seconds depending on OpenAI API response time. |
| **Single-node LangGraph** | The graph runs in a single Python process. There is no checkpointing, persistence of graph state across restarts, or parallel node execution. |
| **No authentication** | All five endpoints are publicly accessible with no authentication or authorization layer. Any client can query any employee's memory or audit logs by passing their `user_id`. |
| **Offset-based pagination** | `GET /audit` uses `LIMIT/OFFSET` pagination. At large offsets (e.g., page 10,000) the query must scan and skip all preceding rows, degrading linearly. Cursor-based pagination would be required at scale. |
| **Shared DB session** | The orchestrator holds a single SQLAlchemy session for the lifetime of the process. Memory writes and audit writes share this session. Under high concurrency this becomes a bottleneck and a source of transaction interference. |
| **Synchronous DB calls in async handlers** | SQLAlchemy ORM calls (in `STMManager`, `LTMManager`, `AuditLogger`) are synchronous and block the event loop. For correctness at scale, these should use `asyncio.to_thread()` or an async SQLAlchemy driver. |

---

## 8. Setup Instructions

### Prerequisites

- Python 3.10 or later
- An OpenAI API key (GPT-4o-mini access)

### Local Setup

```bash
# 1. Clone or enter the project directory
cd hr_agent_system

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env        # Windows
# or: cp .env.example .env   # macOS/Linux
# Then edit .env and set OPENAI_API_KEY=sk-...

# 5. Start the server
uvicorn main:app --reload
# Server starts at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Expected output: 4 tests collected, all pass (tests make real LLM calls; requires a valid `OPENAI_API_KEY` in `.env`).

### Sample curl Commands

**Process an HR request:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"user_id": "emp_001", "message": "I need 5 days off next week"}'
```

**Check service health:**
```bash
curl http://localhost:8000/health
```

**Retrieve audit logs (paginated):**
```bash
curl "http://localhost:8000/audit?user_id=emp_001&limit=10&offset=0"
```

**Retrieve short-term memory:**
```bash
curl "http://localhost:8000/memory/stm?user_id=emp_001&session_id=<session_id>"
```

**Retrieve long-term memory:**
```bash
curl "http://localhost:8000/memory/ltm?user_id=emp_001&limit=5"
```

---

## 9. Future Improvements

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| High | **Real HR system integrations** | Replace mock agents with actual API calls to Workday (leave), Google Calendar (scheduling), and a policy knowledge base (compliance). The agent interface is already well-defined; only `_execute()` needs to change. |
| High | **JWT authentication middleware** | All endpoints are currently open. A FastAPI `Depends` middleware verifying a Bearer token would protect employee data with a one-line change per endpoint. |
| High | **PostgreSQL + connection pooling** | Replace SQLite with PostgreSQL via SQLAlchemy's async driver (`asyncpg`). Enables concurrent writes, JSON column types, and multi-worker deployments. |
| Medium | **Async SQLAlchemy** | Replace synchronous ORM calls with `async_session` and `await session.execute(...)` throughout. Eliminates event-loop blocking on every DB operation. |
| Medium | **Streaming responses** | LangGraph supports token-level streaming. Exposing a `/process/stream` SSE endpoint would allow the UI to display the agent's response word-by-word, reducing perceived latency. |
| Medium | **Per-user LTM eviction** | Add a maximum LTM entry count per user (e.g., 100). When exceeded, evict the lowest-significance entries. Prevents unbounded LTM growth for long-tenure users. |
| Low | **Cursor-based pagination on `/audit`** | Replace `LIMIT/OFFSET` with a keyset cursor (`id > last_seen_id`) for stable, O(1) pagination at any offset. |
| Low | **Pinned dependency versions** | `requirements.txt` currently has no version pins. Pinning ensures reproducible builds across environments and CI. |
| Low | **Distributed LangGraph execution** | LangGraph supports remote graph nodes. Splitting the classifier and each sub-agent into separate services would enable independent scaling and deployment. |
