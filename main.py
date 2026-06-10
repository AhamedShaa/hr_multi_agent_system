"""FastAPI application for the HR Agent System."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from agents.orchestrator import Orchestrator
from config import settings
from database.audit_log import AuditLogger
from database.db import SessionLocal, create_all_tables, engine
from database.models import AuditLog, LTMMemory
from memory.stm import STMManager
from schemas.models import HRRequest, ProcessResponse

logger = logging.getLogger("hr_agent_system")
logging.basicConfig(level=logging.INFO)

orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load settings, initialize tables, and create the singleton orchestrator."""
    global orchestrator
    create_all_tables()
    orchestrator = Orchestrator()
    print("HR Agent System started. Docs at http://localhost:8000/docs")
    try:
        yield
    finally:
        if orchestrator is not None:
            orchestrator.close()
            orchestrator = None


app = FastAPI(title="HR Agent System", version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and processing time for every request."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "method=%s path=%s status=%s processing_time_ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session for endpoint dependencies."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_orchestrator() -> Orchestrator:
    """Return the singleton orchestrator or raise if startup did not initialize it."""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Orchestrator is not initialized",
        )
    return orchestrator


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return validation failures as HTTP 400 responses."""
    request_id = str(uuid.uuid4())
    logger.info("Validation error request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid request", "request_id": request_id},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a safe response for unhandled exceptions without exposing stack traces."""
    request_id = str(uuid.uuid4())
    logger.exception("Unhandled exception request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal error occurred", "request_id": request_id},
    )


@app.post("/process", response_model=ProcessResponse)
async def process_request(
    request_body: HRRequest,
    hr_orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ProcessResponse:
    """Process an HR request through the orchestrated agent pipeline."""
    session_id = request_body.session_id or str(uuid.uuid4())
    message = request_body.message.strip()
    if len(message) < 1 or len(message) > settings.max_message_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must be 1–2000 characters",
        )
    logger.info("POST /process user_id=%s session_id=%s", request_body.user_id, session_id)

    try:
        return await asyncio.wait_for(
            hr_orchestrator.run(request_body.user_id, session_id, message),
            timeout=30,
        )
    except TimeoutError as exc:
        logger.info("POST /process timeout user_id=%s session_id=%s", request_body.user_id, session_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from exc


@app.get("/audit")
async def get_audit_logs(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=settings.audit_page_limit_max),
    offset: int = Query(default=0, ge=0, description="Pagination offset, must be non-negative"),
    intent_filter: str | None = Query(default=None),
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Return paginated audit log entries with optional filters."""
    logger.info("GET /audit user_id=%s limit=%s offset=%s intent=%s", user_id, limit, offset, intent_filter)
    audit_logger = AuditLogger(db)
    entries = audit_logger.get_logs(
        user_id=user_id,
        limit=limit,
        offset=offset,
        intent_filter=intent_filter,
    )
    total = _count_audit_logs(db, user_id=user_id, intent_filter=intent_filter)
    return {"entries": entries, "total": total, "limit": limit, "offset": offset}


@app.get("/memory/stm")
async def get_stm_entries(
    user_id: str = Query(...),
    session_id: str = Query(...),
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Return short-term memory entries for a user session."""
    logger.info("GET /memory/stm user_id=%s session_id=%s", user_id, session_id)
    entries = STMManager(db).get_recent(user_id, session_id)
    if not entries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No STM entries found")
    return {"user_id": user_id, "session_id": session_id, "entries": entries, "count": len(entries)}


@app.get("/memory/ltm")
async def get_ltm_entries(
    user_id: str = Query(...),
    limit: int = Query(default=10, ge=1, le=settings.audit_page_limit_max),
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Return long-term memory entries for a user ordered by significance score."""
    logger.info("GET /memory/ltm user_id=%s limit=%s", user_id, limit)
    rows = db.scalars(
        select(LTMMemory)
        .where(LTMMemory.user_id == user_id)
        .order_by(LTMMemory.significance_score.desc())
        .limit(limit)
    ).all()
    entries = [_deserialize_ltm_entry(row) for row in rows]
    if not entries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No LTM entries found")
    return {"user_id": user_id, "entries": entries, "count": len(entries)}


@app.get("/health")
async def health_check() -> JSONResponse:
    """Return service health and database connectivity status."""
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        content = {
            "status": "healthy",
            "timestamp": timestamp,
            "database": "connected",
            "version": settings.app_version,
        }
        return JSONResponse(status_code=status.HTTP_200_OK, content=content)
    except Exception:
        content = {
            "status": "unhealthy",
            "timestamp": timestamp,
            "database": "disconnected",
            "version": settings.app_version,
        }
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=content)


def _count_audit_logs(db: Session, user_id: str | None, intent_filter: str | None) -> int:
    """Count audit log rows matching optional endpoint filters."""
    query = select(func.count()).select_from(AuditLog)
    if user_id is not None:
        query = query.where(AuditLog.user_id == user_id)
    if intent_filter is not None:
        query = query.where(AuditLog.classified_intent == intent_filter)
    return int(db.scalar(query) or 0)


def _deserialize_ltm_entry(entry: LTMMemory) -> dict[str, Any]:
    """Deserialize a long-term memory ORM row for endpoint output."""
    from json import loads

    data = loads(entry.content)
    data["id"] = entry.id
    data["user_id"] = entry.user_id
    data["significance_score"] = entry.significance_score
    data["created_at"] = entry.created_at.isoformat()
    data["last_accessed"] = entry.last_accessed.isoformat()
    return data
