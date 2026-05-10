from __future__ import annotations

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.requests import Request

from .config import configure_logging, get_settings
from .db import close_engine, init_schema
from .models import (
    MemoriesOut,
    RecallIn,
    RecallOut,
    SearchIn,
    SearchOut,
    SearchResult,
    TurnIn,
    TurnOut,
)
from .recall import recall as do_recall
from .recall import search as do_search
from .storage import (
    delete_session,
    delete_user,
    list_user_memories,
    store_turn,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_schema()
    logger.info(
        "memory service ready (model=%s, embedding=%s)",
        settings.extraction_model,
        settings.embedding_model,
    )
    yield
    await close_engine()


app = FastAPI(title="memory-service", version="0.1.0", lifespan=lifespan)


# Iter 5 (review fix P3.3): request-id correlation. The contextvar is set by
# the middleware on every request; library code (storage.py, recall.py) reads
# it via the helper below for `extra={"request_id": ...}` in log calls. Full
# structured-log migration (every logger.* call) is left as future work; the
# minimum viable correlation is one info-level log per major handler so a
# given request can be traced through extraction → store → recall in CI logs.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Public read for storage.py / recall.py; returns '-' outside a request."""
    return _request_id_ctx.get()


@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    """413 oversized payloads at the front door rather than buffering them
    in memory. 1 MB is generous for our worst-case payload.

    iter 5 (review fix P3.2): added.
    iter 6 (Codex P3.2): guard malformed Content-Length headers. The previous
    `int(content_length)` would raise ValueError on a non-numeric header,
    surfacing as a 500 Internal Server Error from the unhandled-exception
    handler. Return a clear 400 instead — the header is the client's
    responsibility, not ours.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return Response(
                status_code=400,
                content=b'{"detail":"invalid Content-Length"}',
                media_type="application/json",
            )
        if size > 1_000_000:
            return Response(
                status_code=413,
                content=b'{"detail":"payload too large"}',
                media_type="application/json",
            )
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Iter 5 (review fix P3.3): generate an 8-char request id, expose it via
    `request.state.request_id`, the contextvar (for non-FastAPI callees), and
    the `X-Request-ID` response header."""
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    token = _request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
    finally:
        _request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


def require_auth(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.memory_auth_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    # Iter 5 (review fix P2.3): constant-time comparison to avoid leaking the
    # configured token via timing side-channel. compare_digest short-circuits
    # only on length difference; for equal-length inputs it scans the entire
    # byte string regardless of where they diverge.
    if not secrets.compare_digest(token, settings.memory_auth_token):
        raise HTTPException(status_code=401, detail="invalid token")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 422 by default; spec says malformed input should be 4xx, not crash. 422 is fine.
    return JSONResponse(
        status_code=422, content={"detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "internal error"}
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/turns", status_code=status.HTTP_201_CREATED, response_model=TurnOut)
async def post_turn(payload: TurnIn, _auth: None = Depends(require_auth)) -> TurnOut:
    turn_id = await store_turn(
        session_id=payload.session_id,
        user_id=payload.user_id,
        messages=payload.messages,
        timestamp=payload.timestamp,
        metadata=payload.metadata,
    )
    return TurnOut(id=turn_id)


@app.post("/recall", response_model=RecallOut)
async def post_recall(payload: RecallIn, _auth: None = Depends(require_auth)) -> RecallOut:
    context, citations = await do_recall(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        max_tokens=payload.max_tokens,
    )
    return RecallOut(context=context, citations=citations)


@app.post("/search", response_model=SearchOut)
async def post_search(payload: SearchIn, _auth: None = Depends(require_auth)) -> SearchOut:
    results = await do_search(
        query=payload.query,
        session_id=payload.session_id,
        user_id=payload.user_id,
        limit=payload.limit,
    )
    return SearchOut(
        results=[
            SearchResult(
                content=r["content"],
                score=r["score"],
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                metadata=r["metadata"],
            )
            for r in results
        ]
    )


@app.get("/users/{user_id}/memories", response_model=MemoriesOut)
async def get_user_memories(
    user_id: str, _auth: None = Depends(require_auth)
) -> MemoriesOut:
    memories = await list_user_memories(user_id)
    return MemoriesOut(memories=memories)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session_endpoint(
    session_id: str, _auth: None = Depends(require_auth)
) -> Response:
    await delete_session(session_id)
    return Response(status_code=204)


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_endpoint(
    user_id: str, _auth: None = Depends(require_auth)
) -> Response:
    await delete_user(user_id)
    return Response(status_code=204)
