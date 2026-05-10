"""Pydantic request/response schemas for the HTTP contract."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


MemoryType = Literal["fact", "preference", "opinion", "event"]


class Message(BaseModel):
    # iter 6 (Codex P2.3): per-field length bounds. Without these, a single
    # 5MB content field passes the 1MB Content-Length middleware (it doesn't
    # — the body parser would catch it — but a turn split into many large
    # messages can each be sub-1MB while collectively producing an enormous
    # extraction prompt). 32KB per content is generous for normal chat and
    # cheap to enforce at the framework boundary.
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., max_length=32_000)
    name: str | None = Field(default=None, max_length=128)


class TurnIn(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    user_id: str | None = Field(default=None, max_length=256)
    # max_length on list[Message] caps the list size in pydantic v2 (replaces
    # the v1 `max_items` keyword). 50 messages per turn is well above any
    # reasonable conversation chunk; an extraction prompt with more than that
    # is a bug or abuse, not a legitimate use case.
    messages: list[Message] = Field(..., min_length=1, max_length=50)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnOut(BaseModel):
    id: str


class RecallIn(BaseModel):
    query: str
    session_id: str
    user_id: str | None = None
    # Iter 5 (review fix P0.2): bound max_tokens. Without an upper bound a
    # malicious or buggy client can send max_tokens=10_000_000 and force the
    # context-assembly loop to materialise enormous strings before the trip
    # back to the client. 8192 is generous (fits a Sonnet context window
    # comfortably). gt=0 prevents negative values; max_tokens=0 stays valid
    # via the explicit 200+empty path in recall().
    max_tokens: int = Field(default=1024, ge=0, le=8192)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallOut(BaseModel):
    context: str
    citations: list[Citation]


class SearchIn(BaseModel):
    query: str
    session_id: str | None = None
    user_id: str | None = None
    # Iter 5 (review fix P0.2): bound limit. Same DOS reasoning as max_tokens.
    # 100 is the canonical max-page-size for paginated REST APIs.
    limit: int = Field(default=10, gt=0, le=100)


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchOut(BaseModel):
    results: list[SearchResult]


class MemoryRecord(BaseModel):
    id: str
    type: MemoryType
    key: str
    value: str
    confidence: float
    source_session: str | None
    source_turn: str | None
    created_at: datetime
    updated_at: datetime
    supersedes: str | None
    active: bool


class MemoriesOut(BaseModel):
    memories: list[MemoryRecord]
