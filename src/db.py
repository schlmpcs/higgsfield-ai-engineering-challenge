from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from .config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_id TEXT,
    messages JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_text TEXT NOT NULL DEFAULT '',
    embedding VECTOR(__EMBEDDING_DIM__),
    fts_tsv TSVECTOR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_turns_fts ON turns USING GIN(fts_tsv);
CREATE INDEX IF NOT EXISTS idx_turns_embedding
    ON turns USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    session_id TEXT,
    type TEXT NOT NULL CHECK (type IN ('fact', 'preference', 'opinion', 'event')),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_session TEXT,
    source_turn UUID REFERENCES turns(id) ON DELETE CASCADE,
    supersedes UUID REFERENCES memories(id) ON DELETE SET NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    embedding VECTOR(__EMBEDDING_DIM__),
    fts_tsv TSVECTOR,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_active
    ON memories(user_id, active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_memories_user_key
    ON memories(user_id, key);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN(fts_tsv);
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING hnsw (embedding vector_cosine_ops);

-- Concurrency guard for supersession (iter 5 / review fix P0.1; iter 6
-- Codex review fix P0.1).
--
-- Two simultaneous /turns for the same (user_id, key) could both pass the
-- "find existing active row" check, both classify as `supersede`, and both
-- INSERT new active rows. This partial unique index forces serialisation
-- at the DB layer: the second INSERT raises IntegrityError, which the
-- storage layer catches and resolves by re-reading the now-winning row.
--
-- iter 6: the original index covered ALL memory types and silently broke
-- co-active opinion arcs (multiple "opinion_typescript" rows), multiple
-- pets sharing the canonical `pets` key, and `different_subtopic`
-- historical facts where the contradiction classifier deliberately keeps
-- both rows active.
--
-- Two-step narrowing:
--   1) `type IN ('fact', 'preference')` — opinions / events bypass it.
--   2) `COALESCE(metadata->>'cardinality', 'singleton') = 'singleton'`
--      lets a fact/preference flagged as multi-cardinality (e.g. multiple
--      pets, multiple allergies) coexist. Existing rows without metadata
--      default to 'singleton' so the race-condition protection still
--      covers all pre-existing data — only memories explicitly marked
--      `cardinality: "multiple"` (P2.1 schema) are exempted.
--
-- DROP-then-CREATE because `CREATE … IF NOT EXISTS` is a no-op when an
-- index already exists with a different predicate — without the DROP,
-- schema bootstrappers on existing volumes silently keep the older index.
DROP INDEX IF EXISTS uniq_active_user_key;
CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_user_key
    ON memories (user_id, key)
    WHERE active = TRUE
      AND type IN ('fact', 'preference')
      AND COALESCE(metadata->>'cardinality', 'singleton') = 'singleton';

CREATE OR REPLACE FUNCTION fts_trigger_memories() RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_tsv := to_tsvector(
        'english',
        COALESCE(NEW.key, '') || ' ' || COALESCE(NEW.value, '')
    );
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memories_fts ON memories;
CREATE TRIGGER trg_memories_fts
    BEFORE INSERT OR UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION fts_trigger_memories();

CREATE OR REPLACE FUNCTION fts_trigger_turns() RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_tsv := to_tsvector('english', COALESCE(NEW.raw_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_turns_fts ON turns;
CREATE TRIGGER trg_turns_fts
    BEFORE INSERT OR UPDATE ON turns
    FOR EACH ROW EXECUTE FUNCTION fts_trigger_turns();
"""


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_schema(retries: int = 5, base_delay: float = 1.0) -> None:
    """Apply DDL with bounded exponential-backoff retries.

    iter 6 (Codex P2.2). docker-compose's `depends_on: service_healthy` covers
    the local case, but the service may also boot against an external DB (CI,
    cloud) where the listener can flap during the first few seconds. Without
    retries, a transient `OperationalError` during the very first connection
    crashes the FastAPI lifespan and the container exits.

    Backoff: 1s, 2s, 4s, 8s, 16s — total worst-case wait ~31s before giving
    up. The final attempt re-raises so a genuine misconfiguration (wrong
    DATABASE_URL, missing pgvector extension privilege, etc.) still surfaces
    loudly rather than booting into a broken state.
    """
    settings = get_settings()
    engine = get_engine()
    ddl = SCHEMA_DDL.replace("__EMBEDDING_DIM__", str(settings.embedding_dim))

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with engine.begin() as conn:
                for stmt in _split_ddl(ddl):
                    if stmt.strip():
                        await conn.execute(text(stmt))
            logger.info("schema initialized (attempt=%d)", attempt + 1)
            return
        except Exception as e:
            last_exc = e
            if attempt == retries - 1:
                logger.error(
                    "schema init failed after %d attempts: %s", retries, e
                )
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "schema init attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1, retries, e, delay,
            )
            await asyncio.sleep(delay)
    # Defensive — the loop either returns or raises; this is unreachable.
    if last_exc is not None:
        raise last_exc


def _split_ddl(ddl: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_func = False
    for line in ddl.splitlines():
        stripped = line.strip()
        if stripped.startswith("CREATE OR REPLACE FUNCTION") or stripped.startswith(
            "CREATE FUNCTION"
        ):
            in_func = True
        buf.append(line)
        if in_func:
            if stripped.endswith("$$ LANGUAGE plpgsql;"):
                parts.append("\n".join(buf))
                buf = []
                in_func = False
        elif stripped.endswith(";"):
            parts.append("\n".join(buf))
            buf = []
    if buf:
        parts.append("\n".join(buf))
    return parts


async def close_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
