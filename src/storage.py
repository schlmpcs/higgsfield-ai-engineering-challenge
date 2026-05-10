"""Persistence layer: turns + extracted memories.

Synchronous-correctness contract: after `store_turn` returns, the turn and all
extracted memories are visible to recall queries. We achieve this with a single
transaction per turn; commit happens before the HTTP response.

Iter 3 — fact evolution: when an extracted `fact` or `preference` memory shares
a (user_id, key) with an existing active row, we dispatch a small LLM call to
classify the relationship as `supersede`, `consistent`, or `different_subtopic`,
then act accordingly inside the same transaction. Opinions and events skip
this step entirely: opinions evolve as an arc (we want recall to be able to
surface 'previously thought X, now thinks Y'); events are time-bound and do
not supersede each other by definition.

Failure mode: if the contradiction-detection LLM call fails for any reason,
we default to `different_subtopic` (keep both rows active) — a broken
classifier must never destroy data. We complete the turn with the new memory
inserted as-is rather than rolling back, so the rest of the extracted batch
still lands.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .db import session_scope
from .embeddings import embed_batch, embed_text
from .extraction import ExtractedMemory, extract_memories
from .llm_pipeline import classify_contradiction
from .models import Message, MemoryRecord

logger = logging.getLogger(__name__)


def _flatten_messages_to_text(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.role
        if m.name and m.role == "tool":
            role = f"tool[{m.name}]"
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts)


def _to_pgvector(vec: list[float]) -> str:
    """pgvector ASCII format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


_SUPERSEDABLE_TYPES = {"fact", "preference"}


async def _resolve_supersession(
    session,
    user_id: str | None,
    mem: ExtractedMemory,
) -> tuple[str, str | None] | None:
    """Decide what to do for a single extracted memory in light of existing
    active memories with the same (user_id, key).

    Returns:
      - ("insert", None): no collision OR LLM said different_subtopic — INSERT new row, no supersedes link
      - ("insert", old_id): LLM said supersede — INSERT new row pointing at old_id; caller will mark old_id inactive
      - ("skip", None): LLM said consistent — drop the new memory entirely

    Only called for fact/preference; opinions and events bypass this path.
    The DB read happens inside the same transaction as the eventual write,
    so the active set is consistent.
    """
    if user_id is None or mem.type not in _SUPERSEDABLE_TYPES:
        return ("insert", None)

    # iter 6 (Codex P2.1): cardinality short-circuit. If extraction tagged this
    # memory as `multiple` (e.g. allergies, hobbies, multi-element preferences),
    # skip supersession even though the type is fact/preference. The narrowed
    # unique index from P0.1 only enforces uniqueness on type IN
    # ('fact','preference'), so a `multiple`-cardinality preference would still
    # collide if its `key` matched another active row. Trust extraction's
    # signal and keep both rows. The default in `_parse_extraction_response`
    # is "singleton" for fact/preference, so this branch only fires when the
    # LLM explicitly classified the memory as multi-valued.
    if (mem.metadata or {}).get("cardinality") == "multiple":
        return ("insert", None)

    # Most recent active row with the same key wins as the supersession target.
    # If multiple are active (shouldn't happen post-iter-3, but possible from
    # iter-1/2 data), we collide with the newest one.
    #
    # Iter 5 (review fix P0.1): `FOR UPDATE` takes a row-level lock so a
    # concurrent /turns for the same (user_id, key) blocks here until our
    # transaction completes. Combined with the partial unique index
    # `uniq_active_user_key`, the read-then-write race is closed: even if the
    # other transaction commits first, the unique index will reject our
    # subsequent INSERT and `store_turn` will catch the IntegrityError.
    result = await session.execute(
        text(
            """
            SELECT id, value
            FROM memories
            WHERE user_id = :uid AND key = :key AND active = TRUE
              AND type IN ('fact', 'preference')
            ORDER BY updated_at DESC
            LIMIT 1
            FOR UPDATE
            """
        ),
        {"uid": user_id, "key": mem.key},
    )
    row = result.mappings().first()
    if row is None:
        return ("insert", None)

    decision = await classify_contradiction(
        key=mem.key, old_value=row["value"], new_value=mem.value
    )
    logger.info(
        "supersession check user=%s key=%s action=%s reason=%s",
        user_id, mem.key, decision.action, decision.reason,
    )
    if decision.action == "supersede":
        return ("insert", str(row["id"]))
    if decision.action == "consistent":
        return ("skip", None)
    return ("insert", None)  # different_subtopic (also the safe default)


async def store_turn(
    session_id: str,
    user_id: str | None,
    messages: list[Message],
    timestamp: datetime,
    metadata: dict[str, Any],
) -> str:
    """Persist a turn, run extraction, and persist extracted memories.

    Returns the turn UUID as a string. Raises on DB errors. Extraction and
    contradiction-detection failures are swallowed (logged) — the turn is
    still stored.
    """
    # Iter 5 (review fix P3.3): pull request_id from the FastAPI middleware
    # contextvar so this log line can be correlated with the /recall log
    # emitted later in the same request id. Imported lazily to avoid a
    # circular import at module load (main.py imports storage.py).
    from .main import get_request_id

    request_id = get_request_id()
    raw_text = _flatten_messages_to_text(messages)
    logger.info(
        "store_turn start session=%s user=%s msgs=%d",
        session_id,
        user_id,
        len(messages),
        extra={"request_id": request_id},
    )

    # Run extraction concurrently with embedding the turn text would be ideal,
    # but extraction quality is the bigger lever — keep it sequential and clear.
    extracted = await extract_memories(messages)
    turn_embedding = await embed_text(raw_text)

    mem_embeddings: list[list[float]] = []
    if extracted:
        mem_texts = [f"{m.key}: {m.value}" for m in extracted]
        mem_embeddings = await embed_batch(mem_texts)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                INSERT INTO turns (session_id, user_id, messages, timestamp, metadata, raw_text, embedding)
                VALUES (:session_id, :user_id, CAST(:messages AS JSONB), :timestamp, CAST(:metadata AS JSONB), :raw_text, CAST(:embedding AS VECTOR))
                RETURNING id
                """
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "messages": json.dumps([m.model_dump() for m in messages]),
                "timestamp": timestamp,
                "metadata": json.dumps(metadata),
                "raw_text": raw_text,
                "embedding": _to_pgvector(turn_embedding),
            },
        )
        turn_id: UUID = result.scalar_one()

        for mem, emb in zip(extracted, mem_embeddings):
            decision = await _resolve_supersession(session, user_id, mem)
            if decision is None:
                continue
            action, supersedes_id = decision
            if action == "skip":
                # LLM said this restates an existing memory; don't insert a duplicate.
                continue

            # Iter 5 (review fix P0.1): wrap the UPDATE+INSERT pair in a
            # SAVEPOINT so a unique-index conflict on `uniq_active_user_key`
            # (caused by a concurrent transaction inserting an active row for
            # the same user_id+key first) only rolls back this one memory —
            # the rest of the extracted batch and the turn itself still land.
            #
            # Order matters: when superseding, UPDATE the old row to
            # `active=FALSE` BEFORE inserting the new active row. Postgres
            # checks the partial unique index at INSERT-statement boundary,
            # not transaction-end — flipping the old row first means the
            # index has at most one active row at INSERT time, so the
            # constraint is satisfied for sequential supersession (the
            # Stripe→Notion test still passes). Concurrent supersession
            # is still caught by the unique index because both transactions
            # try to INSERT a row with `active=TRUE` for the same key, and
            # Postgres serialises that.
            try:
                async with session.begin_nested():
                    if supersedes_id is not None:
                        # Mark the prior active row inactive first so the
                        # subsequent INSERT doesn't trip the partial unique
                        # index (uniq_active_user_key) on the still-active
                        # old row. History is preserved; recall filters on
                        # active=TRUE so the stale fact stops surfacing.
                        await session.execute(
                            text(
                                """
                                UPDATE memories
                                SET active = FALSE
                                WHERE id = :sid
                                """
                            ),
                            {"sid": supersedes_id},
                        )

                    insert_result = await session.execute(
                        text(
                            """
                            INSERT INTO memories
                                (user_id, session_id, type, key, value, confidence,
                                 source_session, source_turn, supersedes, embedding,
                                 metadata)
                            VALUES
                                (:user_id, :session_id, :type, :key, :value, :confidence,
                                 :source_session, :source_turn, :supersedes, CAST(:embedding AS VECTOR),
                                 CAST(:metadata AS JSONB))
                            RETURNING id
                            """
                        ),
                        {
                            "user_id": user_id,
                            "session_id": session_id,
                            "type": mem.type,
                            "key": mem.key,
                            "value": mem.value,
                            "confidence": mem.confidence,
                            "source_session": session_id,
                            "source_turn": turn_id,
                            "supersedes": supersedes_id,
                            "embedding": _to_pgvector(emb),
                            # iter 6 (Codex P2.1): persist cardinality/subject
                            # so contradiction decisions become auditable and
                            # `_resolve_supersession` can short-circuit on
                            # `multiple` without re-asking the LLM.
                            "metadata": json.dumps(mem.metadata or {}),
                        },
                    )
                    new_id: UUID = insert_result.scalar_one()
            except IntegrityError as e:
                # iter 6 (Codex P0.1): distinguish the supersession race
                # (expected, recoverable) from genuine schema violations
                # (programmer error, must surface). The partial unique index
                # `uniq_active_user_key` is the only constraint we expect to
                # trip from concurrent /turns; anything else means our INSERT
                # is malformed and silently dropping it would mask a real bug.
                #
                # asyncpg/SQLAlchemy expose the violated constraint name via
                # `e.orig.constraint_name` when the underlying driver supports
                # it; we also fall back to substring-matching the formatted
                # error because some asyncpg paths only stringify it.
                constraint_name = (
                    getattr(getattr(e, "orig", None), "constraint_name", None)
                    or str(getattr(e, "orig", e))
                )
                if "uniq_active_user_key" not in (constraint_name or ""):
                    # Genuine schema violation — propagate so the request fails
                    # loudly. The savepoint already rolled back this memory's
                    # writes; the outer transaction will roll back the whole
                    # turn via session_scope's exception handler.
                    raise

                # Concurrent /turns for the same (user_id, key) won the race
                # and inserted their active row first. Drop ours rather than
                # create a duplicate active row.
                logger.warning(
                    "supersession_race_lost user=%s key=%s constraint=%s — "
                    "concurrent transaction beat us to the active row; "
                    "dropping new memory",
                    user_id, mem.key, constraint_name,
                )
                continue

        return str(turn_id)


async def list_user_memories(user_id: str) -> list[MemoryRecord]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, type, key, value, confidence, source_session,
                       source_turn, created_at, updated_at, supersedes, active
                FROM memories
                WHERE user_id = :user_id
                ORDER BY active DESC, updated_at DESC
                """
            ),
            {"user_id": user_id},
        )
        rows = result.mappings().all()

    return [
        MemoryRecord(
            id=str(r["id"]),
            type=r["type"],
            key=r["key"],
            value=r["value"],
            confidence=float(r["confidence"]),
            source_session=r["source_session"],
            source_turn=str(r["source_turn"]) if r["source_turn"] else None,
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            supersedes=str(r["supersedes"]) if r["supersedes"] else None,
            active=bool(r["active"]),
        )
        for r in rows
    ]


async def delete_session(session_id: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text("DELETE FROM memories WHERE session_id = :sid"),
            {"sid": session_id},
        )
        await session.execute(
            text("DELETE FROM turns WHERE session_id = :sid"),
            {"sid": session_id},
        )


async def delete_user(user_id: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text("DELETE FROM memories WHERE user_id = :uid"),
            {"uid": user_id},
        )
        await session.execute(
            text("DELETE FROM turns WHERE user_id = :uid"),
            {"uid": user_id},
        )
