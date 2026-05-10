"""Co-active memory regression tests (iter 6 / Codex review fix P0.2).

These tests lock down the cases the broader-than-intended unique index
silently broke in iter 5:

  - Multiple opinions on the same topic — the opinion arc.
  - Multiple pets sharing a canonical `pets` key.
  - `different_subtopic` historical facts (e.g. "used to work at IDEO,
    now at Notion") where the contradiction classifier deliberately keeps
    both rows active.

If these regress, the unique index has been over-scoped again or the
storage layer's IntegrityError handler is silently dropping rows.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import httpx
import pytest

BASE = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")


def _service_up() -> bool:
    try:
        return httpx.get(f"{BASE}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _service_up(),
    reason=f"memory service not reachable at {BASE} — run `docker compose up -d` first",
)


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_multiple_active_opinions_same_topic() -> None:
    """Opinion arc: two opinions on the same topic must both stay active.

    The fact-evolution flow correctly skips supersession for `opinion` type
    (`_resolve_supersession` returns ('insert', None)). The historical
    bug was that the partial unique index on (user_id, key) WHERE active
    applied to opinions too, so the second active opinion with the same
    `opinion_<topic>` key was silently dropped via the IntegrityError
    handler. Iter 6 narrowed the index to fact/preference only — this
    test pins that behaviour.
    """
    user_id = _unique("opinion-arc")
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": _unique("op-1"),
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "I love TypeScript — it makes large codebases manageable "
                            "and the tooling is great."
                        ),
                    },
                    {"role": "assistant", "content": "Glad to hear it."},
                ],
                "timestamp": "2025-01-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.post(
            "/turns",
            json={
                "session_id": _unique("op-2"),
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Honestly TypeScript generics are getting really annoying lately. "
                            "I'm starting to dislike how much complexity they add."
                        ),
                    },
                    {"role": "assistant", "content": "That's a common frustration."},
                ],
                "timestamp": "2025-01-15T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200
        memories = r.json()["memories"]

        opinion_memories = [
            m
            for m in memories
            if m["type"] == "opinion" and "typescript" in m["key"].lower()
        ]
        active_opinions = [m for m in opinion_memories if m["active"]]
        assert len(active_opinions) >= 2, (
            f"expected >=2 active TypeScript opinions, got {len(active_opinions)}; "
            f"opinion rows: {opinion_memories!r}"
        )

        await c.delete(f"/users/{user_id}")


@pytest.mark.asyncio
async def test_multiple_pets_both_active() -> None:
    """Two pets in separate turns must both stay active.

    `KEY_ALIASES` collapses `pet`/`animal`/`pet_animal` to canonical `pets`,
    so two distinct pet memories naturally share the (user_id, 'pets')
    key. With the old broad unique index, the second pet was silently
    dropped. The narrowed index (fact/preference only via the predicate)
    now only enforces uniqueness on singleton keys.
    """
    user_id = _unique("multi-pet")
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": _unique("pet-1"),
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "My dog Biscuit is a golden retriever, three years old.",
                    },
                    {"role": "assistant", "content": "Biscuit sounds lovely."},
                ],
                "timestamp": "2025-01-01T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.post(
            "/turns",
            json={
                "session_id": _unique("pet-2"),
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I also have a cat named Whiskers — she's a tortoiseshell.",
                    },
                    {"role": "assistant", "content": "A dog and a cat — fun."},
                ],
                "timestamp": "2025-01-15T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200
        memories = r.json()["memories"]

        pet_memories = [
            m
            for m in memories
            if "pet" in m["key"].lower()
            or "biscuit" in m["value"].lower()
            or "whiskers" in m["value"].lower()
            or "dog" in m["value"].lower()
            or "cat" in m["value"].lower()
        ]
        active_pets = [m for m in pet_memories if m["active"]]
        names_in_active = " ".join(m["value"].lower() for m in active_pets)
        assert "biscuit" in names_in_active and "whiskers" in names_in_active, (
            f"expected both Biscuit and Whiskers in active memories; "
            f"active pets: {active_pets!r}"
        )
        assert len(active_pets) >= 2, (
            f"expected >=2 active pet memories, got {len(active_pets)}; "
            f"pet rows: {pet_memories!r}"
        )

        await c.delete(f"/users/{user_id}")


@pytest.mark.asyncio
async def test_different_subtopic_facts_both_active() -> None:
    """Past + current employer in one turn: both should remain in memory.

    When the contradiction classifier returns `different_subtopic` (or its
    safe default in the LLM-failure path), supersession does NOT mark
    either row inactive — both should remain queryable. The previous
    unique index made this impossible whenever the LLM emitted the
    canonical `employment` key for both. With the narrowed index, both
    rows can coexist with `active=TRUE`, or one can carry a more
    specific subkey via the LLM (either is acceptable as long as both
    surface in `/users/{id}/memories`).
    """
    user_id = _unique("history")
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": _unique("hist-1"),
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "I used to work at IDEO before joining Notion as a Product Manager."
                        ),
                    },
                    {"role": "assistant", "content": "Interesting career path!"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200
        memories = r.json()["memories"]

        values = [m["value"].lower() for m in memories]
        assert any("notion" in v for v in values), (
            f"Notion not found in memories: {memories!r}"
        )
        assert any("ideo" in v for v in values), (
            f"IDEO not found in memories: {memories!r}"
        )

        await c.delete(f"/users/{user_id}")
