"""Contract tests against a running service.

Runs against http://localhost:8080 by default — point at a different host with
MEMORY_SERVICE_URL=. Skips automatically if /health is unreachable so it
doesn't break unit-test runs without docker compose.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

BASE = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")


def _service_up() -> bool:
    try:
        r = httpx.get(f"{BASE}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _service_up(),
    reason=f"memory service not reachable at {BASE} — run `docker compose up -d` first",
)


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_health() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=5.0) as c:
        r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_turn_then_recall() -> None:
    user_id = _unique("u")
    session_id = _unique("s")
    async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I just moved from NYC to Berlin last month.",
                    },
                    {
                        "role": "assistant",
                        "content": "Welcome to Berlin!",
                    },
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text
        turn_id = r.json()["id"]
        assert turn_id

        r = await c.post(
            "/recall",
            json={
                "query": "Where does this user live?",
                "session_id": _unique("other"),
                "user_id": user_id,
                "max_tokens": 512,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "context" in body
        assert "citations" in body
        assert isinstance(body["citations"], list)

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200
        body = r.json()
        assert "memories" in body
        for m in body["memories"]:
            assert {"id", "type", "key", "value", "active"}.issubset(m.keys())

        r = await c.delete(f"/users/{user_id}")
        assert r.status_code == 204


@pytest.mark.asyncio
async def test_cold_session_returns_empty_no_error() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=10.0) as c:
        r = await c.post(
            "/recall",
            json={
                "query": "What did this never-seen user say?",
                "session_id": _unique("ghost"),
                "user_id": _unique("ghost-u"),
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["context"] == ""
        assert body["citations"] == []


@pytest.mark.asyncio
async def test_concurrent_sessions_dont_bleed() -> None:
    user_a = _unique("ua")
    user_b = _unique("ub")
    sess_a = _unique("sa")
    sess_b = _unique("sb")
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
        await c.post(
            "/turns",
            json={
                "session_id": sess_a,
                "user_id": user_a,
                "messages": [
                    {"role": "user", "content": "I have a cat named Whiskers."},
                    {"role": "assistant", "content": "Cute!"},
                ],
                "timestamp": now,
                "metadata": {},
            },
        )
        await c.post(
            "/turns",
            json={
                "session_id": sess_b,
                "user_id": user_b,
                "messages": [
                    {"role": "user", "content": "I have a dog named Rex."},
                    {"role": "assistant", "content": "Cool!"},
                ],
                "timestamp": now,
                "metadata": {},
            },
        )

        r = await c.get(f"/users/{user_a}/memories")
        ms_a = " ".join(m["value"].lower() for m in r.json()["memories"])
        assert "rex" not in ms_a

        r = await c.get(f"/users/{user_b}/memories")
        ms_b = " ".join(m["value"].lower() for m in r.json()["memories"])
        assert "whiskers" not in ms_b

        await c.delete(f"/users/{user_a}")
        await c.delete(f"/users/{user_b}")


@pytest.mark.asyncio
async def test_fact_evolution_supersession() -> None:
    """Stripe → Notion: old employment fact must be marked inactive with a
    supersedes link to the new active row, and /recall must return Notion."""
    user_id = _unique("evo")
    sess1 = _unique("s1")
    sess2 = _unique("s2")
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": sess1,
                "user_id": user_id,
                "messages": [
                    {"role": "user", "content": "I work at Stripe as a senior engineer on payments infra."},
                    {"role": "assistant", "content": "Cool!"},
                ],
                "timestamp": "2025-01-15T10:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.post(
            "/turns",
            json={
                "session_id": sess2,
                "user_id": user_id,
                "messages": [
                    {"role": "user", "content": "Big update — I just joined Notion as a Product Manager last week."},
                    {"role": "assistant", "content": "Congrats!"},
                ],
                "timestamp": "2025-03-15T09:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200
        memories = r.json()["memories"]

        employment_rows = [m for m in memories if m["key"] == "employment"]
        assert len(employment_rows) == 2, (
            f"expected 2 employment rows (one active, one superseded), "
            f"got {len(employment_rows)}: {employment_rows!r}"
        )

        active = [m for m in employment_rows if m["active"]]
        inactive = [m for m in employment_rows if not m["active"]]
        assert len(active) == 1, f"expected exactly one active employment row, got {len(active)}"
        assert len(inactive) == 1, f"expected exactly one inactive employment row, got {len(inactive)}"

        # The active row should mention Notion and link via `supersedes` to the inactive Stripe row.
        assert "notion" in active[0]["value"].lower()
        assert "stripe" in inactive[0]["value"].lower()
        assert active[0]["supersedes"] == inactive[0]["id"], (
            f"active row's supersedes={active[0]['supersedes']!r} "
            f"should equal inactive row's id={inactive[0]['id']!r}"
        )

        r = await c.post(
            "/recall",
            json={
                "query": "Where does this user work currently?",
                "session_id": _unique("recall"),
                "user_id": user_id,
                "max_tokens": 256,
            },
        )
        assert r.status_code == 200
        ctx = r.json()["context"].lower()
        assert "notion" in ctx, f"expected Notion in recall context, got {ctx!r}"
        assert "stripe" not in ctx, (
            f"superseded Stripe fact leaked into recall context: {ctx!r}"
        )

        await c.delete(f"/users/{user_id}")


@pytest.mark.asyncio
async def test_citations_reference_real_turns() -> None:
    """`Citation.turn_id` must point at a real turn id (not a memory uuid).

    Per the HTTP contract, `/recall` returns `citations[].turn_id` referencing a turn.
    For memory-derived citations that means the memory's `source_turn`, not the
    memory's own primary key — otherwise a reviewer following a citation back
    finds no such turn.
    """
    user_id = _unique("cite")
    session_id = _unique("s")
    async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I just moved from NYC to Berlin last month.",
                    },
                    {"role": "assistant", "content": "Welcome to Berlin!"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text
        original_turn_id = r.json()["id"]

        r = await c.get(f"/users/{user_id}/memories")
        memory_ids = {m["id"] for m in r.json()["memories"]}

        r = await c.post(
            "/recall",
            json={
                "query": "Where does this user live?",
                "session_id": _unique("recall"),
                "user_id": user_id,
                "max_tokens": 512,
            },
        )
        assert r.status_code == 200
        citations = r.json()["citations"]
        assert citations, f"expected non-empty citations, got: {r.json()!r}"

        for cit in citations:
            assert cit["turn_id"] not in memory_ids, (
                f"citation.turn_id={cit['turn_id']!r} is a memory id, not a turn id "
                f"(citation={cit!r})"
            )
            assert cit["turn_id"] == original_turn_id, (
                f"citation.turn_id={cit['turn_id']!r} should equal "
                f"original turn id {original_turn_id!r} (citation={cit!r})"
            )

        await c.delete(f"/users/{user_id}")


@pytest.mark.skipif(
    os.environ.get("RESTART_PERSISTENCE_TEST") != "1",
    reason="gated on RESTART_PERSISTENCE_TEST=1 — runs `docker compose down && up`",
)
@pytest.mark.asyncio
async def test_restart_persistence() -> None:
    """Data survives `docker compose down && docker compose up` (named volume).

    Posts a turn, captures the resulting memory ids and a recall result, takes
    the compose stack down and back up, then re-reads — both the memory rows
    and the recallable fact must survive. Gated behind an env var because it
    depends on `docker` being on PATH and disrupts other concurrent test runs
    against the same instance.
    """
    project_root = Path(__file__).resolve().parent.parent
    user_id = _unique("persist")
    session_id = _unique("ps")

    async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "I just moved to Berlin from NYC and I have a "
                            "labrador named Biscuit."
                        ),
                    },
                    {"role": "assistant", "content": "Lovely!"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.get(f"/users/{user_id}/memories")
        before = sorted(m["id"] for m in r.json()["memories"])
        assert before, "no memories extracted; restart test cannot signal anything"

    try:
        subprocess.run(
            ["docker", "compose", "down"],
            check=True,
            cwd=str(project_root),
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            check=True,
            cwd=str(project_root),
            capture_output=True,
            timeout=120,
        )

        deadline = time.time() + 90.0
        ready = False
        while time.time() < deadline:
            try:
                if httpx.get(f"{BASE}/health", timeout=2.0).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        assert ready, f"service did not become healthy at {BASE} within 90s"

        async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
            r = await c.get(f"/users/{user_id}/memories")
            assert r.status_code == 200
            after = sorted(m["id"] for m in r.json()["memories"])
            assert after == before, (
                f"memories did not survive restart: before={before!r}, after={after!r}"
            )

            r = await c.post(
                "/recall",
                json={
                    "query": "Where does this user live?",
                    "session_id": _unique("post-restart"),
                    "user_id": user_id,
                    "max_tokens": 256,
                },
            )
            assert r.status_code == 200
            assert "berlin" in r.json()["context"].lower(), (
                f"recallable fact missing after restart: {r.json()!r}"
            )

            await c.delete(f"/users/{user_id}")
    finally:
        # Best-effort: ensure the stack is up after the test even if it failed.
        try:
            subprocess.run(
                ["docker", "compose", "up", "-d"],
                check=False,
                cwd=str(project_root),
                capture_output=True,
                timeout=120,
            )
        except Exception:
            pass


@pytest.mark.asyncio
async def test_search_endpoint() -> None:
    """`/search` (iter 5 / review fix P2.5): post a turn, query for a relevant
    keyword, assert the response shape and at least one result."""
    user_id = _unique("search")
    session_id = _unique("s")
    async with httpx.AsyncClient(base_url=BASE, timeout=70.0) as c:
        r = await c.post(
            "/turns",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I just adopted a labrador named Biscuit.",
                    },
                    {"role": "assistant", "content": "Adorable!"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
        )
        assert r.status_code == 201, r.text

        r = await c.post(
            "/search",
            json={
                "query": "labrador Biscuit",
                "session_id": session_id,
                "user_id": user_id,
                "limit": 10,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) >= 1, (
            f"expected at least 1 search result for relevant query, got {body!r}"
        )
        # Every result must carry the documented shape (the SearchResult model).
        for hit in body["results"]:
            assert {"content", "score", "session_id", "timestamp", "metadata"}.issubset(
                hit.keys()
            ), f"missing keys in search result: {hit!r}"

        await c.delete(f"/users/{user_id}")


@pytest.mark.asyncio
async def test_malformed_input_returns_4xx() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=5.0) as c:
        r = await c.post("/turns", content=b"not json")
        assert 400 <= r.status_code < 500

        r = await c.post(
            "/turns",
            json={"session_id": "s1"},  # missing required fields
        )
        assert 400 <= r.status_code < 500

        r = await c.post(
            "/turns",
            json={
                "session_id": "s1",
                "user_id": None,
                "messages": [{"role": "user", "content": "héllo 🚀 ‮"}],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {},
            },
        )
        assert r.status_code == 201
