"""Concurrency tests against a running service.

Targets the supersession race condition closed in iter 5 / review fix P0.1
(partial unique index `uniq_active_user_key` + `FOR UPDATE` lock +
SAVEPOINT-wrapped INSERT). The previous code could leak two active rows
for the same `(user_id, key)` when two `/turns` arrived concurrently with
conflicting facts. This test fires both turns simultaneously and asserts
exactly one active employment row remains.

Runs against the same MEMORY_SERVICE_URL as test_contract.py. Skips
automatically if the service is not reachable.
"""
from __future__ import annotations

import asyncio
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
async def test_concurrent_supersession() -> None:
    """Two `/turns` arrive simultaneously for the same user with conflicting
    employment facts. After both complete, exactly one active employment row
    must remain.

    The race: without the partial unique index + FOR UPDATE, both transactions
    could read the empty active set, both INSERT new active rows, end state
    is two active employment rows for the same user. The fix serialises the
    second INSERT against the unique index; the loser's INSERT raises
    IntegrityError, the SAVEPOINT rolls back, and the rest of that turn's
    batch (and the turn itself) still commit.

    Note: this is the supersession-race test, not a write-conflict-resolution
    test. The 1-active-row guarantee is the contract; *which* fact wins
    depends on whichever transaction lands first, and is intentionally not
    asserted here.
    """
    user_id = "concurrent-user-1-" + uuid.uuid4().hex[:8]
    sess_a = _unique("ca")
    sess_b = _unique("cb")
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        post_a = c.post(
            "/turns",
            json={
                "session_id": sess_a,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I work at Stripe as a senior engineer on payments infra.",
                    },
                    {"role": "assistant", "content": "Cool!"},
                ],
                "timestamp": now,
                "metadata": {},
            },
        )
        post_b = c.post(
            "/turns",
            json={
                "session_id": sess_b,
                "user_id": user_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "I just joined Notion as a Product Manager last week.",
                    },
                    {"role": "assistant", "content": "Congrats!"},
                ],
                "timestamp": now,
                "metadata": {},
            },
        )

        # Fire both concurrently. Both must land 201 even if one's memory
        # extraction loses the supersession race (turn write is unaffected;
        # only the per-memory SAVEPOINT rolls back).
        ra, rb = await asyncio.gather(post_a, post_b)
        assert ra.status_code == 201, ra.text
        assert rb.status_code == 201, rb.text

        r = await c.get(f"/users/{user_id}/memories")
        assert r.status_code == 200, r.text
        memories = r.json()["memories"]

        active_employment = [
            m for m in memories
            if m["key"] == "employment" and m["active"]
        ]
        assert len(active_employment) == 1, (
            f"expected exactly 1 active employment row after concurrent /turns, "
            f"got {len(active_employment)}: {active_employment!r}"
        )

        await c.delete(f"/users/{user_id}")
