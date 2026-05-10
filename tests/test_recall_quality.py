"""Self-eval driver: ingests fixtures/recall_eval.json and reports
per-scenario + per-tag recall metrics.

Tags: keyword, semantic_only, semantic_lite, multi_hop, implicit,
implicit_semantic, noise. Higher tag granularity makes it obvious which
ranking technique helped which axis.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

BASE = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")
FIXTURE = Path(__file__).parent.parent / "fixtures" / "recall_eval.json"


def _service_up() -> bool:
    try:
        return httpx.get(f"{BASE}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _service_up(),
    reason=f"memory service not reachable at {BASE}",
)


def _suffix() -> str:
    return uuid.uuid4().hex[:6]


@pytest.mark.asyncio
async def test_recall_fixture_score() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    suffix = _suffix()

    total_expected = 0
    total_found = 0
    forbidden_violations: list[str] = []
    per_scenario: list[tuple[str, int, int]] = []
    per_tag_expected: dict[str, int] = defaultdict(int)
    per_tag_found: dict[str, int] = defaultdict(int)
    noise_total = 0
    noise_clean = 0

    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as c:
        for scenario in data["scenarios"]:
            user_id = f"{scenario['user_id']}-{suffix}"

            for turn in scenario["turns"]:
                sid = f"{turn['session_id']}-{suffix}"
                r = await c.post(
                    "/turns",
                    json={
                        "session_id": sid,
                        "user_id": user_id,
                        "messages": turn["messages"],
                        "timestamp": turn.get(
                            "timestamp",
                            datetime.now(timezone.utc).isoformat(),
                        ),
                        "metadata": {},
                    },
                )
                assert r.status_code == 201, r.text

            scenario_expected = 0
            scenario_found = 0
            for probe in scenario["probes"]:
                sid = f"{probe['session_id']}-{suffix}"
                tag = probe.get("tag", "untagged")
                r = await c.post(
                    "/recall",
                    json={
                        "query": probe["query"],
                        "session_id": sid,
                        "user_id": user_id,
                        "max_tokens": probe.get("max_tokens", 1024),
                    },
                )
                assert r.status_code == 200
                ctx = r.json()["context"].lower()

                expected = probe.get("expected_facts", [])
                any_of = probe.get("any_of_expected", False)
                if expected:
                    if any_of:
                        scenario_expected += 1
                        per_tag_expected[tag] += 1
                        if any(f.lower() in ctx for f in expected):
                            scenario_found += 1
                            per_tag_found[tag] += 1
                    else:
                        for fact in expected:
                            scenario_expected += 1
                            per_tag_expected[tag] += 1
                            if fact.lower() in ctx:
                                scenario_found += 1
                                per_tag_found[tag] += 1

                forbidden = probe.get("must_not_contain", [])
                if forbidden:
                    noise_total += 1
                    if not any(f.lower() in ctx for f in forbidden):
                        noise_clean += 1
                    else:
                        hits = [f for f in forbidden if f.lower() in ctx]
                        forbidden_violations.append(
                            f"{scenario['name']}::{probe['query']!r} contained {hits}"
                        )

            per_scenario.append(
                (scenario["name"], scenario_found, scenario_expected)
            )
            total_expected += scenario_expected
            total_found += scenario_found

            r = await c.delete(f"/users/{user_id}")
            assert r.status_code == 204

    print("\n=== recall self-eval ===", file=sys.stderr)
    print("--- per scenario ---", file=sys.stderr)
    for name, found, expected in per_scenario:
        if expected:
            pct = 100.0 * found / expected
            print(f"  {name:25s} {found:2d}/{expected:2d}  ({pct:5.1f}%)", file=sys.stderr)
        else:
            print(f"  {name:25s} (no expected facts)", file=sys.stderr)

    print("--- per tag ---", file=sys.stderr)
    for tag in sorted(per_tag_expected.keys()):
        e = per_tag_expected[tag]
        f = per_tag_found[tag]
        pct = 100.0 * f / e if e else 0.0
        print(f"  {tag:20s} {f:2d}/{e:2d}  ({pct:5.1f}%)", file=sys.stderr)

    overall = (100.0 * total_found / total_expected) if total_expected else 0.0
    noise_pct = (100.0 * noise_clean / noise_total) if noise_total else 100.0
    print("--- overall ---", file=sys.stderr)
    print(f"  recall                {total_found:2d}/{total_expected:2d}  ({overall:5.1f}%)", file=sys.stderr)
    print(f"  noise resistance      {noise_clean:2d}/{noise_total:2d}  ({noise_pct:5.1f}%)", file=sys.stderr)

    if forbidden_violations:
        print("--- noise leaks ---", file=sys.stderr)
        for v in forbidden_violations:
            print(f"  {v}", file=sys.stderr)

    # Iter 2 floor: at least 60% recall AND no noise leaks (raised by iter 4).
    assert total_expected > 0

    # iter 6 (Codex P1.3): hard regression threshold raised from 75% to 85%.
    #   - iter 1 honest baseline on this fixture was 70.6% (CHANGELOG v2);
    #   - iter 2-5 sustain 90-100% across runs (CHANGELOG v4 / v5);
    #   - 85% closes the regression-permitting gap the previous threshold
    #     allowed: the deterministic-tiebreak fix from v5 P1.1 brought
    #     run-to-run variance under control, and the 90% worst-case
    #     measured run still leaves a 5-point margin before the floor fires.
    # If a future change drops below 85%, investigate the underlying recall
    # issue before lowering the bar.
    score = (total_found / total_expected) if total_expected else 0.0
    assert score >= 0.85, (
        f"Recall quality {score:.2f} below 85% threshold "
        f"({total_found}/{total_expected}); see per-tag breakdown above"
    )
