"""Recall pipeline.

Iteration 1: vector cosine top-k baseline over `memories` and `turns`, with
priority-ordered context assembly: stable user facts → query-relevant memories
→ recent turns. We also union with keyword (FTS) matches so cold/keyword-heavy
queries don't return empty before iteration 2 adds proper RRF + reranking.

Token budget enforcement uses tiktoken-style approximation (chars/4) — fast,
good enough for headroom decisions.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text

from .db import session_scope
from .embeddings import embed_batch, embed_text
from .llm_pipeline import RerankItem, rerank, rewrite_query
from .models import Citation

logger = logging.getLogger(__name__)


# Approximate token count: tiktoken would be exact but adds a model load.
# Average English ~4 chars/token; we round up to be safe.
def approx_tokens(s: str) -> int:
    if not s:
        return 0
    return (len(s) + 3) // 4


@dataclass
class _Candidate:
    kind: str  # 'memory' | 'turn'
    id: str
    score: float
    type: str | None
    key: str | None
    value: str
    session_id: str | None
    timestamp: datetime | None
    snippet: str
    confidence: float = 1.0
    active: bool = True
    # For memory candidates this is the turn that produced the memory; for turn
    # candidates it's None (the candidate's own `id` IS the turn id).
    source_turn: str | None = None
    # Iter 5 (review fix P1.4): True if this candidate was matched by FTS
    # keyword search at any point. Flows through RRF (the merge keeps True
    # if any input list had it set) and feeds the simple-query rerank
    # fast-path: a top-1 candidate whose retrieval was driven by FTS, on
    # a short query with a clear margin over rank-2, is the high-confidence
    # case where the LLM rerank reliably agrees with RRF order anyway.
    had_fts_match: bool = False


def _citation_for(c: _Candidate) -> Citation | None:
    """Build a Citation pointing at the underlying turn, or None if unknown.

    Per the HTTP contract, `citations[].turn_id` references a turn — for
    memory-derived candidates we want the memory's `source_turn`, not the
    memory's own row id.
    """
    if c.kind == "turn":
        turn_ref = c.id
    else:
        turn_ref = c.source_turn
    if not turn_ref:
        return None
    return Citation(turn_id=turn_ref, score=c.score, snippet=c.snippet)


def _to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


async def _vector_search_memories(
    user_id: str | None,
    query_emb: list[float],
    limit: int = 20,
    *,
    session: Any = None,
) -> list[_Candidate]:
    """Vector-cosine top-k over the user's active memories.

    `session` is optional. When provided, the query runs in that session
    (allowing recall() to pool all per-facet reads through one connection
    per request — iter 5 / review fix P2.1). When None, opens a one-shot
    `session_scope()` for backward compatibility with /search.
    """
    if user_id is None:
        return []
    sql = text(
        """
        SELECT id, type, key, value, confidence, session_id, source_turn,
               updated_at, active,
               1 - (embedding <=> CAST(:emb AS VECTOR)) AS score
        FROM memories
        WHERE user_id = :uid AND active = TRUE AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:emb AS VECTOR)
        LIMIT :lim
        """
    )
    params = {"uid": user_id, "emb": _to_pgvector(query_emb), "lim": limit}
    if session is not None:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    else:
        async with session_scope() as s:
            result = await s.execute(sql, params)
            rows = result.mappings().all()
    return [
        _Candidate(
            kind="memory",
            id=str(r["id"]),
            score=float(r["score"]),
            type=r["type"],
            key=r["key"],
            value=r["value"],
            session_id=r["session_id"],
            timestamp=r["updated_at"],
            snippet=f"{r['key']}: {r['value']}",
            confidence=float(r["confidence"]),
            active=bool(r["active"]),
            source_turn=str(r["source_turn"]) if r["source_turn"] else None,
        )
        for r in rows
    ]


async def _keyword_search_memories(
    user_id: str | None,
    query: str,
    limit: int = 20,
    *,
    session: Any = None,
) -> list[_Candidate]:
    if user_id is None or not query.strip():
        return []
    sql = text(
        """
        SELECT id, type, key, value, confidence, session_id, source_turn,
               updated_at, active,
               ts_rank(fts_tsv, plainto_tsquery('english', :q)) AS score
        FROM memories
        WHERE user_id = :uid AND active = TRUE
          AND fts_tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :lim
        """
    )
    params = {"uid": user_id, "q": query, "lim": limit}
    if session is not None:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    else:
        async with session_scope() as s:
            result = await s.execute(sql, params)
            rows = result.mappings().all()
    return [
        _Candidate(
            kind="memory",
            id=str(r["id"]),
            score=float(r["score"]) * 0.5,  # de-rank keyword vs cosine; tuned in iter 2
            type=r["type"],
            key=r["key"],
            value=r["value"],
            session_id=r["session_id"],
            timestamp=r["updated_at"],
            snippet=f"{r['key']}: {r['value']}",
            confidence=float(r["confidence"]),
            active=bool(r["active"]),
            source_turn=str(r["source_turn"]) if r["source_turn"] else None,
            had_fts_match=True,
        )
        for r in rows
    ]


async def _vector_search_turns(
    session_id: str,
    query_emb: list[float],
    limit: int = 5,
    *,
    session: Any = None,
) -> list[_Candidate]:
    sql = text(
        """
        SELECT id, session_id, raw_text, timestamp,
               1 - (embedding <=> CAST(:emb AS VECTOR)) AS score
        FROM turns
        WHERE session_id = :sid AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:emb AS VECTOR)
        LIMIT :lim
        """
    )
    params = {"sid": session_id, "emb": _to_pgvector(query_emb), "lim": limit}
    if session is not None:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    else:
        async with session_scope() as s:
            result = await s.execute(sql, params)
            rows = result.mappings().all()
    cands: list[_Candidate] = []
    for r in rows:
        snippet = (r["raw_text"] or "").replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "…"
        cands.append(
            _Candidate(
                kind="turn",
                id=str(r["id"]),
                score=float(r["score"]),
                type=None,
                key=None,
                value=snippet,
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                snippet=snippet,
            )
        )
    return cands


async def _fts_search_turns(
    session_id: str,
    query: str,
    limit: int = 5,
    *,
    session: Any = None,
) -> list[_Candidate]:
    """Postgres FTS over `turns.fts_tsv` scoped to a session.

    iter 6 (Codex P1.2). The recent-turns channel previously ran vector
    search only, so when extraction failed (LLM unavailable, missing API
    key, parse errors) keyword-heavy queries against the raw turn text
    had no FTS path back. This restores that fallback: an exact-keyword
    query still surfaces the turn that introduced the keyword.

    Returns _Candidate rows with `had_fts_match=True` so the rerank
    fast-path correctly counts a turn-side keyword hit as evidence the
    candidate is keyword-driven (mirrors `_keyword_search_memories`).
    """
    if not query.strip():
        return []
    sql = text(
        """
        SELECT id, session_id, raw_text, timestamp,
               ts_rank(fts_tsv, plainto_tsquery('english', :q)) AS score
        FROM turns
        WHERE session_id = :sid
          AND fts_tsv @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :lim
        """
    )
    params = {"sid": session_id, "q": query, "lim": limit}
    if session is not None:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    else:
        async with session_scope() as s:
            result = await s.execute(sql, params)
            rows = result.mappings().all()
    cands: list[_Candidate] = []
    for r in rows:
        snippet = (r["raw_text"] or "").replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "…"
        cands.append(
            _Candidate(
                kind="turn",
                id=str(r["id"]),
                # de-rank FTS vs vector with the same 0.5 factor used for
                # memory FTS — keeps cross-channel score scales comparable
                # before RRF (which is rank-based anyway, but downstream
                # tiebreaks read .score directly).
                score=float(r["score"]) * 0.5,
                type=None,
                key=None,
                value=snippet,
                session_id=r["session_id"],
                timestamp=r["timestamp"],
                snippet=snippet,
                had_fts_match=True,
            )
        )
    return cands


async def _stable_user_facts(
    user_id: str | None,
    limit: int = 20,
    *,
    session: Any = None,
) -> list[_Candidate]:
    """Return active fact + preference memories ordered by confidence then recency.

    `session` is optional. When provided (recall() pools all reads through one
    connection per request — iter 5 / review fix P2.1), the query runs in that
    session. When None, opens a one-shot `session_scope()` for backward
    compatibility with /search.
    """
    if user_id is None:
        return []
    sql = text(
        """
        SELECT id, type, key, value, confidence, session_id, source_turn,
               updated_at, active
        FROM memories
        WHERE user_id = :uid AND active = TRUE
          AND type IN ('fact', 'preference')
        ORDER BY confidence DESC, updated_at DESC
        LIMIT :lim
        """
    )
    params = {"uid": user_id, "lim": limit}
    if session is not None:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
    else:
        async with session_scope() as s:
            result = await s.execute(sql, params)
            rows = result.mappings().all()
    return [
        _Candidate(
            kind="memory",
            id=str(r["id"]),
            score=float(r["confidence"]),
            type=r["type"],
            key=r["key"],
            value=r["value"],
            session_id=r["session_id"],
            timestamp=r["updated_at"],
            snippet=f"{r['key']}: {r['value']}",
            confidence=float(r["confidence"]),
            source_turn=str(r["source_turn"]) if r["source_turn"] else None,
        )
        for r in rows
    ]


def _dedupe(cands: list[_Candidate]) -> list[_Candidate]:
    seen: set[tuple[str, str]] = set()
    out: list[_Candidate] = []
    for c in cands:
        k = (c.kind, c.id)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _reciprocal_rank_fusion(
    *lists: list[_Candidate], k: int = 60
) -> list[_Candidate]:
    """Reciprocal rank fusion (Cormack et al., k=60 is the canonical default).

    score(d) = sum over input lists of 1/(k + rank_in_list(d))

    Rank-based, scale-free — this is the right way to merge cosine cosine
    scores (0..1) with FTS ts_rank (0..0.5) where score magnitudes are not
    comparable. Each input list should already be sorted best-first.
    """
    fused: dict[tuple[str, str], tuple[_Candidate, float]] = {}
    for lst in lists:
        for rank, c in enumerate(lst, start=1):
            key = (c.kind, c.id)
            score = 1.0 / (k + rank)
            existing = fused.get(key)
            if existing is None:
                fused[key] = (c, score)
            else:
                # keep the candidate but accumulate score; pick the higher-ranked
                # source's value/snippet for display (already in `existing`).
                # Iter 5 (P1.4): OR the had_fts_match flag so a candidate that
                # was matched by FTS in any contributing list is marked even if
                # the higher-ranked vector hit happened to be picked for display.
                merged = existing[0]
                if c.had_fts_match and not merged.had_fts_match:
                    merged.had_fts_match = True
                fused[key] = (merged, existing[1] + score)
    out = []
    for cand, score in fused.values():
        # Replace per-list score with fused score so context_assembly can use it.
        cand.score = score
        out.append(cand)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _format_context(
    facts: list[_Candidate],
    relevant_memories: list[_Candidate],
    recent_turns: list[_Candidate],
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Assemble the prompt context within budget, priority-ordered.

    Priority:
      1. Stable user facts (highest)
      2. Query-relevant memories
      3. Recent / relevant turn snippets

    Headers are kept minimal so they don't dominate small budgets.
    """
    sections: list[str] = []
    citations: list[Citation] = []
    used = 0

    def fits(s: str) -> bool:
        return used + approx_tokens(s) <= max_tokens

    if facts:
        header = "## Known facts about this user"
        if fits(header):
            sections.append(header)
            used += approx_tokens(header)
            for c in facts:
                line = f"- {c.value}"
                if fits(line):
                    sections.append(line)
                    used += approx_tokens(line)
                    cit = _citation_for(c)
                    if cit is not None:
                        citations.append(cit)
                else:
                    break

    # Iter 5 (review fix P2.6): split opinions out of `relevant_memories` and
    # emit them in their own section, ordered by recency. Opinions are
    # intentionally kept co-active by `store_turn` (no supersession) so the
    # arc — "previously thought X, now thinks Y" — is preserved. The
    # `[YYYY-MM-DD]` prefix on each bullet gives the agent the temporal
    # signal it needs to read the most recent opinion as "current". Without
    # this split the agent sees both opinions adjacent with no ordering hint.
    opinion_memories = [c for c in relevant_memories if c.type == "opinion"]
    non_opinion_memories = [c for c in relevant_memories if c.type != "opinion"]
    # Most-recent first so the agent reads the current view as the lead bullet.
    opinion_memories.sort(
        key=lambda c: (c.timestamp or datetime.min), reverse=True
    )

    if non_opinion_memories:
        header = "\n## Relevant memories"
        if fits(header):
            sections.append(header)
            used += approx_tokens(header)
            for c in non_opinion_memories:
                stamp = (
                    f"[{c.timestamp.strftime('%Y-%m-%d')}] "
                    if c.timestamp
                    else ""
                )
                line = f"- {stamp}{c.value}"
                if fits(line):
                    sections.append(line)
                    used += approx_tokens(line)
                    cit = _citation_for(c)
                    if cit is not None:
                        citations.append(cit)
                else:
                    break

    if opinion_memories:
        header = "\n## Opinions and views (most recent first)"
        if fits(header):
            sections.append(header)
            used += approx_tokens(header)
            for c in opinion_memories:
                stamp = (
                    f"[{c.timestamp.strftime('%Y-%m-%d')}] "
                    if c.timestamp
                    else ""
                )
                line = f"- {stamp}{c.value}"
                if fits(line):
                    sections.append(line)
                    used += approx_tokens(line)
                    cit = _citation_for(c)
                    if cit is not None:
                        citations.append(cit)
                else:
                    break

    if recent_turns:
        header = "\n## Relevant from recent conversations"
        if fits(header):
            sections.append(header)
            used += approx_tokens(header)
            for c in recent_turns:
                stamp = (
                    f"[{c.timestamp.strftime('%Y-%m-%d')}] "
                    if c.timestamp
                    else ""
                )
                line = f"- {stamp}{c.value}"
                if fits(line):
                    sections.append(line)
                    used += approx_tokens(line)
                    cit = _citation_for(c)
                    if cit is not None:
                        citations.append(cit)
                else:
                    break

    return "\n".join(sections), citations


RERANK_THRESHOLD = 0.25
"""Memories scoring below this in the LLM rerank are dropped before context
assembly. The threshold is the main lever for noise resistance — too low and
unrelated memories leak into the 'Relevant memories' section; too high and
genuine matches get cut. Tune on the fixture."""


def _should_skip_rerank(query: str, fused_candidates: list[_Candidate]) -> bool:
    """Fast-path heuristic: skip the LLM rerank for high-confidence simple
    keyword queries.

    Iter 5 (review fix P1.4). The LLM rerank is the dominant /recall latency
    cost (~3-5s per call). For short keyword-style queries with a clearly
    dominant top-1 candidate that was already matched by FTS, the rerank
    almost always agrees with the RRF order anyway — running it adds latency
    and variance for no recall gain.

    Conditions (ALL must hold):
      1. Query is ≤5 words (keyword-style, not natural language).
      2. Top-1 candidate had an FTS hit (keyword match drove the retrieval).
      3. Top-1 RRF score is > 2x the second candidate's score (clear winner;
         the rerank's role is dispute-resolution, no dispute → no rerank).

    If only one candidate is in the pool, also skip — there's nothing to
    rerank against.
    """
    if len(query.split()) > 5:
        return False
    if not fused_candidates:
        return False
    if len(fused_candidates) == 1:
        return True
    top = fused_candidates[0]
    if not top.had_fts_match:
        return False
    second = fused_candidates[1]
    if second.score <= 0:
        return True
    return top.score > 2.0 * second.score

# Identity-class facts that should appear in priority-1 even when query-irrelevant.
# These are the "agent should always know who they're talking to" basics.
IDENTITY_KEYS = {
    "name",
    "name_first",
    "name_full",
    "location_current",
    "location_city",
    "employment",
    "employment_current",
    "communication_pref",
    "communication_preference",
    "language",
    "preferred_language",
}


async def recall(
    query: str,
    session_id: str,
    user_id: str | None,
    max_tokens: int,
) -> tuple[str, list[Citation]]:
    """Primary recall — returns formatted context + citations within token budget.

    Pipeline:
      1. Rewrite the query into 1-5 facets via LLM (cheap, parallel-safe).
      2. Embed each facet; vector + FTS search per facet against memories.
      3. Vector + FTS search of recent turns scoped to session.
      4. Reciprocal rank fusion across all per-facet result lists.
      5. LLM rerank of the top 20 fused candidates against the original query.
      6. Drop candidates below RERANK_THRESHOLD (noise filter).
      7. Assemble context: stable facts → reranked relevant → recent turns,
         within token budget.
    """
    # Iter 5 (review fix P3.3): pull request_id from the FastAPI middleware
    # contextvar so this log line correlates with the matching store_turn log
    # if the same request triggers both paths. Lazy import to avoid the
    # main.py → recall.py → main.py circular dependency at module load.
    from .main import get_request_id

    request_id = get_request_id()

    if max_tokens <= 0:
        logger.info(
            "recall short-circuit: max_tokens=%d", max_tokens,
            extra={"request_id": request_id},
        )
        return "", []

    query_clean = query.strip()

    # Iter 5 (review fix P2.1): single session per recall. The previous
    # implementation opened 3+ separate `session_scope()` blocks (one per
    # helper × 5 facets × 2 modes ≈ 10+ connection acquisitions per request).
    # Now we open ONE session here and pass it through every helper. Helpers
    # accept `session: Any = None` and fall back to opening their own scope
    # when called outside of recall() (e.g. from /search), so the public API
    # is backward-compatible.
    async with session_scope() as session:
        if not query_clean:
            all_facts = await _stable_user_facts(user_id, limit=30, session=session)
            return _format_context(all_facts[:10], [], [], max_tokens)

        # Run the stable-facts read and the LLM query rewrite concurrently —
        # neither blocks the other. The DB read uses the shared session; the
        # rewrite is an LLM call that doesn't touch the DB.
        all_facts, rewrites = await asyncio.gather(
            _stable_user_facts(user_id, limit=30, session=session),
            rewrite_query(query_clean),
        )

        # Identity facts always go through; non-identity facts compete for slots
        # in priority-1 only if they pass the rerank threshold.
        identity_facts = [c for c in all_facts if (c.key or "") in IDENTITY_KEYS]
        other_facts = [c for c in all_facts if (c.key or "") not in IDENTITY_KEYS]

        rewrite_embs = await embed_batch(rewrites)

        # iter 6 (Codex P1.1): per-facet searches run sequentially over the
        # shared session. SQLAlchemy's AsyncSession is explicitly NOT safe for
        # concurrent use — running multiple `session.execute()` coroutines
        # under a single `asyncio.gather` is a documented foot-gun and can
        # produce corrupted results or InterfaceErrors under real parallelism
        # even though it appeared to work on the eval fixture.
        #
        # Sequential execution costs negligible latency: each query is a fast
        # indexed lookup (< 5ms typical) and the LLM rerank that follows is
        # multiple seconds. The 5 facets × 2 modes = ~10 queries × 5ms ≈ 50ms
        # is dwarfed by every other component of the pipeline.
        mem_lists: list[list[_Candidate]] = []
        if user_id is not None:
            for q, emb in zip(rewrites, rewrite_embs):
                mem_lists.append(
                    await _vector_search_memories(user_id, emb, limit=20, session=session)
                )
                mem_lists.append(
                    await _keyword_search_memories(user_id, q, limit=20, session=session)
                )

        fused = _reciprocal_rank_fusion(*mem_lists) if mem_lists else []

        # Don't repeat identity facts as relevant memories. Non-identity facts
        # remain candidates because the rerank decides whether they're query-relevant.
        identity_ids = {(c.kind, c.id) for c in identity_facts}
        fused = [c for c in fused if (c.kind, c.id) not in identity_ids]

        # Iter 5 (review fix P1.1): record RRF rank for each fused candidate
        # BEFORE we mutate scores via rerank. This is the deterministic tiebreak
        # source — when the LLM reranker gives two candidates the same score
        # (or scores within floating-point noise), we want them to fall out in
        # the original RRF order rather than depending on the LLM's
        # non-deterministic choice. CHANGELOG v4 documented 90-100% run-to-run
        # variance attributable to exactly this. The 1e-6 scaling keeps the
        # tiebreak strictly subordinate to rerank score.
        fused_rank: dict[tuple[str, str], int] = {
            (c.kind, c.id): i for i, c in enumerate(fused)
        }

        # LLM rerank — top 20 candidates by fused RRF score, plus all non-identity
        # facts (so the rerank decides which "always-on" facts are query-relevant
        # and which would just be noise).
        other_fact_ids = {(c.kind, c.id) for c in other_facts}
        pool_seen: set[tuple[str, str]] = set()
        rerank_pool: list[_Candidate] = []
        for c in fused[:20]:
            key = (c.kind, c.id)
            if key in pool_seen:
                continue
            pool_seen.add(key)
            rerank_pool.append(c)
        for c in other_facts:
            key = (c.kind, c.id)
            if key in pool_seen:
                continue
            pool_seen.add(key)
            rerank_pool.append(c)

        # Iter 5 (review fix P1.2): bound the rerank pool. Without this cap a
        # power user with 100+ memories triggers a 100-item rerank prompt every
        # /recall — quadratic-ish token cost growth. 30 is the cap because:
        # - top-20 fused RRF candidates are already the most relevant by retrieval;
        # - top-10 additional non-identity facts cover the "always-on" stable
        #   facts that compete for priority-2 slots;
        # - facts beyond rank 30 are virtually never query-relevant in practice
        #   (the rerank threshold is 0.25, RRF rank correlates with relevance
        #   such that rank-30+ rarely scores above the threshold anyway).
        rerank_pool = rerank_pool[:30]

        # Iter 5 (review fix P1.4): simple-query fast-path. For short keyword
        # queries with a clear top-1 winner driven by FTS, skip the LLM rerank.
        # This drops one ~3-5s LLM hop on ~30% of probes (per CHANGELOG v4
        # latency analysis) and keeps the noise filter intact: only candidates
        # already in the fused RRF list survive (non-identity facts dragged in
        # for rerank consideration are dropped, just like in the rerank-failure
        # fallback path).
        if _should_skip_rerank(query_clean, fused):
            logger.info(
                "rerank skipped: simple query (top1.score=%.4f, top2.score=%.4f)",
                fused[0].score if fused else 0.0,
                fused[1].score if len(fused) > 1 else 0.0,
            )
            # Keep only fused candidates (drop non-identity facts that were never
            # seen in the per-facet retrieval); preserve RRF order.
            fused_ids = {(c.kind, c.id) for c in fused}
            rerank_pool = [c for c in rerank_pool if (c.kind, c.id) in fused_ids]
            rerank_pool.sort(key=lambda c: c.score, reverse=True)
        elif rerank_pool:
            items = [
                RerankItem(id=c.id, text=c.snippet or c.value) for c in rerank_pool
            ]
            scores = await rerank(query_clean, items)
            if scores:
                # Apply rerank scores plus a tiny RRF-rank tiebreak. Candidates
                # not in `fused_rank` (e.g. non-identity facts pulled in directly)
                # are pushed to the end of any tie group via len(fused) fallback.
                tiebreak_floor = len(fused) + len(rerank_pool)
                for c in rerank_pool:
                    rrf_position = fused_rank.get((c.kind, c.id), tiebreak_floor)
                    c.score = scores.get(c.id, 0.0) + 1e-6 * (1.0 / (rrf_position + 1))
                rerank_pool = [c for c in rerank_pool if c.score >= RERANK_THRESHOLD]
                rerank_pool.sort(key=lambda c: c.score, reverse=True)
            else:
                # Rerank unavailable — fall back to RRF order, keep only fused matches
                # (drop the non-identity facts to avoid leaking them on noise queries).
                rerank_pool = [c for c in rerank_pool if (c.kind, c.id) not in other_fact_ids]
        relevant_memories = rerank_pool[:10]

        # iter 6 (Codex P1.2): recent turns now have BOTH a vector channel and
        # an FTS channel, fused with RRF. Previously only vector ran here, so
        # if extraction failed and no memories were created for a turn, a
        # keyword-heavy query against that turn's content had no fallback.
        # FTS over `turns.fts_tsv` recovers exact keyword matches.
        recent_turns: list[_Candidate] = []
        primary_emb = rewrite_embs[0] if rewrite_embs else None
        if primary_emb is not None:
            vec_turns = await _vector_search_turns(
                session_id, primary_emb, limit=5, session=session
            )
            fts_turns = await _fts_search_turns(
                session_id, query_clean, limit=5, session=session
            )
            if fts_turns:
                # Merge via RRF when we have both channels; otherwise use
                # whichever returned rows (fall back to vector-only behaviour
                # if FTS is empty).
                recent_turns = _reciprocal_rank_fusion(vec_turns, fts_turns)[:5]
            else:
                recent_turns = vec_turns

    logger.info(
        "recall completed: query=%r facets=%d candidates=%d kept=%d",
        query_clean[:80],
        len(rewrites),
        len(fused),
        len(relevant_memories),
        extra={"request_id": request_id},
    )
    return _format_context(
        identity_facts[:10], relevant_memories, _dedupe(recent_turns)[:5], max_tokens
    )


async def search(
    query: str,
    session_id: str | None,
    user_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Structured search results (for /search endpoint)."""
    if not query.strip():
        return []

    query_emb = await embed_text(query)

    vec_mem: list[_Candidate] = []
    kw_mem: list[_Candidate] = []
    turn_mem: list[_Candidate] = []
    if user_id is not None:
        vec_mem = await _vector_search_memories(user_id, query_emb, limit=limit * 2)
        kw_mem = await _keyword_search_memories(user_id, query, limit=limit * 2)
    if session_id is not None:
        turn_mem = await _vector_search_turns(session_id, query_emb, limit=limit * 2)

    merged = _reciprocal_rank_fusion(vec_mem, kw_mem, turn_mem)[:limit]
    return [
        {
            "content": c.value,
            "score": c.score,
            "session_id": c.session_id or "",
            "timestamp": c.timestamp,
            "metadata": {
                "kind": c.kind,
                "type": c.type,
                "key": c.key,
                "id": c.id,
            },
        }
        for c in merged
    ]
