# memory-service

A Dockerized memory service for an AI agent. Ingests conversation turns,
extracts structured knowledge with an LLM, and answers recall queries that
decide what context the agent sees on the next turn. The HTTP contract is
seven endpoints — see the architecture diagram in §1 and `src/main.py` for
implementations.

This document is the design rationale. For iteration history with measured
deltas, see [`CHANGELOG.md`](./CHANGELOG.md).

---

## Quick start

```sh
git clone <this repo> memory-service
cd memory-service
cp .env.example .env             # set ANTHROPIC_API_KEY (required) and OPENAI_API_KEY (recommended)
docker compose up -d --build
until curl -sf http://localhost:8080/health; do sleep 1; done
```

The service listens on `:8080`. Persistence lives in the `memory_pgdata`
named volume — `docker compose down && up` is invisible to clients.

---

## 1. Architecture

```
                  ┌────────────────────────────┐
                  │  HTTP client (eval harness │
                  │  / agent / curl)           │
                  └──────────────┬─────────────┘
                                 │  JSON
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  api  (FastAPI, src/main.py)                    │
        │   • lifespan → init_schema (idempotent DDL)     │
        │   • optional Bearer auth (MEMORY_AUTH_TOKEN)    │
        │   • 7 endpoints: /health, /turns, /recall,      │
        │     /search, /users/{id}/memories,              │
        │     DELETE /sessions/{id}, DELETE /users/{id}   │
        └─────┬─────────────────────────────┬─────────────┘
              │                             │
              │  POST /turns                │  POST /recall, /search
              │  (sync extract + store)     │
              ▼                             ▼
   ┌────────────────────────┐   ┌──────────────────────────────┐
   │ src/extraction.py      │   │ src/recall.py                │
   │  • Anthropic Claude    │   │  • src/llm_pipeline.py       │
   │    (tool-forced JSON)  │   │     query_rewrite + rerank   │
   │  • record_memories     │   │  • per-facet vector + FTS    │
   │    schema = source     │   │  • RRF fusion (k=60)         │
   │    of truth            │   │  • LLM rerank, threshold 0.25│
   │                        │   │  • priority-ordered context  │
   └─────┬──────────────────┘   └─────────────┬────────────────┘
         │                                    │
         │  src/storage.py                    │
         │  (single transaction:              │
         │   turn + memories)                 │
         ▼                                    ▼
   ┌────────────────────────────────────────────────────┐
   │  db   (postgres:16 + pgvector + pgcrypto)          │
   │   • turns(... raw_text, embedding, fts_tsv ...)    │
   │   • memories(... type/key/value, supersedes,       │
   │     active, embedding, fts_tsv ...)                │
   │   • HNSW index on embeddings                       │
   │   • GIN index on tsvectors                         │
   │   • named volume memory_pgdata                     │
   └────────────────────────────────────────────────────┘

         External API clients (off-path):
           • Anthropic  — extraction (Sonnet 4) + rewrite/rerank (Haiku 4.5)
           • OpenAI     — text-embedding-3-small (1536-dim)
                          (deterministic hash fallback if absent)
```

Two containers, one volume. The api process is a single FastAPI monolith;
the db process is stock pgvector/pgvector:pg16. Inside the api process,
`extraction.py` and `recall.py` are the two main pipelines, with
`llm_pipeline.py` shared between recall (rewrite + rerank) and reused by
extraction (record_memories tool). All state lives in Postgres; the api
container is stateless and can be rebuilt without data loss.

---

## 2. Backing store choice — Postgres 16 + pgvector

**Defended against the alternatives:**

| Store                        | Why we considered it           | Why we rejected it                                                                                                |
|------------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Qdrant / Weaviate + Postgres | Best-in-class HNSW             | Two engines means two transactions. The challenge brief's synchronous-correctness requirement needs turn write + memory write atomic. |
| Redis (RediSearch + vector)  | Fast, low ops                  | No durable transactions across keys. Lossy on restart unless you tune AOF aggressively.                            |
| SQLite + FTS5 + sqlite-vss   | Zero ops, great for embedding  | No HNSW; vector queries are linear. FTS5 BM25 is fine but losing concurrent writers hurts when extraction is slow. |
| Mongo                        | Flexible schema for memories   | Vector support is bolt-on; transactional consistency across `turns` + `memories` collections is awkward.            |
| Flat files + clever index    | Tempting for a small service   | Loses structured filters (`active=TRUE`, `supersedes IS NULL`, `user_id=…`); rebuilding indices on restart hurts.   |

**Why Postgres + pgvector wins this task specifically:**

1. **Synchronous correctness is non-negotiable.** The challenge brief
   requires that after `POST /turns` returns, the ingested data and
   extracted memories must be immediately available via `/recall`. A
   two-store design needs distributed-write coordination. With one
   Postgres engine, `src/storage.py::store_turn` inserts the turn and N
   memory rows in a single transaction; commit happens before the HTTP
   response.
2. **HNSW vector + GIN FTS coexist in one query plan.** A single
   `SELECT … WHERE user_id=:uid AND active=TRUE ORDER BY embedding <=> :q`
   composes the structured filter, the vector index, and (for keyword) the
   tsvector index. No fan-out, no client-side joins.
3. **Restart-safety is one named volume.** `docker compose down && up`
   preserves everything because the data lives in `memory_pgdata`. This is
   verified end-to-end in iteration 1 (CHANGELOG v1).
4. **Supersession (iter 3) wants self-referential FK.** `memories.supersedes
   REFERENCES memories(id) ON DELETE SET NULL` is trivial in Postgres,
   awkward in a vector-store-only design.
5. **Operational footprint is small.** One `pgvector/pgvector:pg16` image,
   one volume, no schema-migration story (single version).

The cost is that vector recall is HNSW on Postgres rather than a dedicated
vector DB — slightly slower at scale, but at memory-service-per-user
scale (~25 memories/user in our fixture) it's negligible.

---

## 3. Extraction pipeline

`src/extraction.py`. One Claude call per turn, run synchronously inside
`POST /turns` (before commit). The call uses the **tool-forced JSON**
pattern:

```python
client.messages.create(
    model=settings.extraction_model,           # claude-sonnet-4-* by default
    tools=[EXTRACTION_TOOL],                   # record_memories schema
    tool_choice={"type": "tool", "name": "record_memories"},
    ...
)
```

**Why tool-forced JSON, not prefill or `output_format`:**
- Assistant-prefill returns a 400 on the Claude 4 family.
- The legacy `output_format` parameter is deprecated.
- A tool with a strict JSON-Schema is the source of truth for what a
  memory looks like. The schema lives next to the prompt; reviewers can
  read it at the top of `src/extraction.py`.

**Type taxonomy:**

| `type`       | What it is                                | Supersedes? |
|--------------|-------------------------------------------|-------------|
| `fact`       | Stable user attribute (name, location, employment, family) | yes |
| `preference` | Durable like/dislike (dietary, communication style, tools)  | yes |
| `opinion`    | Subjective view, may evolve gradually (see §5)              | no — both stay active |
| `event`      | Time-bound activity or change (joined-job, trip-planning)   | no — events compose |

**The load-bearing role of `key`.** The prompt forces a stable `snake_case`
slug *for the topic*, not the value. `employment` is the key for both
"Works at Stripe" and "Just joined Notion." This is what lets iter 3's
contradiction detection do its job — `(user_id, key='employment')` is the
join column for "is this a contradiction or a new fact?" If extraction
emits `employment_stripe` and `employment_notion` we lose. The prompt is
explicit about the same-topic-same-key requirement and gives examples.

**Implicit-fact handling.** The prompt walks through cases:
- "Walking Biscuit this morning" → `pet_name=Biscuit`, `pet=has dog`.
- "Mei and I are flying to Lisbon" → `family_partner=Mei`,
  `event_trip_lisbon`.
- "actually, I meant X" → record corrected fact, set `confidence`
  appropriately so a later supersession step can prefer the correction.

**Confidence floor.** Anything below 0.5 is dropped at the parser.
Self-confessed uncertainty ("I think she said her name was Mei") doesn't
become a high-confidence row.

**What it doesn't catch.** Long multi-turn arcs the model can't see in a
single call. If a user reveals their job over 3 turns ("I'm in payments
infra" / "stripe is a wild place" / "we just shipped X") the model sees
each turn in isolation and may not connect them. Iter 4 candidate: a
small "stitch" pass that re-runs extraction on a session digest. Out of
scope for the current implementation.

---

## 4. Recall strategy

`src/recall.py::recall`. Six stages.

### 4.1 Query rewriting

`src/llm_pipeline.py::rewrite_query`. Haiku 4.5 expands the input query
into 2–5 alternate phrasings via tool-forced JSON (`expand_query` tool).
Tactics in the prompt:

- Replace abstract terms with concrete topic words.
- Decompose multi-hop queries into atomic sub-queries.
- Add likely keyword anchors (synonyms, related entity types).
- Drop noise words.

Example: `"Does the user with a labrador have any food allergies?"` →
`["food allergies", "shellfish allergy", "what dog do they have", ...]`.

This is the largest single lift in the recall pipeline — it's what carried
multi-hop probes from 0% → 100% in CHANGELOG v2.

### 4.2 Per-facet vector + FTS

For each rewrite (and the original query), we run two searches against the
`memories` table:

- **Vector**: cosine via `vector_cosine_ops` HNSW index, `WHERE user_id=:uid AND active=TRUE`.
- **FTS**: `fts_tsv @@ plainto_tsquery('english', :q)`, ranked by `ts_rank`.

That's `1 + N_rewrites` queries × 2 = up to ~10 lists, each top-20.

### 4.3 Reciprocal rank fusion

`src/recall.py::_reciprocal_rank_fusion`, k=60 (Cormack et al.). For each
input list, position `r` contributes `1/(k+r)` to that candidate's fused
score. Rank-based and scale-free — fixes the broken `max(score)` merge
from iter 1, which mixed cosine (0..1) and ts_rank (0..0.5) on the same
axis.

### 4.4 LLM rerank

`src/llm_pipeline.py::rerank`. Top 20 fused candidates, plus all
non-identity stable facts (so the rerank can decide which "always-on"
facts are query-relevant), go to Haiku 4.5 with a single tool-forced
`rerank` call returning per-id 0–1 relevance scores.

`RERANK_THRESHOLD = 0.25` in `src/recall.py`. Anything below is dropped.
This is the noise filter: queries about topics the user never discussed
("favorite color?") get zero high-relevance candidates, so the "Relevant
memories" section is empty rather than leaking distractor memories that
happen to share keywords.

### 4.5 Priority-ordered context assembly

`src/recall.py::_format_context` walks three tiers, taking what fits in
the `max_tokens` budget:

1. **Identity facts** — facts whose `key` is in `IDENTITY_KEYS`
   (`name`, `location_current`, `employment`, `communication_pref`,
   `preferred_language`, etc. in `src/recall.py`). Auto-included
   regardless of query relevance.
2. **Reranked relevant memories** — only candidates that passed
   `RERANK_THRESHOLD`, sorted by rerank score.
3. **Recent turn snippets** — vector-similar matches against the `turns`
   table, scoped to the current `session_id`.

**Why this order.** The agent always needs to know who it's talking to
(name, location, role, communication style) — that's the stable context
no query can do without. Query-relevant memories are the variable part.
Recent turn snippets are grounding for cross-references ("…the project
they mentioned earlier"). Under tight budgets, the agent loses
fine-grained recall before it loses identity context, which is the right
failure mode.

### 4.6 Token budget

Approximate via `len(text) // 4` (`src/recall.py::approx_tokens`). Off by
~10% in either direction; cheaper than loading tiktoken on every recall.
Headers and bullet bodies are added one at a time and stop as soon as the
next addition would overshoot. tiktoken-accurate counting is on the iter
4 list.

`max_tokens=0` returns an empty context with status 200 — never errors.

---

## 5. Fact evolution

Shipped in iteration 3 and tightened in iteration 4 (temporal-marker prompt)
and iteration 5 (concurrent-write race closed via partial unique index).
See [`CHANGELOG.md`](./CHANGELOG.md) v3 / v4 / v5 for the iteration history
and measured deltas; the design is summarised below.

**Supersession on extract.** When `extraction.py` produces a new
`fact`/`preference` whose `(user_id, key)` matches an existing active
memory, a small Haiku 4.5 call classifies the relationship:

- `supersede`: new fact replaces old (employment changed, location moved).
  → old row gets `active=FALSE`, new row's `supersedes` points to old's
  `id`. Both rows are preserved.
- `consistent`: same fact restated; no write needed (idempotency).
- `different_subtopic`: same `key` but truly distinct (e.g. `employment`
  meaning current vs. side gig). New row inserted, old stays active.

Recall queries always include `WHERE active = TRUE`, so only the current
fact surfaces in `/recall`. The `GET /users/{user_id}/memories` endpoint
returns the full chain (active and superseded), which is what the
reviewer needs to verify history is preserved.

The Stripe → Notion case becomes:

```
employment  Works at Stripe  active=FALSE   supersedes=NULL
employment  Works at Notion  active=TRUE    supersedes=<stripe-id>
```

`/recall` returns "Works at Notion." `GET /users/{u}/memories` shows both.

**Opinion arc.** Opinions don't auto-supersede. Two reasons:

1. They legitimately evolve gradually rather than flip
   ("I love TypeScript" → "TypeScript generics are getting annoying" →
   "TypeScript is fine for big projects but I'd use Python for scripts").
   None of those is wrong; the arc is the truth.
2. The agent often wants to see the trajectory ("you used to love
   TypeScript — what changed?").

Current behavior: opinions stay active, recall presents them in temporal
order. The harder variant — full arc reconstruction with explicit
trajectory annotations — is a known design limitation and a future
iteration. The current implementation is partial and documented as such.

---

## 6. Tradeoffs

**Optimized for:**

- **Recall quality.** Per-facet retrieval + RRF + LLM rerank — three
  layers of signal aggregation, each fixing a specific failure mode of
  vanilla cosine-top-k. Per-tag breakdown in CHANGELOG v2 shows the lift
  isn't uniform — multi-hop went 0%→100%, that's the load-bearing one.
- **Synchronous correctness.** Extraction runs inside the `POST /turns`
  transaction. After 201, `/recall` and `/users/.../memories` are
  guaranteed to see the new memories — no eventual consistency, no race
  windows.
- **Simplicity.** One storage tier, one DDL file, one container besides
  the API. Faster to reason about and debug than a polyglot persistence
  story.
- **History preservation.** Supersession marks rows inactive, never
  deletes. `GET /users/{u}/memories` is the audit trail.

**Gave up:**

- **Latency.** ~13s/probe in iter 2 (CHANGELOG v2 timings) — two LLM
  calls per `/recall` (rewrite + rerank) plus ~10 sequential per-facet DB
  hits. iter 4 work: parallelise per-facet queries with `asyncio.gather`,
  skip rerank when fused candidates ≤ 3.
- **Horizontal scalability.** Single Postgres. No read replicas, no
  sharding. The spec explicitly takes this off the table; if it became
  relevant, splitting on `user_id` is the obvious path.
- **Perfect token accounting.** `len(s) // 4` is ±10%. Iter 4 swap to
  tiktoken when budget pressure becomes a constraint.
- **Provider lock-in.** Anthropic for extraction/rewrite/rerank, OpenAI
  for embeddings. Both are configurable via env, both have documented
  fallback paths (see §7).

---

## 7. Failure modes

| Condition                    | Behavior                                                                                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Missing `ANTHROPIC_API_KEY`  | Extraction disabled; turns still stored and FTS-searchable. Query rewriting + LLM rerank also degrade silently to single-facet RRF (`rewrite_query` returns `[query]`, `rerank` returns `{}`). Logged once on first call. |
| Missing `OPENAI_API_KEY`     | `src/embeddings.py` falls back to a deterministic SHA-256 hash projected onto the unit sphere — recall quality on cosine drops sharply, FTS still works. Logged once. Documented in `.env.example`. |
| Cold session (no memories)   | `/recall` returns `200 {"context": "", "citations": []}`. Never errors.                                                         |
| Malformed JSON / missing fields | `422 Unprocessable Entity` (FastAPI default). Service stays up.                                                              |
| Unicode / oversized payload  | Stored as JSONB; `messages` column accepts arbitrary nested content. Tested end-to-end (contract test `test_malformed_input_returns_4xx` includes a unicode round-trip). |
| Postgres restart             | State survives via the `memory_pgdata` named volume. Verified by the persistence path in iter 1; smoke test re-runs after `docker compose down && up`. |
| Slow disk                    | Higher latency, no correctness impact. Recall is bounded by the LLM calls, not Postgres I/O.                                    |
| Anthropic rate-limit / 5xx   | Both `extract_memories` and `llm_pipeline.{rewrite_query,rerank}` catch broad exceptions, log them, and return their "no-op" value. The request still completes. |
| `max_tokens=0` on `/recall`  | Returns empty context (200, not 4xx). Spec doesn't require an error here; this is the more useful behavior.                     |

---

## 8. How to run the tests

### Boot

```sh
docker compose up -d --build
until curl -sf http://localhost:8080/health; do sleep 1; done
```

### Smoke test

```sh
curl -X POST http://localhost:8080/turns \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month. Loving it so far."},
      {"role": "assistant", "content": "That sounds exciting! Berlin is a great city. How are you settling in?"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }'

curl -X POST http://localhost:8080/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Where does this user live?",
    "session_id": "smoke-2",
    "user_id": "user-1",
    "max_tokens": 512
  }'

curl http://localhost:8080/users/user-1/memories | jq .
```

### Contract tests

```sh
MEMORY_SERVICE_URL=http://localhost:8080 pytest tests/test_contract.py -v
```

Covers: `/health`, turn → recall roundtrip, cold session returns empty
(200, not 4xx), concurrent sessions don't bleed, malformed input → 4xx
(not crash).

### Recall self-eval

```sh
MEMORY_SERVICE_URL=http://localhost:8080 pytest tests/test_recall_quality.py -s
```

Ingests `fixtures/recall_eval.json` (4 scenarios, 24 probes), runs each
probe against `/recall`, and reports a per-tag breakdown:

```
=== recall self-eval ===
  keyword            ...
  semantic_only      ...
  semantic_lite      ...
  multi_hop          ...
  noise              ...
  ---
  overall recall     ...
  noise resistance   ...
```

This is the iteration-loop signal — it's what we re-run after every
ranking change. CHANGELOG v2 records the latest numbers.

### Required env

Document in `.env.example`. Provide via `.env` or `docker compose --env-file`:

| Variable             | Required? | Purpose                                                    |
|----------------------|-----------|------------------------------------------------------------|
| `ANTHROPIC_API_KEY`  | Recommended | Extraction + query rewriting + LLM rerank. If absent, all three degrade silently. |
| `OPENAI_API_KEY`     | Recommended | text-embedding-3-small. If absent, deterministic hash fallback. |
| `MEMORY_AUTH_TOKEN`  | Optional    | If set, all endpoints require `Authorization: Bearer <token>`. |
| `EXTRACTION_MODEL`   | Optional    | Override the extraction model (default `claude-sonnet-4-*`). |
| `EMBEDDING_MODEL`    | Optional    | Override embeddings (default `text-embedding-3-small`).      |

---

## File map

| Path                       | Role                                                          |
|----------------------------|---------------------------------------------------------------|
| `src/main.py`              | FastAPI app, all 7 endpoints, lifespan does `init_schema`.    |
| `src/config.py`            | pydantic-settings; single cached `Settings`.                   |
| `src/db.py`                | Engine, session factory, idempotent DDL.                       |
| `src/models.py`            | Pydantic request/response shapes; matches the HTTP contract exactly. |
| `src/embeddings.py`        | OpenAI client + deterministic hash fallback.                   |
| `src/extraction.py`        | Tool-forced extraction; `record_memories` schema is here.      |
| `src/storage.py`           | Transactional turn + memories writes; list/delete.             |
| `src/recall.py`            | RRF, priority assembly, `IDENTITY_KEYS`, `RERANK_THRESHOLD`.   |
| `src/llm_pipeline.py`      | `rewrite_query` and `rerank` (Haiku 4.5).                      |
| `tests/test_contract.py`   | HTTP contract tests.                                           |
| `tests/test_recall_quality.py` | Self-eval driver; per-tag breakdown.                       |
| `fixtures/recall_eval.json`| 4 scenarios, 24 probes; the iteration-loop signal.             |
| `docker-compose.yml`       | api + db + `memory_pgdata` volume.                             |
| `Dockerfile`               | Multi-stage Python 3.12-slim build (deps installed in builder; runtime keeps src + curl), runs uvicorn. |
| `CHANGELOG.md`             | Iteration history with measured deltas.                        |

---

## What's not here

- **Agent-side code.** Out of scope per the challenge brief.
- **UI.** Same.
- **Multi-tenant production hardening.** Same. One auth token, one user
  scope per `user_id`, no per-tenant isolation beyond row-level filters.
- **Migrations.** Single schema version. `init_schema` is idempotent
  (`CREATE TABLE IF NOT EXISTS`); no Alembic.
- **Horizontal scalability.** Single Postgres, single api. Splittable on
  `user_id` if it ever became relevant.

---

## Originality

This is original. The recall pipeline (per-facet rewriting → RRF →
LLM rerank → identity-gated priority assembly) was designed against the
specific failure modes captured in `CHANGELOG.md` — not lifted from
mem0, hindsight, honcho, or other published memory-system designs.
Parameters worth defending in interview: `RERANK_THRESHOLD = 0.25`,
`IDENTITY_KEYS` membership, RRF `k=60`, the choice to keep opinions
co-active rather than supersede, the chars/4 token approximation.
