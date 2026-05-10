"""LLM-driven recall helpers: query rewriting + reranking.

Both are best-effort: if the Anthropic API is unavailable or returns
malformed output, we degrade silently to "no rewrite" / "no rerank".
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic

from .config import get_settings

logger = logging.getLogger(__name__)


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic | None:
    global _client
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# Smaller / cheaper than the extraction model — these calls are on the recall
# hot path and run twice per /recall request, so latency matters more than
# the marginal extraction-quality bump.
RERANK_MODEL_DEFAULT = "claude-haiku-4-5"


# ----- Query rewriting -----

REWRITE_TOOL = {
    "name": "expand_query",
    "description": "Generate paraphrases and topic keywords to improve memory retrieval.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rewrites": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "description": (
                    "Up to 5 alternate phrasings of the query, each focused on a "
                    "different facet (synonym, topic shift, decomposed sub-question). "
                    "These will be embedded and BM25-searched separately, then fused."
                ),
                "items": {"type": "string"},
            }
        },
        "required": ["rewrites"],
    },
}

REWRITE_SYSTEM = """You expand recall queries to improve retrieval over a user-memory store.

Given a query, produce 2-5 alternate phrasings that target the same intent but
use different vocabulary. Tactics:
- Replace abstract terms with concrete topic words ("favorite tools" → "tools they use, software they rely on")
- Decompose multi-hop queries into atomic sub-queries ("does the user with a labrador have allergies" → "food allergies", "dietary restrictions", "what dog do they have")
- Add likely keyword anchors ("where do they live" → "current city", "address", "location")
- Drop noise words; keep concrete nouns

Each rewrite should be a short phrase or question, no more than ~10 words.
Always call the expand_query tool exactly once.
"""


async def rewrite_query(query: str, max_rewrites: int = 4) -> list[str]:
    """Return up to `max_rewrites` alternate phrasings (always includes the
    original at index 0). Empty/short queries pass through unchanged."""
    if not query.strip() or len(query.strip()) < 3:
        return [query]

    client = _get_client()
    if client is None:
        return [query]

    settings = get_settings()
    try:
        resp = await client.messages.create(
            model=RERANK_MODEL_DEFAULT,
            max_tokens=512,
            system=REWRITE_SYSTEM,
            tools=[REWRITE_TOOL],
            tool_choice={"type": "tool", "name": "expand_query"},
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
    except Exception:  # noqa: BLE001
        logger.exception("query rewrite failed; using original")
        return [query]

    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"), None
    )
    if tool_block is None:
        return [query]
    raw = tool_block.input
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [query]
    rewrites = raw.get("rewrites", []) if isinstance(raw, dict) else []
    out = [query]
    for r in rewrites[:max_rewrites]:
        if isinstance(r, str) and r.strip() and r.strip() != query.strip():
            out.append(r.strip())
    return out


# ----- Reranking -----

@dataclass
class RerankItem:
    id: str
    text: str


RERANK_TOOL = {
    "name": "rerank",
    "description": "Score each candidate memory's relevance to the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": (
                    "One entry per candidate, in the SAME ORDER as the candidates "
                    "in the user message. Each score must be 0.0-1.0 where 1.0 = "
                    "directly answers the query, 0.5 = somewhat related, 0.0 = "
                    "irrelevant."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["scores"],
    },
}

RERANK_SYSTEM = """You are a relevance scorer for a memory recall system.

Given a user query and a list of candidate memories, score each candidate's
relevance to the query on a 0.0-1.0 scale.

Scoring guidance:
- 0.9-1.0: directly answers the query
- 0.5-0.8: related, would be useful context
- 0.2-0.4: tangential, mentions adjacent topic
- 0.0-0.1: unrelated noise

Be strict — if the query is "where do they live" and the memory is about
their job, that's 0.1, not 0.5. The point is to filter, not to be charitable.

For multi-hop queries ("does the user with a labrador have allergies"),
score memories that match either hop highly so context assembly can include
both.

Always call the rerank tool exactly once with one score per candidate, in
the same order they were given.
"""


async def rerank(
    query: str, items: list[RerankItem], top_k: int = 10
) -> dict[str, float]:
    """Return {id: relevance_score} for items the LLM judges relevant.

    Returns an empty dict if the LLM is unavailable — caller should fall back
    to the original ranking. The caller is responsible for keeping or
    discarding items based on the returned scores.
    """
    if not items:
        return {}
    client = _get_client()
    if client is None:
        return {}

    listing = "\n".join(f"[{i.id}] {i.text}" for i in items)
    settings = get_settings()
    try:
        resp = await client.messages.create(
            model=RERANK_MODEL_DEFAULT,
            max_tokens=2048,
            system=RERANK_SYSTEM,
            tools=[RERANK_TOOL],
            tool_choice={"type": "tool", "name": "rerank"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nCandidates:\n{listing}\n\n"
                        f"Score each candidate by ID."
                    ),
                }
            ],
        )
    except Exception:  # noqa: BLE001
        logger.exception("rerank failed; falling back to original ranking")
        return {}

    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"), None
    )
    if tool_block is None:
        return {}
    raw = tool_block.input
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    scores = raw.get("scores", []) if isinstance(raw, dict) else []
    out: dict[str, float] = {}
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        try:
            out[str(entry["id"])] = max(0.0, min(1.0, float(entry["score"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ----- Contradiction detection (iter 3) -----

ContradictionAction = Literal["supersede", "consistent", "different_subtopic"]


CONTRADICTION_TOOL = {
    "name": "classify_memory_relationship",
    "description": (
        "Decide whether a newly extracted memory supersedes an existing memory, "
        "is consistent (re-states) it, or is a different sub-topic that just "
        "happens to share a key."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["supersede", "consistent", "different_subtopic"],
                "description": (
                    "supersede: same topic, value has changed — old fact is no "
                    "longer true (e.g. employment changed Stripe → Notion). "
                    "consistent: new memory restates / refines / agrees with old "
                    "one (e.g. 'Lives in Berlin' vs 'Currently in Berlin'). "
                    "different_subtopic: same key was used by accident, the two "
                    "memories are about distinguishable sub-topics that can both "
                    "be true (rare; only when there's clearly no contradiction "
                    "and clearly no supersession)."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the choice.",
            },
        },
        "required": ["action", "reason"],
    },
}

CONTRADICTION_SYSTEM = """You classify the relationship between an existing user memory and a newly extracted one.

Both memories share a topic key (e.g. 'employment', 'location_current'). Decide:

- supersede: the new memory replaces the old one. Same topic, NEW value is the
  current state, OLD value is no longer true. Only choose this when the new
  memory is unambiguously stating present-tense reality.
  Example: old='Works at Stripe' / new='Works at Notion' → supersede.
  Example: old='Lives in NYC' / new='Lives in Berlin' → supersede.

- consistent: the new memory restates, refines, or agrees with the old one.
  Don't create a duplicate row.
  Example: old='Lives in Berlin' / new='Currently in Berlin' → consistent.
  Example: old='Has a dog named Biscuit' / new='Owns Biscuit, a labrador' → consistent (same dog; the labrador detail belongs in a different key like pet_breed).

- different_subtopic: the two memories are about distinguishable sub-topics that
  can both be true at the same time, OR the new memory is describing a PAST
  state rather than asserting a new present state. Both rows stay active.
  Example: old='Has a dog named Biscuit' / new='Has a cat named Whiskers' →
    different_subtopic (multiple pets).
  Example: old='Senior product designer at IDEO' / new='Worked at Stripe before
    IDEO for three years' → different_subtopic (the new memory is historical
    context — past employer, not a contradiction). Critical: when the new
    memory contains temporal markers like 'before', 'previously', 'used to',
    'past', 'former', 'prior to', 'ago', or otherwise frames the value as
    historical, return different_subtopic — that lets the past mention stay
    active without overwriting the current state.

Bias toward supersede when both values are stated in present tense and the
fact has clearly changed. Bias toward consistent when the values describe the
same state with slightly different wording. Bias toward different_subtopic
when the new memory has any temporal marker indicating it's about a past
state. Always call classify_memory_relationship exactly once.
"""


@dataclass
class ContradictionDecision:
    action: ContradictionAction
    reason: str


async def classify_contradiction(
    key: str, old_value: str, new_value: str
) -> ContradictionDecision:
    """Decide what to do when a new memory shares a key with an existing active one.

    Defaults to `different_subtopic` on any failure (LLM unavailable, malformed
    output, exception) — a broken classifier must NEVER destroy data by
    superseding incorrectly. The cost of "different_subtopic" when "supersede"
    was right is two active rows for the same topic; the cost of "supersede"
    when wrong is silently losing a valid memory.
    """
    client = _get_client()
    if client is None:
        return ContradictionDecision(action="different_subtopic", reason="no api client")

    prompt = (
        f"Topic key: {key}\n\n"
        f"Existing memory: {old_value}\n"
        f"New memory:      {new_value}\n\n"
        f"Classify the relationship."
    )
    try:
        resp = await client.messages.create(
            model=RERANK_MODEL_DEFAULT,
            max_tokens=512,
            system=CONTRADICTION_SYSTEM,
            tools=[CONTRADICTION_TOOL],
            tool_choice={"type": "tool", "name": "classify_memory_relationship"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:  # noqa: BLE001
        logger.exception("contradiction-detection call failed; defaulting to different_subtopic")
        return ContradictionDecision(action="different_subtopic", reason="api error")

    tool_block = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"), None
    )
    if tool_block is None:
        return ContradictionDecision(action="different_subtopic", reason="no tool_use block")
    raw = tool_block.input
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ContradictionDecision(action="different_subtopic", reason="malformed json")
    if not isinstance(raw, dict):
        return ContradictionDecision(action="different_subtopic", reason="not a dict")
    action = raw.get("action")
    if action not in ("supersede", "consistent", "different_subtopic"):
        return ContradictionDecision(action="different_subtopic", reason=f"unknown action {action!r}")
    reason = str(raw.get("reason", ""))[:500]
    return ContradictionDecision(action=action, reason=reason)  # type: ignore[arg-type]
