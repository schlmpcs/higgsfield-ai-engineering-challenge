"""LLM-driven extraction of structured memories from raw conversation turns.

Strategy: a single Claude call per turn. The prompt forces JSON output via
`tool_choice` (avoiding the deprecated `output_format` and assistant-prefill
patterns). Each extracted memory carries a type, a stable `key` (for
contradiction detection in later iterations), a `value`, and a confidence.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic

from .config import get_settings
from .models import Message

logger = logging.getLogger(__name__)


MemoryType = Literal["fact", "preference", "opinion", "event"]


@dataclass
class ExtractedMemory:
    type: MemoryType
    key: str
    value: str
    confidence: float
    metadata: dict[str, Any]


_client: AsyncAnthropic | None = None
_warned_no_key = False


def _get_client() -> AsyncAnthropic | None:
    global _client
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# Tool schema forces structured JSON output. Claude's tool runner is overkill
# here — we only need one call, no loop.
EXTRACTION_TOOL = {
    "name": "record_memories",
    "description": (
        "Record structured memories extracted from the conversation turn. "
        "Call this exactly once with all extracted memories."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "description": "All memories extracted from this turn.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["fact", "preference", "opinion", "event"],
                            "description": (
                                "fact: stable user attribute (job, location, family, pet). "
                                "preference: durable like/dislike. "
                                "opinion: subjective view (may evolve). "
                                "event: time-bound activity or state change."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Short stable topic slug — REQUIRED for contradiction detection. "
                                "Use snake_case slugs like 'employment', 'location', 'pet_name', "
                                "'dietary_preference', 'opinion_typescript'. The same topic mentioned "
                                "across sessions MUST get the same key."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "The concrete fact in natural language — what is true now. "
                                "E.g. 'Works at Notion as a PM' or 'Allergic to shellfish'."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "0.95+ for explicit statements, 0.7-0.9 for strong implicit, "
                                "0.5-0.7 for uncertain inferences. Below 0.5: don't extract."
                            ),
                        },
                        "cardinality": {
                            "type": "string",
                            "enum": ["singleton", "multiple"],
                            "description": (
                                "iter 6 (Codex P2.1). 'singleton' = at most one value can be "
                                "true at a time for this user — supersession applies "
                                "(current_job, current_city, current_relationship_status). "
                                "'multiple' = many values coexist — no supersession "
                                "(pets, skills, opinions, allergies, projects, hobbies). "
                                "When in doubt prefer 'multiple' — the cost of an extra row "
                                "is far smaller than the cost of silently dropping data."
                            ),
                        },
                        "subject": {
                            "type": "string",
                            "description": (
                                "iter 6 (Codex P2.1). The entity this memory is about. "
                                "'user' for memories about the user themselves; otherwise the "
                                "named entity, e.g. 'Biscuit' for the user's dog, 'Stripe' for "
                                "the user's employer. Defaults to 'user' if unsure."
                            ),
                        },
                    },
                    "required": ["type", "key", "value", "confidence"],
                },
            }
        },
        "required": ["memories"],
    },
}


# Iter 5 (review fix P1.3): controlled-vocabulary key aliases.
#
# The extraction prompt asks the LLM for stable snake_case slugs, but in
# practice it drifts: `employment` one turn, `employment_current` the next;
# `location` here, `location_city` there. Supersession joins on exact `key`
# equality (`src/storage.py:_resolve_supersession`), so vocabulary drift
# silently disables contradiction detection — two "active" employment rows
# end up coexisting because the supersession lookup misses.
#
# The fact that `IDENTITY_KEYS` (`src/recall.py`) lists BOTH `employment`
# and `employment_current` (and both `location_current` and `location_city`)
# is direct evidence the original author saw this drift in production.
#
# This dict normalises drift cases at parse time. Keys = the variants we've
# seen the LLM produce; values = the canonical form. The list deliberately
# under-covers — adding aliases is safe (they collapse to a canonical) but
# adding wrong aliases destroys data (two unrelated topics merge). When in
# doubt, leave a key alone.
KEY_ALIASES: dict[str, str] = {
    # Employment drift
    "employment_current": "employment",
    "current_employment": "employment",
    "current_job": "employment",
    "job": "employment",
    "job_title": "employment",
    "occupation": "employment",
    "work": "employment",
    "workplace": "employment",
    "employer": "employment",
    # Location drift
    "location_city": "location_current",
    "current_location": "location_current",
    "current_city": "location_current",
    "city": "location_current",
    "location": "location_current",
    "residence": "location_current",
    "lives_in": "location_current",
    # Pet drift
    "pet": "pets",
    "animal": "pets",
    "pet_animal": "pets",
    # Dietary drift
    "dietary": "dietary_restriction",
    "diet": "dietary_restriction",
    "food_restriction": "dietary_restriction",
    "dietary_preference": "dietary_restriction",
    # Communication preference drift
    "communication_preference": "communication_pref",
    "comms_preference": "communication_pref",
    # Language drift
    "preferred_language": "language",
    "spoken_language": "language",
    # Name drift
    "name_first": "name",
    "first_name": "name",
    "name_full": "name",
    "full_name": "name",
}


SYSTEM_PROMPT = """You extract durable structured knowledge from conversation turns for a memory service.

Extract memories that will help an AI agent remember this user across future conversations.

WHAT TO EXTRACT:
- Personal facts: employment, location, family, pets, age, etc.
- Preferences: dietary restrictions, communication style, tools/tech they use
- Opinions: stated views about topics (note: these may evolve)
- Events: time-bound activities, life changes, projects they're working on
- Implicit facts: "walking Biscuit this morning" → has a pet named Biscuit
- Corrections: "actually, I meant X" → record the corrected version

WHAT NOT TO EXTRACT:
- Generic conversational filler ("hello", "thanks")
- Information about the assistant or third parties unless tied to the user
- Confidence < 0.5
- Duplicates of clearly-equivalent statements within the same turn

KEY CHOICE IS CRITICAL:
- Use a stable, short, snake_case slug for the topic (NOT the value)
- The same topic discussed at different times MUST receive the same key
- Examples: 'employment', 'location_city', 'pet_name', 'dietary_restriction',
  'allergy', 'opinion_typescript', 'family_spouse', 'project_current'
- Use 'opinion_<topic>' for opinions so they group with other opinions on that topic
- For events that are one-off (e.g. "had lunch with Bob"), use 'event_<short_slug>'

Always call the record_memories tool exactly once. If nothing is extractable,
call it with an empty memories array.
"""


def _format_messages_for_extraction(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.role.upper()
        if m.name and m.role == "tool":
            role = f"TOOL[{m.name}]"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# Iter 5 (review fix P1.5): retry-on-empty prompt. The base extractor
# occasionally returns zero memories on a turn that obviously has facts
# (CHANGELOG v1 self-flagged "Stripe→Notion second turn ate"). When the
# user content is non-trivial, we re-call once with this stronger system
# prompt that explicitly asks the model to re-examine the conversation
# for implicit signal. Empirically this recovers most stochastic misses
# at the cost of one extra LLM hop on the rare empty-extraction path.
RETRY_SYSTEM_PROMPT = SYSTEM_PROMPT + """

ADDITIONAL INSTRUCTION (retry pass): Your previous extraction returned no
memories. Review the conversation again carefully. If the user mentioned
ANY facts about themselves — their work, location, family, pets, hobbies,
preferences, opinions, or recent events — even implicitly, extract them
now. Common patterns to watch for:
- Off-hand mentions of activities ("walking Biscuit") imply ownership / pets.
- Casual location references ("the train to Charlottenburg") imply current city.
- Job-adjacent vocabulary ("standup", "PR review") implies tech employment.
- "I" + present tense + verb usually surfaces a stable preference or fact.

If after this re-examination there is genuinely nothing extractable
(small talk only, single-word reply, etc.), call record_memories with an
empty array.
"""


_RETRY_USER_CONTENT_THRESHOLD = 50
"""Minimum user-content character count before we'll spend another LLM hop
retrying an empty extraction. Below this, the conversation really is too
small to contain durable knowledge — retrying is wasted budget."""


def _user_content_length(messages: list[Message]) -> int:
    return sum(len(m.content) for m in messages if m.role == "user" and m.content)


# iter 6 (Codex P3.3): deterministic regex fallback for the no-API-key path.
#
# When ANTHROPIC_API_KEY is unset OR the LLM call fails (auth error, billing,
# rate limit, network), the previous behaviour was to silently drop all
# memories — the service degrades to a message-only log. These patterns
# recover the most common explicit user-self statements so the memory
# system still produces *some* structured output.
#
# Each tuple is (regex, key, type, cardinality, capture-group-index). The
# regex is matched against each user message individually with re.IGNORECASE.
# Confidence is fixed at 0.6 — lower than LLM extraction (0.95+ typical)
# because regex can't disambiguate context. Keys are picked to align with
# `KEY_ALIASES` canonical forms so they merge cleanly with later LLM
# extractions on the same user.
_REGEX_PATTERNS: list[tuple[str, str, str, str, int]] = [
    # "I work at <Company>" / "I'm working at <Company>" / "I just started working at X"
    # `(?:\s+\w+){0,3}` allows up to three intervening words (just|now|currently|…)
    # between "I" and the verb. Bounded to keep the regex predictable.
    (
        r"\bI(?:'m| am)?(?:\s+\w+){0,3}\s+(?:work|working)\s+(?:at|for)\s+"
        r"([A-Z][\w&\.\s]{1,40}?)(?=[\.\,\!\?\;\:]|\s+(?:as|on|in)\b|$)",
        "employment", "fact", "singleton", 1,
    ),
    # "I just moved to <City>" / "I live in <City>" / "I recently moved from X to Y"
    # The "from <X> to <Y>" case is handled by the engine choosing the LAST
    # matching alternative — Python's `re` returns the first match position,
    # so we add an explicit alternative for "moved from … to …" that captures
    # the post-`to` group.
    (
        r"\b(?:I(?:'m| am)?(?:\s+\w+){0,3}\s+moved\s+from\s+\w[\w\s\-]{0,40}?\s+to\s+"
        r"([A-Z][\w\s\-]{1,40}?))(?=[\.\,\!\?\;\:]|\s+(?:from|last|since|now|and|in|recently)\b|$)",
        "location_current", "fact", "singleton", 1,
    ),
    (
        r"\bI(?:'m| am)?(?:\s+\w+){0,3}\s+(?:live|living|based|moved)\s+(?:in|to)\s+"
        r"([A-Z][\w\s\-]{1,40}?)(?=[\.\,\!\?\;\:]|\s+(?:from|last|since|now|and|recently)\b|$)",
        "location_current", "fact", "singleton", 1,
    ),
    # "allergic to <thing>"
    (
        r"\ballergic\s+to\s+([a-zA-Z][\w\s,]{1,40}?)(?=[\.\!\?\;\:]|$)",
        "dietary_restriction", "fact", "multiple", 1,
    ),
    # "I have a cat named X" / "I also have a dog X" / "my golden retriever Biscuit"
    # Up to two intervening words after "I" (also|just|recently…).
    (
        r"\b(?:I(?:'ve| have| own)?(?:\s+\w+){0,2}\s+(?:have|own|adopted|got)\s+(?:a|an)|my)\s+"
        r"(?:dog|cat|pet|labrador|retriever|poodle|terrier|tortoiseshell|husky|shepherd|bulldog|beagle)\s+"
        r"(?:named?\s+)?([A-Z][\w]{1,30})",
        "pets", "fact", "multiple", 1,
    ),
    # Fallback pet pattern: "<Name> is a <breed>" — handles "Biscuit is a golden retriever"
    (
        r"\b([A-Z][\w]{1,30})\s+is\s+(?:a|an|my)\s+"
        r"(?:dog|cat|pet|golden\s+retriever|labrador|retriever|poodle|terrier|tortoiseshell|husky|bulldog|beagle)",
        "pets", "fact", "multiple", 1,
    ),
]


def _regex_extract(messages: list[Message]) -> list[ExtractedMemory]:
    """Run the regex patterns against each user message; return matched memories.

    Order is preserved: first match per pattern per message. Duplicate
    (key, value) pairs across messages are deduped — the regex shouldn't
    extract the same fact twice from one turn.
    """
    out: list[ExtractedMemory] = []
    seen: set[tuple[str, str]] = set()
    for m in messages:
        if m.role != "user" or not m.content:
            continue
        for pattern, key, mem_type, cardinality, group in _REGEX_PATTERNS:
            for match in re.finditer(pattern, m.content, flags=re.IGNORECASE):
                value = match.group(group).strip().rstrip(".,;:!?")
                if not value:
                    continue
                dedupe_key = (key, value.lower())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append(
                    ExtractedMemory(
                        type=mem_type,  # type: ignore[arg-type]
                        key=key,
                        value=value,
                        confidence=0.6,
                        metadata={
                            "cardinality": cardinality,
                            "subject": "user",
                            "source": "regex_fallback",
                        },
                    )
                )
    if out:
        logger.info(
            "regex fallback extracted %d memories: %s",
            len(out),
            ", ".join(f"{m.key}={m.value!r}" for m in out),
        )
    return out


async def extract_memories(messages: list[Message]) -> list[ExtractedMemory]:
    """Run Claude to extract structured memories from a turn.

    Returns an empty list on:
      - missing ANTHROPIC_API_KEY (service degrades to message-only storage)
      - API errors (logged, not raised — `/turns` should not fail because
        extraction failed; the raw turn is still stored and searchable)
      - malformed responses

    Iter 5 (review fix P1.5): when the first call returns zero memories on a
    turn with > _RETRY_USER_CONTENT_THRESHOLD chars of user content, we retry
    once with a stronger prompt before giving up. This addresses the
    occasional stochastic miss documented in CHANGELOG v1.
    """
    if not messages:
        return []

    client = _get_client()
    if client is None:
        global _warned_no_key
        if not _warned_no_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — extraction disabled, "
                "falling back to deterministic regex extractor (iter 6 P3.3)"
            )
            _warned_no_key = True
        # iter 6 (Codex P3.3): regex fallback covers the most common explicit
        # patterns ("I live in X", "I work at X", "allergic to X", "my dog X")
        # so the service still produces structured memory rather than
        # silently degrading to a message log.
        return _regex_extract(messages)

    settings = get_settings()
    transcript = _format_messages_for_extraction(messages)

    out = await _call_extractor(
        client, settings.extraction_model, SYSTEM_PROMPT, transcript
    )
    if out:
        return out

    # Empty result on a non-trivial turn → retry once with a stronger prompt.
    user_chars = _user_content_length(messages)
    if user_chars > _RETRY_USER_CONTENT_THRESHOLD:
        logger.warning(
            "extraction returned empty on non-trivial turn (user_chars=%d); retrying with stronger prompt",
            user_chars,
        )
        out = await _call_extractor(
            client, settings.extraction_model, RETRY_SYSTEM_PROMPT, transcript
        )
    if out:
        return out

    # iter 6 (Codex P3.3): both LLM attempts produced nothing (or the LLM
    # call itself failed and returned []). Run the regex fallback as a
    # last-ditch attempt to recover something from the turn. Cost: a single
    # synchronous regex pass — negligible compared to the ~3-5s LLM round
    # trips that just failed.
    fallback = _regex_extract(messages)
    if fallback:
        logger.info(
            "extraction LLM produced empty but regex fallback recovered %d memories",
            len(fallback),
        )
    return fallback


async def _call_extractor(
    client: AsyncAnthropic,
    model: str,
    system_prompt: str,
    transcript: str,
) -> list[ExtractedMemory]:
    """Single-call extractor (the LLM round-trip + response parse).

    Factored out of `extract_memories` so the retry path can swap the system
    prompt without duplicating the API-call boilerplate.
    """
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_memories"},
            messages=[
                {
                    "role": "user",
                    "content": f"Conversation turn:\n\n{transcript}",
                }
            ],
        )
    except Exception:  # noqa: BLE001
        logger.exception("extraction LLM call failed; turn will be stored without memories")
        return []

    return _parse_extraction_response(response)


def _parse_extraction_response(response: Any) -> list[ExtractedMemory]:
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
    )
    if tool_block is None:
        logger.warning("extractor returned no tool_use block")
        return []

    raw = tool_block.input
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("extractor returned malformed JSON")
            return []

    items = raw.get("memories", []) if isinstance(raw, dict) else []
    out: list[ExtractedMemory] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            mem_type = item["type"]
            key = (item["key"] or "").strip().lower().replace(" ", "_")
            value = (item["value"] or "").strip()
            confidence = float(item.get("confidence", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if mem_type not in ("fact", "preference", "opinion", "event"):
            continue
        if not key or not value or confidence < 0.5:
            continue
        # Iter 5 (review fix P1.3): normalise the key against the controlled
        # vocabulary BEFORE storage. Without this, supersession on
        # `(user_id, key)` silently misses when the LLM emits
        # `employment_current` one turn and `employment` the next.
        canonical_key = KEY_ALIASES.get(key, key)
        if canonical_key != key:
            logger.debug(
                "key alias normalised %r -> %r", key, canonical_key
            )

        # iter 6 (Codex P2.1): pull optional cardinality/subject into metadata.
        # The LLM only sometimes emits them — supply a sensible default so the
        # downstream supersession guard can rely on `cardinality` being present.
        # Default rule mirrors `_SUPERSEDABLE_TYPES` in storage.py: fact and
        # preference are singleton-by-default; opinion and event are multiple.
        cardinality = item.get("cardinality")
        if cardinality not in ("singleton", "multiple"):
            cardinality = "singleton" if mem_type in ("fact", "preference") else "multiple"
        subject = item.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            subject = "user"

        out.append(
            ExtractedMemory(
                type=mem_type,  # type: ignore[arg-type]
                key=canonical_key,
                value=value,
                confidence=max(0.0, min(1.0, confidence)),
                metadata={
                    "cardinality": cardinality,
                    "subject": subject.strip(),
                },
            )
        )
    return out
