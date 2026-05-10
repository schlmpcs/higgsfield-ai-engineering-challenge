from __future__ import annotations

import hashlib
import logging
import math
from typing import Sequence

from openai import AsyncOpenAI

from .config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None
_warned_no_key = False


def _get_client() -> AsyncOpenAI | None:
    global _client
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _hash_embedding(text: str, dim: int) -> list[float]:
    """Deterministic fallback when no OPENAI_API_KEY is configured.

    Quality is far below a real embedding model — the service degrades to
    keyword + fact-priority recall, which is documented in the README.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = _LCG(seed)
    vec = [rng.next_normal() for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class _LCG:
    """Tiny deterministic PRNG (no numpy dep at hot path)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF or 1

    def next_uniform(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 11) / (1 << 53)

    def next_normal(self) -> float:
        u1 = max(self.next_uniform(), 1e-12)
        u2 = self.next_uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


async def embed_text(text: str) -> list[float]:
    settings = get_settings()
    client = _get_client()
    if client is None:
        global _warned_no_key
        if not _warned_no_key:
            logger.warning(
                "OPENAI_API_KEY not set — using deterministic hash embeddings (low recall quality)"
            )
            _warned_no_key = True
        return _hash_embedding(text, settings.embedding_dim)
    try:
        resp = await client.embeddings.create(
            model=settings.embedding_model, input=text[:8000]
        )
        return resp.data[0].embedding
    except Exception:  # noqa: BLE001
        logger.exception("embedding API call failed; using hash fallback")
        return _hash_embedding(text, settings.embedding_dim)


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    settings = get_settings()
    client = _get_client()
    cleaned = [t[:8000] if t else " " for t in texts]
    if client is None:
        return [_hash_embedding(t, settings.embedding_dim) for t in cleaned]
    try:
        resp = await client.embeddings.create(
            model=settings.embedding_model, input=list(cleaned)
        )
        return [d.embedding for d in resp.data]
    except Exception:  # noqa: BLE001
        logger.exception("batch embedding API call failed; using hash fallback")
        return [_hash_embedding(t, settings.embedding_dim) for t in cleaned]
