"""Semantic RAG response cache backed by Redis (cosine similarity + LSH bucket)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import redis

from app.config import get_settings
from app.metrics import observe_semantic_cache

_client: redis.Redis | None = None
TTL_SECONDS = 24 * 3600
SIMILARITY_THRESHOLD = 0.95
MAX_ENTRIES_PER_BUCKET = 64


def _redis() -> redis.Redis | None:
    global _client
    settings = get_settings()
    if not getattr(settings, "semantic_cache_enabled", True):
        return None
    if _client is None:
        try:
            c = redis.from_url(settings.redis_url, decode_responses=True)
            c.ping()
            _client = c
        except Exception:
            return None
    return _client


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lsh_bucket_key(embedding: list[float]) -> str:
    """Locality-sensitive style bucket from sign bits of leading dimensions."""
    bits = "".join("1" if embedding[i] >= 0 else "0" for i in range(min(48, len(embedding))))
    return hashlib.sha256(bits.encode()).hexdigest()[:28]


def _cache_key(bucket: str) -> str:
    return f"semantic_cache:v1:{bucket}"


def get_cached_rag_answer(
    question: str,
    embedding: list[float],
    *,
    top_k: int,
    provider: str | None,
    model: str | None,
) -> dict[str, Any] | None:
    r = _redis()
    if r is None:
        observe_semantic_cache(False)
        return None

    bucket = _lsh_bucket_key(embedding)
    key = _cache_key(bucket)
    raw = r.get(key)
    if not isinstance(raw, str):
        observe_semantic_cache(False)
        return None

    try:
        entries: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError:
        observe_semantic_cache(False)
        return None

    for entry in entries:
        emb = entry.get("embedding")
        if not isinstance(emb, list):
            continue
        if _cosine(embedding, [float(x) for x in emb]) < SIMILARITY_THRESHOLD:
            continue
        meta = entry.get("meta") or {}
        if int(meta.get("top_k", -1)) != top_k:
            continue
        if (meta.get("provider") or "") != (provider or ""):
            continue
        if (meta.get("model") or "") != (model or ""):
            continue
        observe_semantic_cache(True)
        r.expire(key, TTL_SECONDS)
        return entry.get("payload")

    observe_semantic_cache(False)
    return None


def set_cached_rag_answer(
    question: str,
    embedding: list[float],
    *,
    top_k: int,
    provider: str | None,
    model: str | None,
    payload: dict[str, Any],
) -> None:
    r = _redis()
    if r is None:
        return

    bucket = _lsh_bucket_key(embedding)
    key = _cache_key(bucket)
    entry = {
        "embedding": embedding,
        "meta": {"top_k": top_k, "provider": provider or "", "model": model or ""},
        "payload": payload,
        "ts": time.time(),
    }

    raw_existing = r.get(key)
    entries: list[dict[str, Any]] = []
    if isinstance(raw_existing, str):
        try:
            entries = json.loads(raw_existing)
        except json.JSONDecodeError:
            entries = []

    entries = [e for e in entries if isinstance(e, dict)]
    entries.append(entry)
    entries = entries[-MAX_ENTRIES_PER_BUCKET:]

    r.setex(key, TTL_SECONDS, json.dumps(entries, ensure_ascii=False))
