"""Semantic RAG response cache backed by Redis (cosine similarity + LSH bucket)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from decimal import Decimal
from typing import Any

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

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
            _client = redis.from_url(settings.redis_url, decode_responses=True,
                                     socket_connect_timeout=0.5, socket_timeout=0.5,
                                     retry=Retry(NoBackoff(), 0))
        except (redis.RedisError, ValueError):
            return None
    return _client


def _cosine(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0 or n != len(b) or not all(math.isfinite(x) for x in (*a, *b)):
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    score = dot / (na * nb)
    return score if math.isfinite(score) else 0.0


def _lsh_bucket_key(embedding: list[float]) -> str:
    """Locality-sensitive style bucket from sign bits of leading dimensions."""
    bits = "".join("1" if embedding[i] >= 0 else "0" for i in range(min(48, len(embedding))))
    return hashlib.sha256(bits.encode()).hexdigest()[:28]


def _cache_key(bucket: str) -> str:
    return f"semantic_cache:v1:{bucket}"


def _entries(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, str) or len(raw) > 4 * 1024 * 1024:
        return []
    try:
        decoded = json.loads(raw)
    except (ValueError, RecursionError):
        return []
    if not isinstance(decoded, list):
        return []
    now = time.time()
    return [e for e in decoded[-MAX_ENTRIES_PER_BUCKET:] if isinstance(e, dict)
            and type(e.get("ts")) in (int, float) and now - TTL_SECONDS < e["ts"] <= now]


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
    try:
        entries = _entries(r.get(key))
    except (redis.RedisError, UnicodeError):
        observe_semantic_cache(False)
        return None
    for entry in entries:
        meta = entry.get("meta")
        payload = entry.get("payload")
        if not isinstance(meta, dict) or not isinstance(payload, dict):
            continue
        if not isinstance(payload.get("answer"), str) or not isinstance(payload.get("sources"), list):
            continue
        if not all(isinstance(source, dict)
                   and type(source.get("chunk_id")) is int
                   and isinstance(source.get("document_title"), str)
                   and isinstance(source.get("chunk_text"), str)
                   and type(source.get("rerank_score", 0)) in (int, float)
                   and -1e308 <= source.get("rerank_score", 0) <= 1e308
                   for source in payload["sources"]):
            continue
        text_fields = [payload["answer"]]
        for source in payload["sources"]:
            text_fields.extend((source["document_title"], source["chunk_text"]))
        try:
            if any("\x00" in value for value in text_fields):
                continue
            for value in text_fields:
                value.encode("utf-8")
        except UnicodeError:
            continue
        if (meta.get("top_k") != top_k or meta.get("provider", "") != (provider or "")
            or meta.get("model", "") != (model or "")):
            continue
        emb = entry.get("embedding")
        if not isinstance(emb, list) or any(type(x) not in (int, float) for x in emb):
            continue
        try:
            similarity = _cosine(embedding, emb)
        except (ValueError, OverflowError):
            continue
        if similarity < SIMILARITY_THRESHOLD:
            continue
        observe_semantic_cache(True)
        return payload

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

    try:
        entries = _entries(r.get(key))
    except (redis.RedisError, UnicodeError):
        return
    entries.append(entry)
    entries = entries[-MAX_ENTRIES_PER_BUCKET:]

    def _default(obj: object) -> object:
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        encoded = json.dumps(entries, ensure_ascii=False, default=_default, allow_nan=False)
        if len(encoded) <= 4 * 1024 * 1024:
            r.set(key, encoded, ex=TTL_SECONDS)
    except (redis.RedisError, UnicodeError, ValueError, TypeError):
        return
