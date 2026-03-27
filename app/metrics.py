"""Prometheus metrics for AgentHub."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

llm_requests_total = Counter(
    "llm_requests_total",
    "LLM API requests",
    ["provider", "model", "status"],
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "LLM tokens",
    ["provider", "model", "direction"],
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Estimated LLM cost in USD",
    ["provider", "model"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["provider", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieve+rerank duration",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

documents_total = Gauge("documents_total", "Uploaded documents count")
chunks_total = Gauge("chunks_total", "Total chunks in database")

semantic_cache_hit_ratio = Gauge(
    "semantic_cache_hit_ratio",
    "Rolling ratio of semantic cache hits (hits / (hits + misses))",
)

embedding_duration_seconds = Histogram(
    "embedding_duration_seconds",
    "Embedding API / fallback duration",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

mcp_tool_calls_total = Counter(
    "mcp_tool_calls_total",
    "MCP tool invocations",
    ["server", "tool"],
)

_semantic_hits = 0
_semantic_misses = 0


def observe_semantic_cache(hit: bool) -> None:
    global _semantic_hits, _semantic_misses
    if hit:
        _semantic_hits += 1
    else:
        _semantic_misses += 1
    total = _semantic_hits + _semantic_misses
    semantic_cache_hit_ratio.set(_semantic_hits / total if total else 0.0)


def observe_llm_response(
    *,
    provider: str,
    model: str,
    status: str,
    inp: int,
    out_t: int,
    cost_usd: float,
    duration_s: float,
) -> None:
    prov = (provider or "unknown").lower()
    mod = model or "unknown"
    llm_requests_total.labels(provider=prov, model=mod, status=status).inc()
    llm_tokens_used_total.labels(provider=prov, model=mod, direction="input").inc(inp)
    llm_tokens_used_total.labels(provider=prov, model=mod, direction="output").inc(out_t)
    llm_cost_usd_total.labels(provider=prov, model=mod).inc(cost_usd)
    llm_request_duration_seconds.labels(provider=prov, model=mod).observe(duration_s)


@contextmanager
def rag_retrieval_timer() -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rag_retrieval_duration_seconds.observe(time.perf_counter() - t0)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
