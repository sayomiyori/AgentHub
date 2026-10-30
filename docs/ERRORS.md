# Reproduced defects (2026-10-03)

## 2026-10-27: Ambiguous relative downgrade at merge

The candidate merge has two parents, so Alembic rejected `downgrade -1` with
`Ambiguous walk` on an isolated PostgreSQL database. The verified offline SQL
for `downgrade 003_tenant_resources` removes only the merge marker and restores
both parent version rows; the test/CI commands now use the explicit target.
Further branch downgrades can remove data and must not be used for verification.

## 2026-10-27: Uncertain document acknowledgements

The unified release candidate deleted the upload after any database commit error,
even if the database committed but its acknowledgement was lost. A queue error
also overwrote document status even though the worker could already have run.
The upload now retains the file on an uncertain database result and does not
change status on an uncertain queue result. These cases may leave an orphan file
or a pending document and require operator reconciliation; they do not trigger
automatic retries. `tests/test_document_uncertain_outcomes.py` reproduces both
failure paths without contacting a database or broker.

## 2026-10-26: Invalid operator model could fail before a bounded job claim

The unshipped settings resolver rejected invalid operator identifiers that
enabled PlatformSettings still accepted. claim_job runs before the processor's
generation try block, so a configuration ValueError could roll back the claim
and leave a job pending for scanner publication. Independent review reproduced
the accepted invalid config and throwing resolver; no real provider was used.

Enabled startup now requires the same shared ASCII model-identifier guard used
by saved settings, read metadata and generation validation. Invalid enabled
operator settings fail startup; invalid frozen job models fail before provider
access in the existing bounded generation error path. Disabled reads retain
their fixed unavailable error. Startup regressions include HTML/query/newline
identifiers. No existing deployment configuration was changed.

## 2026-10-09: Optional semantic cache resilience

Redis read/write errors escaped the optional cache and interrupted RAG requests.
Reads now become misses and writes become no-ops, with 0.5 s socket/connect
timeouts and no transport retry. Malformed/oversized JSON, expired timestamps,
invalid vectors and unusable source/answer data are ignored. Hits no longer
renew an entry's age. Metadata rejection precedes vector comparison.

`python -m pytest tests/test_semantic_cache_resilience.py -q`: 22 passed.
Full suite: 250 passed; Ruff and the CI Mypy scope passed. Independent adversarial
review approved after huge-number, malformed source and invalid UTF-8/NUL cases.
The legacy cache remains global and has no document-change invalidation;
it must not be exposed as tenant-safe RAG.

- Cache hits returned the original generation's tokens and cost, charging the message again. `test_rag_http_cache_and_usage` reproduced `15 != 0`; cache hits now report zero incremental generation tokens/cost and create no new usage row.
- Duplicate document tasks appended duplicate chunks (`2` stored vs `chunk_count=1`). `test_document_worker_duplicate_does_not_duplicate_chunks` reproduced this. Processing now locks the document row until the atomic final commit and skips already-ready documents.
- Invalid embedding dimensions poisoned the SQLAlchemy transaction; error handling raised `PendingRollbackError` instead of marking the document failed. `test_worker_database_failure_marks_document_failed` reproduced this. Error handling rolls back before reading/updating failure status; embedding count mismatches raise instead of silently dropping chunks.
- Docker's `COPY . .` included the local `.env`, Git history and virtual environment because no `.dockerignore` existed. The ignore file now excludes these paths, runtime uploads and caches. Container path assertions passed after rebuild.
# Pause addendum — 2026-10-03

Reviewer confirmed a concurrent retry race in `app/workers/embed_worker.py`:
the exception handler released the document lock on rollback, then unconditionally
marked the row failed even if another PostgreSQL session had completed processing.
Reproduced with `test_failed_worker_cannot_overwrite_successful_concurrent_retry`.
Recovery now reloads under `FOR UPDATE` and preserves `ready`. Final suite 15 passed;
At the user-requested pause, Docker rebuild/HTTP smoke remained pending.

## Resume verification — 2026-10-04

Final full suite: **15 passed, 3 warnings in 6.49s**; Ruff and limited CI Mypy
passed. The image was rebuilt with the concurrent retry recovery fix; built HTTP
upload → Celery → pgvector → boundary provider → usage → Redis cache and provider
timeout smoke passed. Image assertions confirmed the latest recovery code and
absence of `.env`, `.git` and `.venv`. Exact commands are in
`verification-checkpoint.md`. The initial infrastructure setup errors were caused
by stopped Docker Desktop and disappeared after Engine/retained containers started.

## MCP SDK major-version incompatibility - 2026-10-06

A clean CI install selected MCP 2.x from the unconstrained dependency.
`mcp.server.fastmcp.FastMCP` no longer exists in that major version, so the
legacy HTTP test and three platform startup tests failed during application
import (40 passed, 4 failed). The existing server uses the SDK v1 SSE API.

Constrain the dependency to `mcp>=1.0.0,<2.0.0` (commit b3b4cf1). The local
44-test suite uses SDK 1.30.0; independent review also verified server creation
and its `/sse` and `/messages` routes. A clean CI install and its HTTP/startup
tests detect this compatibility regression without provider API calls.

## Groq platform completion boundary - 2026-10-06

Task3's first review reproduced three completion-boundary defects with the real
SDK over controlled HTTP: confirmed authentication rejection lost its error
classification, legacy function-call/non-assistant responses were accepted, and
an escaped lone surrogate passed validation but failed UTF-8 serialization.
The bounded platform path now returns static typed errors and rejects invalid
role/tool/refusal/text/usage before storage. Standalone optional tools remain
supported. Transport errors and quota failures do not trigger a second request
or fallback; their effect remains uncertain for the durable processor.

## Answer envelope timezone - 2026-10-06

An isolated PostgreSQL session with `SET LOCAL TIME ZONE 'Europe/Moscow'`
reproduced an answer envelope timestamp ending in `+03:00`, violating the UTC
contract. Completion now converts the database clock to UTC before serializing
`occurred_at`; lease comparisons continue to use the actual database clock.
`test_answer_timestamp_remains_utc_with_non_utc_database_session` covers it.

## Container packaging tool advisories - 2026-10-07

The controlled Task8 image audit found advisory-bearing pip24.0 and
setuptools79.0.1 inherited from the cached Python base/build environment.
The Dockerfile now upgrades existing packaging tools to pip>=26.2 and
setuptools>=83.0.0 before installing the unchanged application requirements.
Verify the built image's installed package inventory with pip-audit; auditing
only the developer virtualenv does not establish container dependency safety.

## 2026-10-29: Configured embedding provider failures became fallback vectors

With a configured Gemini client, an exception or an empty response produced a
SHA-256-derived vector. Partial batch failures could therefore store mixed
representations and mark the document ready. The existing no-key offline mode
is retained for compatibility and is not semantic search evidence.

A configured client now rejects exceptions and invalid vectors with a static
`EmbeddingUnavailable` error. The document worker rolls back and marks failed;
direct RAG requests return a fixed 503. Agent knowledge-tool failures stop further
generation and return a fixed unavailable answer through the existing response
path, preserving already incurred usage. Real PostgreSQL fixture checks verify
no chunk persistence and retained linked usage. Existing stored vectors have no
provenance metadata, and require a separate reindex/adoption decision.

## 2026-10-29: Tenant indexing used standalone fallback without a provider

Tenant document workers and both tenant search paths used deterministic offline
vectors when Gemini was not configured. Tenant operations now require a provider
client; standalone fallback remains for backward compatibility. A missing client
marks a tenant document failed and prevents direct or agent knowledge retrieval.
In the isolated no-network PostgreSQL/Redis fixture, 28 tests passed, including
worker failure, tenant isolation and agent usage preservation.

Review also reproduced pgvector cosine NaN at extreme float32 vector scales and
eight-decimal query serialization rounding small values to zero. Provider vectors
now reject unsafe cosine norms and query serialization keeps nine significant
digits. Regressions include the actual pgvector search on the isolated fixture.
Existing vectors still have no source metadata and are not repaired by this guard.

## Gemini embedding output dimensions — 2026-10-29

For `gemini-embedding-001`, document and query requests explicitly ask for 1536
values, matching the existing database column. The older `models/embedding-001`
does not support the dimension parameter and remains unchanged; other model
identifiers also retain their existing request shape. Unexpected dimensions still
fail closed. No model configuration, existing vector or database was changed.
The supported parameter and the new model's 3072 default are documented at
https://ai.google.dev/gemini-api/docs/embeddings and
https://ai.google.dev/api/embeddings. This is an offline SDK boundary test, not
verification against the live provider or proof of retrieval quality.

## 2026-10-30: Source file changed while indexing

A source replacement or removal during provider work previously allowed the
worker to commit ready chunks derived from the older input without noticing.
Before writing chunks, the worker now compares the current source SHA-256 with
the exact parsed snapshot and fails with a static error when they differ or the
file is unreadable. The existing transaction rollback/failed-state path applies.
This is an observed-change check, not a lock against external changes after the
check. Generation activation still requires a separate source fence.
Isolated PostgreSQL regressions exercise replacement, removal and existing
worker idempotency/concurrent retry/tenant behavior.

## 2026-10-30: Explicit embedding SDK request limits

Embedding client construction previously left request timeout and transport
retry policy to SDK defaults. It now sets a 20,000 ms SDK request timeout and
one total attempt. Genuine SDK tests over local HTTP transport prove no repeat
request for 429, 503 and a read timeout; errors remain static without provider
details. This bounds individual transport operations, not total document time
or all SDK execution. No live provider or deployment was used for verification.

## 2026-10-30: Limit embedding work per document

A valid upload could split into many chunks and trigger a provider request for
each. The worker now checks MAX_DOCUMENT_CHUNKS after splitting and before any
embedding call. The validated operator setting defaults to 1000 and accepts
1..10000. Larger documents follow the existing failed-state transaction path;
no new chunks or partial embeddings are committed. Tests cover the exact limit
and one over on isolated PostgreSQL with controlled embedding calls. This is a
chunk-count bound, not a guarantee about parsing memory, elapsed time or billing.

## 2026-10-30: Tenant search did not enforce recorded embedding model

Tenant retrieval previously compared vectors without checking their recorded
embedding source/model, and an old cached answer could hide a newly invalid
receipt. The shared tenant pgvector query now checks the exact Gemini/model/1536
receipt for the full selected corpus in the same SQL snapshot as vector selection.
Missing, malformed, offline or mismatched receipts fail with the existing fixed
embedding-unavailable path. Direct RAG validates before cache access, tenant cache
policy is versioned, and the agent passes its query embedder's actual model.
Standalone queries and cache identities remain compatible.

The intermediate guard materializes at most 10001 tenant chunks and fails closed
above 10000; this bounds row count, not row bytes or elapsed time. Recorded
metadata is not independent proof of quality or vector origin. This check does
not implement the future generation activation fence or complete concurrent
corpus/cache snapshot semantics. Existing unknown vectors are not reindexed.

## 2026-10-30: Tenant search could return retained chunks from failed documents

The worker preserves previous chunks on a failed reindex for recovery, but tenant
retrieval checked only their embedding receipts. A cached or newly generated
answer could therefore use chunks belonging to a pending, processing or failed
document. The shared tenant query now requires every selected chunk's document
to be ready in the same SQL snapshot as receipt validation and vector selection.
It fails closed rather than returning a partial corpus. Direct RAG checks before
cache reads; the agent uses the same guard. Foreign tenants and documents without
chunks do not block the selected corpus, and retained chunks are not deleted.
This is a per-query eligibility check, not the planned generation activation fence.
