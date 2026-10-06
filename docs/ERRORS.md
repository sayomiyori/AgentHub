# Reproduced defects (2026-10-03)

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
