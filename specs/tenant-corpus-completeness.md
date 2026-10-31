# Tenant corpus structural completeness

Extend the existing tenant-only shared retrieval guard in one SQL snapshot.
For every document with stored chunks, and every ready document even without
stored chunks, require its actual chunk count to equal document.chunk_count.
The chunk indexes must be unique and cover exactly 0..count-1. A ready empty
document with declared count zero is valid. Non-ready documents without chunks
remain outside retrieval eligibility; the existing readiness guard still rejects
non-ready documents with chunks. Foreign tenant corruption must not interfere.

Fail with the existing fixed EmbeddingUnavailable response, including before
direct cache reads and through the agent. Do not silently serve a partial corpus
or mutate existing data. Standalone behavior is unchanged. Materialize at most
10001 selected documents and reject above 10000; retain the existing 10000-chunk
limit. These bounds limit row counts, not row bytes, elapsed time or query cost.

Plan: add PostgreSQL regressions first, reproduce corrupt counts/indexes and total
chunk loss after a cached answer, then extend the shared SQL guard and run the
worker/tenant fixture, host tests, Ruff, Mypy and independent review. Also verify
valid multi-chunk/empty documents, foreign corruption and the document-count cap.
No migration, source reindex, provider activation or deployment is included.
Matching structural metadata does not prove source completeness or vector quality;
the planned generation fence and concurrent response/cache semantics remain open.
