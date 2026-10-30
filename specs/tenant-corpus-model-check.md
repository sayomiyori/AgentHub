# Tenant corpus model receipt check

For tenant retrieval require every stored chunk in the selected tenant corpus to
have exactly the expected `{provider: gemini, model: configured query model,
dimensions: 1536}` embedding receipt. Missing, malformed, offline or mismatched
receipts fail with the existing fixed embedding-unavailable response; do not
silently skip bad chunks or label unrecorded vectors compatible. Use one SQL
snapshot for the receipt check and vector selection, including invalid chunks
outside the top-k. Scope the whole query to the authorized tenant. Empty corpus
returns empty results. Standalone null-tenant retrieval retains its old behavior.

Direct RAG must perform this check before cache reads, and agent knowledge search
uses the same retrieval path with the query embedder's actual configured model.
Version the tenant cache namespace to exclude old-policy cached answers. Exact
model strings are compared conservatively without inferring aliases. Receipt
matching is not independent proof of provider quality or correct provenance.
No migrations, reindex, data changes or real provider calls; the future generation
switch/fence and concurrent corpus-snapshot design remain separate requirements.
Until a persisted generation receipt exists, this path scans at most 10001 tenant
chunks and rejects corpora over 10000 rather than search a partial corpus. This
bounds materialized row count, not row bytes or elapsed query time. Vector index
acceleration and large-corpus performance are not established. Standalone cache
keys retain their previous format.

Real PostgreSQL tests cover matching/mismatched/unknown/mixed/offline data,
foreign invalid data, empty corpus, malformed receipt types and a cached answer
whose chunk receipt becomes invalid without a document revision change. Verify
both agent and direct RAG paths, existing tenant isolation, and standalone tests.
