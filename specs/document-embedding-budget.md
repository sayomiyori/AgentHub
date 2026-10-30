# Per-document embedding work limit

Add operator setting MAX_DOCUMENT_CHUNKS (default 1000, valid range 1..10000).
After parsing/splitting, reject a document with more chunks than this limit before
any embedding provider request. Use the existing failed-document transaction
path and a fixed error; no automatic retry or partial chunk commit. Keep files,
old chunks and old ready documents intact. This bounds chunk/SDK-call count
for newly processed documents, not parsing memory, corpus size, total elapsed
time, HTTP exchanges during SDK redirects, provider billing or task cancellation. Existing deployments are unchanged
until separately configured/deployed. Document the intentional new rejection.

Tests: validated settings, boundary/oversize documents, no provider call on
rejection, persisted failed state and no new chunks, existing worker/tenant
regressions on isolated PostgreSQL. No real provider or migration.
