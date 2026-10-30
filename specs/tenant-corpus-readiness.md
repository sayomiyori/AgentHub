# Tenant retrieval requires ready indexed documents

Extend the shared tenant corpus guard: a stored chunk whose owning document is
pending, processing or failed makes that tenant corpus unavailable. Apply the
check in the same SQL snapshot as model compatibility and hit selection; keep
all stored chunks for recovery and do not silently return a partial corpus.
Direct RAG performs this before cache access; the agent uses the same guard.
Documents with no chunks are not candidates and do not block other indexed
documents. Foreign tenant states do not affect the selected corpus. Preserve
standalone behavior. No schema, data mutation, reindex or provider changes.
This is a per-query eligibility check, not the planned generation switch/fence.
Test state changes after a cached answer, both reader paths, foreign and no-chunk
cases, preservation of stored chunks, and the existing source-mutation failures
on isolated PostgreSQL.
