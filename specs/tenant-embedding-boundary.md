# Tenant embedding provider boundary

Tenant document indexing and tenant RAG queries must not generate deterministic
fallback vectors when the embedding provider is unconfigured. Require an actual
configured client before reading/indexing a tenant document and before direct or
agent knowledge search. Missing provider uses the existing bounded embedding
unavailable error: worker marks the document failed, direct RAG returns 503,
and agent returns the fixed unavailable answer while retaining incurred usage.
Standalone no-tenant fallback remains available for backwards compatibility.
No credentials, migration, database changes, reindexing or provider calls are
introduced. Existing vectors have unknown origin and are not made compatible by
this guard; enabling a provider against them still requires a separate adoption
and reindex plan.

Reject provider vectors whose float32 cosine norms can underflow or overflow;
the guard deliberately rejects extreme scales even when some might work.
Serialize query components with nine significant digits so small nonzero
float32 values do not become zero before the pgvector cosine calculation.

Test missing-provider tenant worker, direct query and agent tool, including no
fallback call; preserve legacy path; exercise worker with isolated PostgreSQL,
no external network, and independent review. Existing synthetic tenant tests
that mock embeddings must explicitly supply a controlled provider boundary.
