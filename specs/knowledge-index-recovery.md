# Knowledge index recovery contract (planned)

## Goal and boundary

Enable an operator to rebuild a tenant's knowledge index when its existing
embedding source is unknown or incompatible with the intended model. Existing
fallback matches are evidence of the fallback algorithm; nonmatches are unknown,
not proof of a provider or model. This document specifies future work only.
No live deployment, provider activation, existing-data write or reindex is
approved by this specification. Standalone legacy documents require a separate
adoption decision; do not silently attach them to a tenant.

## Safety properties

- A tenant's active searchable corpus has one explicit embedding provider,
  model, output dimension and generation identifier. Existing unlabelled vectors
  remain unknown; a migration must not infer their origin or certify them.
- Build the complete replacement in isolated staging without modifying the
  active corpus. A source snapshot must cover every included document, its tenant,
  revision and a hash of the exact source bytes processed. Recheck the input at
  activation and prevent file changes between that check and the switch, including
  changes outside the database; `content_revision` alone does not prove file
  stability. Missing or changed input aborts activation. Never mix
  old and new model generations during retrieval, including cache hits and the
  agent knowledge tool.
- Activate the complete generation atomically only after all vectors are present,
  valid and verified. Concurrent document upload, delete or update must either
  be included in the generation or cause activation to fail; never silently
  omit a concurrent change. Tenant A must not read or alter tenant B's index.
- On failure before commit preserve the previous generation's data. An uncertain
  commit can have activated either generation: do not claim an outcome or retry
  based solely on a lost acknowledgement; reconcile durable state first and allow
  only a complete old or complete new generation. A generation of unknown or
  incompatible origin must not be made searchable merely because it was retained;
  return unavailable until compatibility is established. Rollback must be a
  reviewed generation switch, not a destructive cleanup.
- Fence older workers, delayed jobs and mixed service versions: after activation
  they cannot insert legacy vectors into the active generation or query chunks
  without applying its generation filter. Activation must fail closed while a
  writer or reader capable of bypassing that fence remains active.
- Cache identity includes tenant, corpus generation and embedding model. Queries
  must bind the active generation and model consistently even during activation;
  old in-flight writers cannot populate the new generation's cache namespace.
- Never log source text, embeddings, credentials, file paths or provider error
  bodies. Return fixed status and bounded progress only after authorization.
  Estimate provider usage and cost before a real rebuild. Do not infer quality
  from successful indexing or deterministic test vectors.

## Acceptance before any adoption

1. Specify the exact schema migration, operator interface, permissions, workload
   bounds and recovery procedure before implementation. Migration upgrades and
   downgrades must be tested on disposable PostgreSQL with pre-existing unknown
   vectors; the original data remains intact and unlabelled.
2. Synthetic PostgreSQL/provider tests cover successful staging and activation,
   empty corpora, mixed/unknown old vectors, tenant isolation, model mismatch,
   missing files, provider errors, duplicate jobs, stale source revisions,
   worker crashes and lost commit acknowledgement, for both committed and
   uncommitted outcomes. Failure preserves the old data and creates no mixed-model
   result; unknown or incompatible old data remains unavailable for search.
3. Concurrent tests exercise upload/delete/activation, stale readers and cache
   writes across the switch, including delayed old workers and old-version readers.
   Validate authorization before both reads and writes.
   Prove active generation and cache namespace agree on every retrieval path.
4. Exercise backup and restore on an isolated copy of representative synthetic
   data. Before touching a real deployment, separately inventory targets, verify
   backups and restore, review the exact adoption plan and obtain the user's
   approval for changes to existing data and provider use.
5. Measure retrieval on representative question/source pairs before and after;
   report actual scope and metrics without claiming production quality from mocks.
   No provider traffic or real data is required for the initial implementation.

A per-document replacement in the current table is insufficient because queries
could observe documents from incompatible models during a partial rebuild. The
specific storage layout and coordination protocol remain to be designed and
reviewed; this contract deliberately does not assert that they already exist.
