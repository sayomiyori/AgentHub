# Tenant knowledge generation design (draft, not implemented)

This design refines `knowledge-index-recovery.md`. Its first release is a new
opt-in path; existing standalone behavior and tables are not migrated in place.
No existing vector is assumed to have a known model or provider. Deployment and
real provider operations remain separate from a tested implementation.

## Proposed storage

An Alembic revision after `004_merge_ai_resources` adds four tables:

- `knowledge_generations`: UUID primary key, non-null tenant UUID, explicit
  provider/model/dimensions, state (`building`, `ready`, `retired`), a source
  manifest digest and creation timestamp. Unique `(tenant_id, id)` supports
  tenant-qualified foreign keys. It does not make a generation active by itself.
- `knowledge_generation_documents`: tenant UUID, generation UUID, existing
  document ID, source revision, SHA-256 digest of exact source bytes and chunk
  count. Composite foreign key to `(tenant_id, id)` in generations and unique
  `(generation_id, document_id)`. A tenant-qualified foreign key to a new unique
  `(tenant_id, id)` key on documents prevents cross-tenant attachment; legacy
  null-tenant rows are never assigned. Define deletion semantics and validate
  foreign document IDs before insertion and activation.
- `knowledge_generation_chunks`: generation UUID, document ID, chunk index,
  text, metadata and `vector(1536)`, with a composite foreign key to the staged
  document and unique `(generation_id, document_id, chunk_index)`; retrieval
  queries must filter by generation and tenant; indexes alone do not enforce it.
- `knowledge_active_generations`: one row per tenant with an active generation
  UUID and monotonic revision; composite FK prevents cross-tenant pointers.
  Missing row means unavailable, never a fallback to the old `chunks` table.

Migration is additive: no old document, vector, status, cache or index is changed.
Downgrade would remove new-generation data and must be explicitly documented and
exercised only on disposable databases; production downgrade requires an approved
backup and data-impact plan. No stamping or applied migration edits.

## Coordination and operator boundary

1. Authenticate the operator and authorize the tenant on every request. An
   eventual owner-only API must issue a bounded rebuild request using server-set
   provider/model settings, not arbitrary credentials or file paths. The exact
   public API is intentionally deferred until the durable protocol is tested.
2. Freeze immutable input bytes for every selected tenant document into private,
   access-controlled staging. Staged bytes and their SHA-256 are the actual
   processing input. Establish an explicit source-change fence before final
   validation: upload/delete/update and out-of-band file changes cannot occur
   through the switch. If the deployment cannot enforce that fence, do not switch.
   Source retention and storage budget require a separate implementation decision.
3. Generate and validate staged vectors only. Record the full document/revision/
   source manifest; duplicates must not create duplicate staged chunks. On failure
   do not modify the active pointer. All staged writers and activation must share
   a lock/state protocol; ready or active generations are immutable, including to
   delayed duplicate jobs. Unknown acknowledgement is reconciled by
   reading durable generation state before any repeated operation.
4. Under a per-tenant database transaction lock, confirm the source fence,
   tenant, manifest, generation completeness and expected pointer revision,
   then atomically replace the active pointer. Lost COMMIT acknowledgements may
   mean old or new: re-read the pointer and never assume failure. Changes to the
   corpus invalidate the revision and prevent activation of stale staging.
5. Query resolves and binds tenant, active generation and model consistently for
   the entire read, including agent retrieval and cache namespace. Missing,
   unready, mismatched or unknown generation means fixed unavailable response;
   no fallback to legacy `chunks`. Define snapshot isolation and cache handling
   before implementing reads, and prove old in-flight cache writers cannot poison
   the new namespace.
6. Block activation unless all readers/writers that can bypass the new scope have
   been drained or denied database access. This includes delayed Celery jobs and
   separately deployed old versions. A boolean flag alone does not fence an old
   process. The exact deployment fencing proof is an open requirement.
7. Define ordinary uploads, updates and deletes after activation under the same
   tenant coordination. A delete must make removed content unavailable to all
   retrieval and cache paths, including concurrent readers, in accordance with
   an explicit transaction contract. Define the completion boundary for in-flight
   responses; a response already sent cannot be revoked. Staged and active copies must not bypass
   tenant-qualified deletion. Until implemented, do not activate this design.

## Workload and acceptance gates

Use bounded document count, bytes, chunks, provider calls and elapsed time;
choose and document concrete limits before exposing an operator endpoint. Stage
and switch on synthetic isolated PostgreSQL with no external provider. Verify
additive migration up/down/up, model drift, pre-existing unknown vectors, unique
and tenant constraints, duplicate submissions, source changes, stale revisions,
crashes, both lost-commit outcomes, all read paths and cache isolation. Use
concurrent sessions to demonstrate that old and new corpora never mix, and test
that old workers cannot write after the fence, staged writers cannot race the
switch, and deleted content cannot reappear in results or cache. Backup/restore and a representative
retrieval evaluation precede production adoption. Unknown implementation details
above block activation and must not be treated as approved defaults.
