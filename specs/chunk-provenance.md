# Chunk provenance receipt

Newly indexed chunks record internal provenance in their existing JSON metadata:
provider (`gemini` or `offline_sha256`), exact configured model (or `sha256-v1`),
dimension 1536 and SHA-256 of the exact source bytes used by the parser. Read the
source once into bytes before parsing and hashing; PDF parsing also uses those
same bytes. Preserve text decoding semantics, existing source/chunk index fields,
worker idempotency, scope and atomic chunk/status commit. Old chunks remain
unlabelled. No schema migration, reindex, provider activation or API expansion.
This receipt is a record of processing, not a quality certificate or a replacement
for the planned generation fence. Tests cover metadata persistence, source bytes,
PDF/text parsing, ready-task idempotency and failure rollback on isolated PostgreSQL.
