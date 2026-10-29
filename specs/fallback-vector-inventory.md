# Read-only fallback vector inventory

Provide an operator-only command that inspects the configured AgentHub database
in a repeatable-read, read-only transaction and prints aggregate counts only.
Match every stored float32 vector against the existing SHA-256 fallback algorithm
for its chunk text. A match identifies this algorithm; a non-match is unknown and
must never be labelled Gemini or compatible. No document contents, identifiers,
credentials, vectors, file paths or database errors are printed. No writes,
provider calls, reindexing, schema changes or automatic repairs. A bounded
statement timeout fails safely; output is emitted only after the full scan.
Report document counts separately for documents with at least one matching
vector and at least one unknown vector. A mixed document belongs to both counts;
documents without chunks are not included. Existing vector count fields remain.
Run from the repository root with `python -m scripts.audit_embedding_fallback`.
Tests must cover exact/mismatched float32 values, empty datasets, failure privacy,
actual isolated PostgreSQL reads, and verify no state changes. The command is an
operator tool and can be slow for large corpora; the SQL timeout does not bound
local computation time. No readiness claim is implied.
