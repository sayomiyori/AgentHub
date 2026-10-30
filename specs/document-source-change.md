# Detect observed source changes during indexing

The worker parses an immutable byte snapshot. Before persisting chunks as ready,
read the source again and compare its SHA-256 with that snapshot. If the file is
missing, unreadable or changed, abort the transaction, preserve existing chunks,
and follow the existing failed-document path. Provider work already performed is
not automatically retried. Do not log file content or expose paths in a new error.
No schema, public API, provider configuration or deployment change.

This detects changes observed before the final check; it does not prevent an
external change after the check or prove source immutability. The planned index
generation fence is still required. Test successful stable sources, mutation and
removal during the provider call on isolated PostgreSQL; failure must not commit
new chunks or ready state. Keep existing duplicate-task and concurrent retry
acceptance intact.
