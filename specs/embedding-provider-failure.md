# Embedding failure in configured-provider mode

When Gemini is configured, provider errors or invalid vectors must not silently
become deterministic fallback vectors. A failure in any batch item aborts that
batch; the existing worker rolls back and records failed indexing. A direct RAG
query returns a fixed HTTP 503 without provider details. If the agent knowledge
tool fails after a model call, it stops and returns a fixed unavailable answer
through the existing successful response path so incurred usage is committed;
it does not continue generation or search using fallback vectors. Preserve the
existing no-key offline fallback for standalone compatibility; it is not a
semantic-quality claim. Existing data is neither identified nor reindexed.

Validate vectors as exactly 1536 finite float32-range numeric non-boolean
elements, with at least one nonzero value. Propagate
a fixed exception without chaining private provider errors. No credential or
provider text enters errors. No retries, schema/config/dependency changes, live
provider calls or root deployment. The existing query path may have already
created an empty conversation before a provider failure; this slice does not
change that legacy transaction boundary.

Tests: no-key fallback unchanged; configured provider happy, error, empty,
malformed, invalid dimensions/non-finite and later-batch failure; direct query
safe 503 and agent fixed unavailable answer; worker failure leaves no chunks and records failed in an
isolated PostgreSQL fixture where available. Run full available tests, Ruff,
CI Mypy scope and independent security/correctness review. Residual: existing
fallback and provider vectors have no provenance and remain unsafe to mix;
model changes and missing-key offline mode need a separate design before rollout.
