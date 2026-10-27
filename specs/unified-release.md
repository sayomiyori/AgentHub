# Unified AI settings and tenant resources

Combine the existing tenant-resource release with published AI model settings,
without editing either applied revision or adopting any existing deployment.
Both migration branches must reach one merge revision; standalone legacy
behavior and tenant isolation must remain intact. Real providers remain off.

Acceptance requires new isolated PostgreSQL databases for fresh installation
and for each 003 branch, populated with synthetic data before upgrade. Verify
row, scope, revision, constraint and index preservation, one final head, model
metadata drift, and isolated downgrade/upgrade semantics. Both incomplete heads
must fail platform startup. Verify scoped API, RAG/cache and usage isolation,
legacy endpoint closure in each platform mode, and AI settings save/inherit/job
claim/retry together. Run the service suite and isolated integrated smoke with
synthetic provider boundaries, plus independent security review.

Existing databases and migrations are untouched during acceptance. The merge
revision requires an explicit parent target for its isolated downgrade; it
removes only the merge marker. Subsequent branch downgrades can remove tenant
scope or settings data. Production adoption requires its own reviewed plan.
