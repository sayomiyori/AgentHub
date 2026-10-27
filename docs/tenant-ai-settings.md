# Persisted tenant and bot model settings

New GET/PUT `/api/v1/tenants/{tenant}/ai/settings` and
`/api/v1/tenants/{tenant}/bots/{bot}/ai/settings` use the existing opt-in
`PLATFORM_READ_ENABLED` feature. GET requires fresh issuer ai.read; PUT requires
fresh ai.configure with an independently checked owner receipt. Bot operations
also require fresh active WebHook bot/tenant context. Members can read only.
The previous process-policy `/ai/configuration` endpoint remains compatible.

PUT accepts exactly `{"model":"model-id","expected_revision":0}`; model null
means inheritance. The JSON body is duplicate-free, bounded16KiB/5s and strict;
no flags, provider, credentials or body scope are writable. Model identifiers
match `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}`. Version0 creates; later versions use
atomic compare-and-swap. Stale/concurrent writes return409 without extra audit.
Every successful save appends its actor/revision/model receipt in the same
transaction. Reset uses a null row and a new revision; no business DELETE exists.

Responses contain safe scope IDs, target, saved model/revision/UTC time,
effective scope/provider/model and the operator processing flag. Actor UUIDs,
credentials, prompts and audit internals are excluded. Precedence is explicit
bot model > tenant model > operator platform model. A tenant default can be
edited without registering a bot. Groq remains the only platform provider.

The first job claim freezes the effective provider/model into its existing job
row. Retry keeps that frozen pair after later edits. This applies only to the
Telegram platform path, not standalone RAG/cache. Saving never enables processing,
starts a worker or contacts a provider; operator readiness checks still apply.
After an uncertain write outcome, reload the stored revision before retrying.

## Migration and deployment boundary

`003_ai_settings` adds current settings and append-only audit tables after the
published `002_telegram_ai`. `004_merge_ai_resources` joins that history with
the separate `003_tenant_resources` history without modifying either revision.
Enabled platform startup requires the merged head. Existing deployments require
separately reviewed adoption and verified backups; do not stamp an existing
database to bypass the graph. This candidate has not been deployed.

Only in a new empty owned test database:

```powershell
python -m alembic upgrade head
python -m alembic downgrade 003_tenant_resources
python -m alembic upgrade head
python -m alembic check
python -m pytest tests/ -q
python -m ruff check .
python -m mypy app/platform app/api/v1/ai_settings.py app/models/ai_settings.py
```

The full suite needs the documented separate PostgreSQL/Redis test URLs. Its
legacy test DB also needs the vector extension. A Docker test process must not
inherit CELERY_BROKER_URL overriding REDIS_URL in the broker-outage probe.
The migration branch test also requires `AGENTHUB_CREATE_ISOLATED_TEST_DATABASES=yes`;
it creates uniquely named test databases and retains them. Set it only after
verifying the PostgreSQL instance belongs to the isolated test fixture.
No production/retained database may be downgraded as verification.
