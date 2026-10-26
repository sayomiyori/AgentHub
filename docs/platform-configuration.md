# Platform AI configuration read

`GET /api/v1/tenants/{tenant_id}/bots/{bot_id}/ai/configuration` is available
only with the existing `PLATFORM_READ_ENABLED=true` feature. Every request uses
the existing fresh AuthFortress `ai.read` authorization and active WebHook
bot/tenant context checks. Owner and member may read it. Reads neither use a
database session nor contact a provider or execute jobs.

The response contains only `tenant_id`, `bot_id`, `scope: "platform"`,
`provider: "groq"`, `model` (null if unset) and `processing_enabled`. These are
the cached operator-managed `TELEGRAM_AI_*` process settings, shared across
authorized bots. This is not a bot-specific saved override, standalone RAG
configuration, credential-readiness or provider-connectivity check. An enabled
processing flag does not establish valid credentials or a running worker.

Model identifiers must match `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}` or be empty.
Malformed configured identifiers return fixed 503 without their value. Existing
disabled/authentication/authorization errors remain 404/401/404/503. No keys,
key-presence booleans, service URLs, prompts or full settings dump are returned.
No new environment variables, dependency, migration or deployment change.

Verification (from repository root, with its Python environment):

```powershell
python -m pytest tests/test_platform_configuration.py -q
python -m pytest tests/test_platform_configuration.py tests/test_platform_schema.py -q
python -m ruff check .
python -m mypy app/platform app/api/v1/platform_reads.py
```

The 24 new tests use real issuer/context clients with controlled HTTP responses;
they do not require database access. They cover owner/member, unset/selected/
invalid models, processing flag as metadata, disabled mode, missing/duplicate/
revoked identity, foreign/inactive context, outages and safe projection. The
database dependency raises if accidentally used. Existing DB integration tests
still require a separate isolated migrated PostgreSQL as documented in README.
