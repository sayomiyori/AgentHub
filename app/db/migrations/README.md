# AgentHub migrations

`001_legacy_baseline` is a frozen snapshot of the five standalone tables and
creates the vector extension if absent. `002_telegram_ai` adds three independent
platform tables: jobs, usage and reply outbox. The Alembic environment compares
both legacy Base and PlatformBase metadata; revisions never import mutable model
metadata as their schema definition.

`003_ai_settings` adds tenant/bot model settings and atomic revision audit.
It is based on the published002 line; it does not silently merge a separate
unpublished003_tenant_resources branch. See [settings boundary](../../../docs/tenant-ai-settings.md).

For a new empty isolated PostgreSQL database, set DATABASE_URL and run:

```powershell
python -m alembic upgrade head
python -m alembic check
```

Do not run the baseline against a populated database initialized by create_all.
Before adoption, compare its actual schema with the frozen baseline and obtain
operator approval for an explicit adoption procedure. Startup never stamps or
applies migrations automatically. When TELEGRAM_AI_ENABLED is true, startup
requires revision003_ai_settings (also required with PLATFORM_READ_ENABLED).
Legacy standalone startup remains available
with the platform flag false and uses only legacy metadata.

Rollback003_ai_settings to002_telegram_ai removes the two settings/audit tables.
Rollback all the way to001_legacy_baseline removes FIVE platform tables:
ai_model_settings_audit, ai_model_settings, telegram_reply_outbox,
telegram_ai_usage and telegram_ai_jobs. Obtain confirmation before DROP and
verify only in a new empty isolated test database; retain all standalone tables
and data. Never downgrade a live database as a test. The vector extension is
retained on baseline rollback.

Processing and Telegram delivery remain separate implementation stages.
