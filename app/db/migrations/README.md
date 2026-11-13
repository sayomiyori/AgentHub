# AgentHub migrations

`001_legacy_baseline` is a frozen snapshot of the five standalone tables and
creates the vector extension if absent. `002_telegram_ai` adds three independent
platform tables: jobs, usage and reply outbox. `003_tenant_resources` adds nullable
tenant UUIDs to documents, conversations and standalone usage, plus an immutable
document content revision. Existing rows stay NULL; no tenant is inferred.
The Alembic environment compares
both legacy Base and PlatformBase metadata; revisions never import mutable model
metadata as their schema definition.

For a new empty isolated PostgreSQL database, set DATABASE_URL and run:

```powershell
python -m alembic upgrade head
python -m alembic check
```

Do not run the baseline against a populated database initialized by create_all.
Before adoption, compare its actual schema with the frozen baseline and obtain
operator approval for an explicit adoption procedure. Startup never stamps or
applies migrations automatically. When TELEGRAM_AI_ENABLED is true, startup
requires revision 003_tenant_resources, as do PLATFORM_READ_ENABLED and
PLATFORM_TENANT_ENABLED. A populated standalone database must also receive the
forward resource migration before running this release. Startup rejects missing
resource columns explicitly; a fresh standalone database still uses create_all.

Rollback to 001_legacy_baseline deletes only the three platform tables. Obtain
confirmation before DROP and verify in an empty isolated test database; retain
all standalone tables and data. Never downgrade a live database as a test. The
vector extension is retained on baseline rollback. Downgrading 003 to 002 removes
tenant associations and document revisions; it is destructive for scoped data
and was rehearsed only in a new disposable database with synthetic legacy rows.

Processing and Telegram delivery remain separate implementation stages.
