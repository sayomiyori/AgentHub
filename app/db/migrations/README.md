# AgentHub migrations

`001_legacy_baseline` freezes the standalone schema; `002_telegram_ai` adds
the job, usage and reply tables. Independent branches `003_tenant_resources`
and `003_ai_settings` add resource scope and model settings, respectively.
`004_merge_ai_resources` combines both without modifying either parent.

Enabled platform startup requires the merged revision. A populated standalone
database needs the resource migration before using this release. New standalone
databases may use create_all. Never stamp or migrate an existing database without
a separately reviewed adoption procedure and verified backup.

The merge revision does not change data. `downgrade -1` is ambiguous at this
merge. On a new isolated test database, `downgrade 003_tenant_resources` removes
only the merge marker and exposes both 003 heads. Downgrading either parent branch can remove resource tenant IDs and
document revisions or drop the settings and audit tables. Going to 001 removes
all five platform tables. Do not downgrade an existing database as a test.
