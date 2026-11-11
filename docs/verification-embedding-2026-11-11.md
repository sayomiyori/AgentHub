# Embedding suite verification — 2026-11-11

Recorded execution of the AgentHub suite at `4234a19` against a disposable
PostgreSQL/pgvector + Redis fixture. This receipt closes the "no recorded run"
gap reported in `NexusCore/docs/knowledge-embedding-review-2026-11-11.md` for
the range `11352ff..4234a19`. It covers the committed test suite only; it is not
a live-provider, deployment or reindex certification.

## Revision and worktree

| Item | Value |
| --- | --- |
| Repository | `sayomiyori/AgentHub` |
| Worktree | `C:/Users/flamx/.t3/worktrees/AgentHub/agent-embedding-verify-b8287eb4d1ea1031` |
| Branch | `agent/embedding-verify` |
| HEAD | `4234a19ad799c5274bd330312e99e32a683f30d5` |
| Subject | `fix(rag): reject structurally incomplete tenant indexes` |
| `origin/main` at run time | `4234a19ad799c5274bd330312e99e32a683f30d5` |
| Worktree status before run | clean (`git status --porcelain` empty) |
| Range under test | `11352ff..4234a19`, 14 commits |

No application, migration or test file was changed. The only repository change
from this session is this document.

## Environment

- Windows host, PowerShell 7.
- Docker Engine/CLI `29.8.2`, Docker Compose `5.6.0`.
- Python `3.12.10` in a fresh worktree-local `.venv`.
- Packages installed from the unpinned `requirements.txt` on 2026-11-11:
  `pytest 9.2.0`, `pytest-mock 3.17.0`, `SQLAlchemy 2.1.8`, `psycopg 3.3.8`,
  `pgvector 0.5.1`, `redis 8.1.0`, `google-genai 3.2.0`, `alembic 1.20.1`,
  `fastapi 0.162.4`, `pypdf 7.0.1`, `mcp 1.30.0`.
- Images (digests observed on the host):
  - `pgvector/pgvector:pg16` →
    `pgvector/pgvector@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b`;
    server PostgreSQL `16.15`, `vector` extension `0.8.6`.
  - `redis:7-alpine` →
    `redis@sha256:8b81dd37ff027bec4e516d41acfbe9fe2460070dc6d4a4570a2ac5b9d59df065`;
    server Redis `7.4.8`.

## Fixture

The fixture follows the repository's `docker-compose.verification.yml` and
`docs/verification-checkpoint.md`: loopback-only ports, `verification` database
credentials and the documented PostgreSQL/pgvector and Redis images. Only
`postgres` and `redis` were started; the `api` and `worker` services were not
built or started, so no provider or Telegram traffic was possible.

| Resource | Value |
| --- | --- |
| Compose project | `nexus-embedding-verify` |
| Containers | `nexus-embedding-verify-postgres-1`, `nexus-embedding-verify-redis-1` |
| Loopback ports | `127.0.0.1:55434` → PostgreSQL `5432`; `127.0.0.1:56381` → Redis `6379` |
| Network | `nexus-embedding-verify_default` (bridge, `10.255.255.0/24`) |
| Named volumes | none used by `postgres`/`redis`; the only declared volume (`verification-uploads`) belongs to the unstarted `api`/`worker` |
| Generic test DB | `agenthub_verification` (`AGENTHUB_TEST_DATABASE_URL`) |
| Platform test DB | `nexus_agent_ai_embedding_verify_test`, migrated to head (`AGENTHUB_PLATFORM_TEST_DATABASE_URL`) |
| Redis database | `15` (`AGENTHUB_TEST_REDIS_URL`) |

The Docker daemon refused Compose's implicit default network with
`all predefined address pools have been fully subnetted` (the host has many
retained Docker networks). The network was pinned to an unused `/24` through a
temporary, uncommitted override. This changes only address allocation; the
fixture remains an isolated bridge network with no connection to the existing
NexusCore or retained AgentHub databases.

```yaml
services: {}
networks:
  default:
    name: nexus-embedding-verify_default
    driver: bridge
    ipam:
      config:
        - subnet: 10.255.255.0/24
```

## Commands executed

Run from the worktree root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt

docker compose -p nexus-embedding-verify -f docker-compose.verification.yml `
  -f <temporary-network-override.yml> up -d --wait --wait-timeout 120 postgres redis

docker compose -p nexus-embedding-verify -f docker-compose.verification.yml exec -T postgres `
  psql -U verification -d agenthub_verification -c 'CREATE EXTENSION IF NOT EXISTS vector'
docker compose -p nexus-embedding-verify -f docker-compose.verification.yml exec -T postgres `
  psql -U verification -d postgres -c 'CREATE DATABASE "nexus_agent_ai_embedding_verify_test"'
docker compose -p nexus-embedding-verify -f docker-compose.verification.yml exec -T postgres `
  psql -U verification -d nexus_agent_ai_embedding_verify_test -c 'CREATE EXTENSION IF NOT EXISTS vector'

$env:DATABASE_URL='postgresql+psycopg://verification:verification@127.0.0.1:55434/nexus_agent_ai_embedding_verify_test?connect_timeout=5'
.\.venv\Scripts\python.exe -m alembic upgrade head

$env:AGENTHUB_TEST_DATABASE_URL='postgresql+psycopg://verification:verification@127.0.0.1:55434/agenthub_verification?connect_timeout=5'
$env:AGENTHUB_PLATFORM_TEST_DATABASE_URL='postgresql+psycopg://verification:verification@127.0.0.1:55434/nexus_agent_ai_embedding_verify_test?connect_timeout=5'
$env:AGENTHUB_TEST_REDIS_URL='redis://127.0.0.1:56381/15'
$env:AGENTHUB_CREATE_ISOLATED_TEST_DATABASES='yes'
$env:DATABASE_URL=$env:AGENTHUB_PLATFORM_TEST_DATABASE_URL
$env:GEMINI_API_KEY=''; $env:OPENAI_API_KEY=''; $env:ANTHROPIC_API_KEY=''; $env:GROQ_API_KEY=''
$env:MCP_SERVERS=''; $env:MCP_SERVERS_CONFIG=''

.\.venv\Scripts\python.exe -m pytest -q tests/
```

`alembic current` after the upgrade reported `004_merge_ai_resources (head)`;
`alembic heads` reported the single head `004_merge_ai_resources`. The platform
database was empty before the pytest run.

## Result

```text
431 passed in 40.61s
pytest exit code: 0
```

- `pytest --collect-only -q tests/` collected **431 tests** in 22 files.
- **0 failed, 0 skipped, 0 errors, 0 xfail/xpass.** No warnings summary was
  emitted.
- The integration tests were not skipped: the PostgreSQL/Redis environment
  variables were present, so `test_infrastructure.py`, `test_tenant_resources.py`,
  `test_tenant_migration.py`, `test_platform_postgres.py` and the PostgreSQL
  case in `test_fallback_inventory.py` executed.

Per-file collected counts:

| Tests | File |
| ---: | --- |
| 12 | `tests/test_agent.py` |
| 35 | `tests/test_ai_settings.py` |
| 7 | `tests/test_document_embedding_budget.py` |
| 4 | `tests/test_document_snapshot.py` |
| 4 | `tests/test_document_uncertain_outcomes.py` |
| 18 | `tests/test_embedding_provider_failure.py` |
| 5 | `tests/test_embedding_sdk_transport.py` |
| 4 | `tests/test_fallback_inventory.py` |
| 17 | `tests/test_infrastructure.py` |
| 24 | `tests/test_platform_configuration.py` |
| 34 | `tests/test_platform_generation.py` |
| 10 | `tests/test_platform_postgres.py` |
| 49 | `tests/test_platform_reads.py` |
| 24 | `tests/test_platform_schema.py` |
| 22 | `tests/test_semantic_cache_resilience.py` |
| 35 | `tests/test_telegram_admission.py` |
| 41 | `tests/test_telegram_reply_publication.py` |
| 23 | `tests/test_telegram_worker.py` |
| 4 | `tests/test_tenant_embedding_boundary.py` |
| 3 | `tests/test_tenant_migration.py` |
| 47 | `tests/test_tenant_resources.py` |
| 9 | `tests/test_unified_release_guard.py` |

## What this run covers

All suites listed as implemented in the NexusCore review executed and passed:

| Spec | Test coverage in this run |
| --- | --- |
| `embedding-provider-failure.md` | `test_embedding_provider_failure.py` (18), infrastructure worker cases |
| `tenant-embedding-boundary.md` | `test_tenant_embedding_boundary.py` (4), tenant/infrastructure cases |
| `embedding-request-bounds.md` | `test_embedding_sdk_transport.py` (5) |
| `chunk-provenance.md` | `test_document_snapshot.py` (4), infrastructure snapshot cases |
| `document-source-change.md` | infrastructure source-change cases |
| `document-embedding-budget.md` | `test_document_embedding_budget.py` (7), infrastructure budget cases |
| `document-retry-replacement.md` | infrastructure retry-replacement cases |
| `tenant-corpus-model-check.md` | `test_tenant_resources.py` |
| `tenant-corpus-readiness.md` | `test_tenant_resources.py` |
| `tenant-corpus-completeness.md` | `test_tenant_resources.py` |
| `fallback-vector-inventory.md` | `test_fallback_inventory.py` (4) |
| `unified-release.md` | `test_unified_release_guard.py`, `test_tenant_migration.py`, `test_platform_schema.py` |

The observed result supports the statement that the committed tests for the
range pass against the documented disposable fixture. It does not promote the
draft/planned documents below to implemented work.

## Not covered by this receipt

- **Live providers.** No Gemini, Groq, OpenAI or Anthropic request was made;
  all provider keys were empty. Provider behavior is verified at mocked or
  local-SDK-transport boundaries only.
- **Built services.** The API/worker images were not built, no Celery worker
  ran and `tests/http_smoke.py` was not executed.
- **Real data and production migrations.** Migration tests used synthetic,
  newly created databases on the disposable fixture. No existing database was
  upgraded, downgraded or inspected.
- **Planned work.** `knowledge-index-recovery.md` and
  `knowledge-generation-design.md` remain design-only; no code or migration
  exists for them.
- **Generation fence and concurrency semantics.** The receipt covers the
  per-query receipt/readiness/completeness guards, not a tenant-wide generation
  activation fence, concurrent corpus/cache snapshots, reindexing or vector
  quality.
- **Auth and tenant authorization.** The suite exercises synthetic tenant
  fixtures, not the open AH-01 authorization gap recorded in
  `docs/verification-checkpoint.md`.
- **Scale.** The 10 000-chunk/document caps are exercised by tests, not by load
  or performance measurement.
- **Static checks.** Ruff, Mypy, coverage and dependency audits were not run as
  part of this receipt.

**Database-state limitation.** The full suite is green on a freshly migrated,
empty platform database. `test_fallback_inventory_on_isolated_postgres` asserts
an empty chunk corpus at the start and then commits five synthetic chunks, so a
second full-suite run against the same database is not expected to pass. CI and
this run create the database first. Re-running this receipt requires recreating
the platform database (or using a new name), not cleaning data in place.

## Cleanup

Teardown was limited to this session's resources:

```powershell
docker compose -p nexus-embedding-verify -f docker-compose.verification.yml down --remove-orphans
```

Both containers and the `nexus-embedding-verify_default` network were removed.
No `docker prune`, no volume deletion and no operation against existing
NexusCore, retained AgentHub or other project resources was performed. After
teardown, filters for `nexus-embedding-verify-` containers, networks and volumes
returned nothing.

## Verdict

**Verified for the recorded environment:** the AgentHub suite at
`4234a19ad799c5274bd330312e99e32a683f30d5` passes as `431 passed` against a
fresh disposable PostgreSQL/pgvector + Redis fixture, with no skips or errors.
The residual gaps above are unchanged and remain outside this receipt.
