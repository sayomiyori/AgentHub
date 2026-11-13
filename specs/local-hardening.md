# Local tenant isolation and cache consistency

Scope: close isolation gaps in existing documents, RAG/agent queries,
conversations, usage and semantic cache. No external provider calls, deployment,
new providers, Telegram activation or RAG quality claims.

## Contract

- `PLATFORM_TENANT_ENABLED=false` preserves standalone behavior. Enabled mode
  requires a fixed `AUTHFORTRESS_BASE_URL` and migration `003_tenant_resources`.
  Existing standalone schemas also need the forward migration before upgrading
  this release; startup rejects old columns explicitly and never migrates itself.
- Existing resource routers are also served under
  `/api/v1/tenants/{tenant_id}`. Each request forwards one bearer to AuthFortress
  with a server-selected permission. Reads/query use `ai.read`; upload/delete
  require `ai.configure`. Authentication failure is 401, foreign scope 404,
  insufficient role 403, issuer outage or malformed authorization 503.
- No positive authorization caching. Disabled scoped APIs return 404. Any
  enabled platform mode disables global resource APIs, `/metrics` and unscoped
  local MCP. Existing Prometheus scrapes receive 404 in platform mode until a
  separately authorized collector contract exists; standalone metrics retain 200.
- Documents/conversations/usage carry nullable tenant UUIDs. Existing NULL rows
  remain standalone and are invisible to scoped requests; no automatic adoption.
  Chunks/messages inherit scope through their owning document/conversation.
  Unknown/foreign conversation IDs return 404 before any write/provider call.
- All retrieval paths, agent history/knowledge tools and usage queries restrict
  the canonical tenant. Unscoped external MCP tools are unavailable to tenant
  agents. Tenant selectors in query/body never grant access.
- Cache keys include tenant and a PostgreSQL corpus fingerprint. Each successful
  document embedding replaces its immutable revision in the same transaction as
  chunks/status; deletion changes the fingerprint. In-flight old generations
  write only their old namespace, including during Redis outages.
- Uploads use random stored names, at most 10 MiB and valid UTF-8/PDF signatures;
  failed writes remove their partial file, deleted documents remove their file.
  Lists use bounded pagination and stable ID order.

## Implementation and acceptance

1. Add forward nullable columns/indexes and document revision. Rehearse upgrade,
   downgrade and upgrade on an isolated PostgreSQL DB; `alembic check` must pass.
2. Reuse existing routers/services and authorization client; test owner/member,
   foreign and NULL resources, spoofed selectors, missing/duplicate bearers,
   revocation/outage, scoped usage/history/retrieval and global/MCP closures.
3. Test cache separation and invalidation after embed/delete, concurrent old
   generation and Redis interruption without provider traffic.
4. Run full suite, Ruff, scoped Mypy, Docker build and focused security checks.

Safe defaults: resources are tenant-level; bots share their tenant knowledge.
This does not add a bot knowledge/configuration API. No active database is
migrated and no stored legacy records are assigned to a tenant. Downgrading 003
loses tenant associations/revisions and is only rehearsed on disposable data.
