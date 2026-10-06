# AgentHub

[![CI](https://github.com/sayomiyori/AgentHub/actions/workflows/ci.yml/badge.svg)](https://github.com/sayomiyori/AgentHub/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)](#)
[![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)](#)
[![Celery](https://img.shields.io/badge/Celery-37814A?logo=celery&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](#)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)](#)
[![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?logo=google&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI platform with **RAG pipeline** (pgvector + semantic search + reranking), **AI agents** with function calling, **MCP server** (SSE), **multi-provider LLM routing** (Gemini / Anthropic / OpenAI), **semantic cache**, **cost tracking**, and **Prometheus** metrics.

## Architecture

```
┌──────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Client   │────▶│ FastAPI Gateway  │────▶│ Agent Orchestrator│
│ REST API │     │ /api/v1/*        │     │ tool use + RAG    │
└──────────┘     │ /mcp/sse         │     └────┬────────┬─────┘
                 │ /metrics         │          │        │
                 └──────────────────┘          ▼        ▼
                              ┌─────────────────┐  ┌──────────────────┐
                              │ RAG Pipeline    │  │ Tool Executor    │
                              │ pgvector search │  │ MCP client       │
                              │ chunk + embed   │  │ KB / calc / web  │
                              │ rerank          │  │ datetime         │
                              └────────┬────────┘  └──────────────────┘
                                       │
                              ┌────────┴────────┐
                              │ PostgreSQL      │
                              │ + pgvector      │
                              │ + llm_usage     │
                              └─────────────────┘

  ┌─────────┐  ┌──────────┐  ┌────────────┐
  │ Redis   │  │ Celery   │  │ Prometheus │
  │ semantic│  │ embed    │  │ + Grafana  │
  │ cache   │  │ worker   │  │            │
  └─────────┘  └──────────┘  └────────────┘
```

## Screenshots

### Swagger UI
![Swagger UI](docs/images/swagger-ui.png)

### RAG Query — answer with citations
![RAG Query Result](docs/images/rag-query-result.png)

### Agent — knowledge base search tool
![Agent RAG](docs/images/agent-rag-query.png)

### Agent — calculator tool
![Agent Calculator](docs/images/agent-calculator.png)

### Conversation History
![Conversation History](docs/images/conversation-history.png)

### Documents List
![Documents List](docs/images/documents-list.png)

### Grafana Dashboard
![Grafana Dashboard](docs/images/grafana-dashboard.png)

### Prometheus Metrics
![Prometheus Metrics](docs/images/prometheus-metrics.png)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (asyncio) |
| Database | PostgreSQL 16 + pgvector extension |
| ORM | SQLAlchemy 2 (async) + Alembic |
| Embeddings | Gemini `gemini-embedding-001` (3072d) |
| LLM | Gemini (default), Anthropic, OpenAI — multi-provider |
| Agent | JSON tool protocol: KB search, web search, calculator, datetime, MCP tools |
| MCP | SSE server + external MCP client |
| Async tasks | Celery + Redis broker (document embedding) |
| Cache | Redis semantic cache (LSH + cosine ≥ 0.95, TTL 24h) |
| Cost tracking | `llm_usage_records` table + `/api/v1/usage/stats` |
| Metrics | Prometheus + Grafana |
| CI | GitHub Actions (ruff + mypy + pytest + Docker build) |

## Architecture Decisions

**pgvector over dedicated vector DB (Pinecone, Weaviate)** — keeps the entire data layer in a single PostgreSQL instance. No extra infrastructure, simpler backups, transactional consistency between document metadata and embeddings. IVFFlat index handles the expected scale.

**Gemini as default provider** — free tier for embeddings + LLM, sufficient for development and demo. Multi-provider factory (`LLMFactory`) allows switching to Anthropic or OpenAI per-request without code changes.

**Semantic cache in Redis (LSH buckets)** — near-identical questions return cached answers without burning tokens. Cosine similarity ≥ 0.95 threshold balances hit rate vs answer relevance. TTL 24h prevents stale answers.

**Celery for embedding, not in-request** — embedding a large document blocks the API for seconds. Celery worker processes documents asynchronously; the client polls `GET /documents/{id}` for status.

**MCP over custom tool protocol** — Model Context Protocol is an emerging standard. Implementing it means external agents (Claude Desktop, Cursor) can use AgentHub's knowledge base as a tool — not just our own agent.

## Quick Start

```bash
cp .env.example .env
# Set GEMINI_API_KEY in .env

docker compose up --build -d
```

| Service | URL |
|---------|-----|
| API | `http://localhost:8014` |
| Health | `http://localhost:8014/health` |
| Swagger | `http://localhost:8014/docs` |
| Prometheus | `http://localhost:59090` |
| Grafana | `http://localhost:3005` (admin / admin) |

## API

### Documents

#### `POST /api/v1/documents`

Upload a document (txt, md, pdf). Triggers async Celery embedding.

```bash
curl.exe -s -X POST "http://localhost:8014/api/v1/documents" -F "file=@document.txt"
```

Response: `{"document_id": "...", "status": "processing"}`

#### `GET /api/v1/documents`

List all documents with statuses (`pending` / `processing` / `ready` / `failed`).

#### `GET /api/v1/documents/{id}`

Document metadata + chunk count.

#### `DELETE /api/v1/documents/{id}`

Delete document + cascade chunks.

---

### Query

#### `POST /api/v1/query`

Ask a question. Supports RAG and agent mode.

```json
{
  "question": "What is FastAPI built on?",
  "use_agent": false,
  "top_k": 5,
  "provider": "gemini",
  "model": "models/gemini-2.0-flash"
}
```

Response:
```json
{
  "answer": "FastAPI is built on Starlette for web handling and Pydantic for data validation...",
  "sources": [
    {"document_title": "docs.txt", "chunk_text_preview": "FastAPI is built on...", "score": 0.92}
  ],
  "tokens_used": 450,
  "cost_usd": 0.0003,
  "conversation_id": "uuid..."
}
```

| Parameter | Description |
|-----------|-------------|
| `question` | Required. The question to ask |
| `use_agent` | `false` = direct RAG, `true` = agent with tools |
| `conversation_id` | Continue existing conversation |
| `top_k` | Number of chunks to retrieve (default: 5) |
| `provider` | `gemini` / `anthropic` / `openai` |
| `model` | Model name (e.g. `models/gemini-2.0-flash`) |

---

### Conversations

#### `GET /api/v1/conversations`

List all conversations.

#### `GET /api/v1/conversations/{id}/messages`

Message history for a conversation.

---

### Cost Tracking

#### `GET /api/v1/usage/stats`

Aggregated LLM cost and token usage.

```json
{
  "total_cost_usd": 0.0123,
  "total_tokens": 45000,
  "cost_by_provider": {"gemini": 0.01},
  "cost_by_model": {"models/gemini-2.5-flash": 0.01},
  "cost_by_day": [{"day": "2026-03-27", "cost_usd": 0.0123}]
}
```

---

### Metrics

#### `GET /metrics`

Prometheus text exposition.

| Metric | Type | Description |
|--------|------|-------------|
| `llm_requests_total` | Counter | LLM calls; labels: `provider`, `model`, `status` |
| `llm_tokens_used_total` | Counter | Tokens; labels: `provider`, `model`, `direction` |
| `llm_cost_usd_total` | Counter | Cost in USD; labels: `provider`, `model` |
| `llm_request_duration_seconds` | Histogram | LLM request latency |
| `rag_retrieval_duration_seconds` | Histogram | Vector search latency |
| `embedding_duration_seconds` | Histogram | Embedding generation latency |
| `documents_total` | Gauge | Total documents |
| `chunks_total` | Gauge | Total chunks |
| `semantic_cache_hit_ratio` | Gauge | Cache hit ratio |
| `mcp_tool_calls_total` | Counter | MCP tool invocations |

## MCP

### Local MCP Server

SSE endpoint: `GET http://localhost:8014/mcp/sse`

Tools: `search_documents`, `list_documents`. Resource template: `document://{document_id}`.

### External MCP Servers

```env
MCP_SERVERS=[{"name":"local-docs","url":"http://127.0.0.1:8014/mcp/sse"}]
```

On startup, AgentHub connects via SSE and merges external tools into the agent (prefixed names like `agenthub__search_documents`).

## Running Tests

```bash
pip install -r requirements.txt

# Lint + type check
ruff check .
python -m mypy app/models/llm_usage.py app/services/usage_tracker.py app/metrics.py

# Tests
pytest tests/
```

CI: GitHub Actions — ruff, mypy (subset), pytest, Docker build, PostgreSQL + Redis services.

## Environment Variables

### Free demo providers

Direct API probes on 2026-10-06 returned complete text from Gemini
`gemini-2.5-flash`, OpenRouter `liquid/lfm-2.5-2.6b:free` and Cloudflare Workers AI
`@cf/meta/llama-3.2-3b-instruct`. The operator confirmed Gemini Free Tier and
Workers Free; OpenRouter reported Free Tier and zero cost for the probe.
These checks do not verify the Telegram flow or account billing history.

The working checkout includes Groq integration, selected by NexusCore Compose.
Gemini already has an adapter. OpenRouter and Cloudflare adapters are planned:
adding their `.env` variables does not enable them. OpenRouter uses
`OPENROUTER_TOKEN`; Cloudflare requires `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID`. Root Compose currently forwards only Groq credentials;
standalone Gemini reads this repository's `GEMINI_API_KEY`.

Use free accounts and leave `LLM_FALLBACK_PROVIDER`/`LLM_FALLBACK_MODEL` empty.
The existing standalone factory supports paid providers and fallback; it does
not enforce a free-only policy. The planned platform worker will enforce its
own provider policy. OpenAI's SDK in the Groq adapter connects to Groq's endpoint
and does not require an OpenAI account.

[OpenRouter Free](https://openrouter.ai/pricing) allows 50 requests/day; choose
catalog models with `:free` and verified zero pricing.
[Workers Free](https://developers.cloudflare.com/workers-ai/platform/pricing/)
includes 10,000 Neurons/day; some models require paid access.
[Gemini quotas](https://ai.google.dev/gemini-api/docs/rate-limits) depend on project
and model. Quota exhaustion is an error, not permission to use paid inference.
Token-price estimates are not evidence of actual billed cost.

Future UI token onboarding is a separate feature: encrypted tenant-scoped
credentials, masked metadata and no token readback. Keep real keys in local
`.env` or secret storage; examples contain empty values only.

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini LLM + embeddings |
| `DATABASE_URL` | PostgreSQL connection |
| `REDIS_URL` | Celery broker + semantic cache |
| `LLM_PROVIDER` | Default provider (`gemini`) |
| `LLM_MODEL` | Default model |
| `EMBEDDING_MODEL` | Embedding model |
| `SEMANTIC_CACHE_ENABLED` | `true` / `false` |
| `MCP_SERVERS` | JSON list of `{name, url}` |
| `LLM_FALLBACK_PROVIDER` | Fallback if primary fails |
| `ANTHROPIC_API_KEY` | Optional: Anthropic provider |
| `OPENAI_API_KEY` | Optional: OpenAI provider |
| `GROQ_API_KEY` | Working-checkout Groq adapter; NexusCore demo provider |
| `OPENROUTER_TOKEN` | Reserved: upcoming OpenRouter adapter |
| `CLOUDFLARE_API_TOKEN` | Reserved: upcoming Workers AI adapter |
| `CLOUDFLARE_ACCOUNT_ID` | Reserved: Workers AI account identifier |

## Project Structure

```
agenthub/
├── app/
│   ├── api/v1/          # REST endpoints
│   ├── models/          # SQLAlchemy models (document, chunk, conversation, usage)
│   ├── services/
│   │   ├── rag/         # chunker, embedder, retriever, reranker, generator
│   │   ├── agent/       # orchestrator, tools (KB, calc, web, datetime, MCP)
│   │   └── llm/         # multi-provider factory (Gemini, Anthropic, OpenAI, Groq)
│   ├── mcp/             # MCP server + client
│   ├── cache/           # semantic cache (Redis + LSH)
│   └── metrics.py       # Prometheus metrics
├── monitoring/          # Prometheus + Grafana configs
├── tests/
├── docs/images/         # Screenshots
├── docker-compose.yml
├── Dockerfile
└── alembic.ini
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Telegram platform migration checkpoint

Platform job, usage and reply-outbox models use separate metadata from legacy
standalone records. Processing and Telegram delivery are not implemented yet.
`TELEGRAM_AI_ENABLED=false` keeps the standalone startup behavior. Enabling it
requires three independent service keys, an explicit Groq model, a fixed WebHook
origin and migration head `002_telegram_ai`; startup never stamps the database.

For a new isolated PostgreSQL database, configure `DATABASE_URL` and run:

```powershell
python -m alembic upgrade head
python -m alembic check
```

The baseline creates five standalone tables and the vector extension; the second
revision adds three platform tables. Do not apply the baseline blindly to an
existing `create_all` database or automatically stamp it. Compare its actual
schema with the frozen baseline and plan explicit operator-approved adoption.
A rollback to `001_legacy_baseline` removes the three platform tables and requires
confirmation before executing DROP; test only in an empty isolated database.

## Signed Telegram admission checkpoint

`POST /internal/v1/telegram/updates` is available only with
`TELEGRAM_AI_ENABLED=true` and the migrated schema. It verifies exactly one
`X-Webhook-Signature: sha256=<hex>` over the raw body, caps streamed input at
1 MiB, validates the v1 envelope and resolves fresh active bot/tenant context
through WebHook Manager. No provider call runs in this endpoint.

`WEBHOOK_AGENT_SERVICE_KEY` in AgentHub must match WebHook Manager's
`WEBHOOK_AGENT_CONTEXT_KEY`; the independent `WEBHOOK_AGENT_INGRESS_KEY` signs
updates. The fixed `WEBHOOK_INTERNAL_URL` origin cannot contain credentials,
a path, query or fragment. Context lookup has an 8-second total deadline,
5-second HTTP timeout and a 64 KiB response cap, with no redirects or proxy
inheritance. Error responses do not echo event data or remote error bodies.

First durable admission returns 202, identical replay 200, content/identity
conflict 409. Receipts contain exactly `event_id`, `job_id`, `state`.
PostgreSQL uniqueness prevents concurrent duplicate jobs; commit precedes the
best-effort UUID notification to the dedicated `telegram_ai` queue. Broker outage
still returns the durable receipt and leaves the job pending. Failed notification
logs only a static warning. Replays always reauthorize current bot context.

The processing worker and recovery scanner remain subsequent stages. The queue
notification names `platform.process_telegram_job`; do not attach the legacy
embedding worker to this queue. Pending admission is not an AI result or Telegram
delivery. Existing standalone routes and defaults retain their behavior.

## Bounded platform generation

The platform generation function calls Groq directly through the existing
provider interface; it never enters the standalone factory's fallback path.
It sends the versioned `telegram-plain-v1` system prompt and the admitted
question only. No tools, RAG, history or cache are consulted. The fixed endpoint
is `https://api.groq.com/openai/v1`; proxy inheritance and redirects are disabled.
One request has a 20-second total deadline, SDK retries disabled and
`max_completion_tokens=1024`. Empty, refused, tool-bearing or incomplete
responses and invalid token usage are rejected. Stored text is capped at 4096
characters with an ellipsis; remote errors never appear in platform errors.

Usage cost is an approximate list-price estimate, not the actual charge to the
operator's free account. Current GPT-OSS 20B and Qwen3.8 estimates use the
[Groq model catalog](https://console.groq.com/docs/models); older model entries
retain historical estimates and do not establish current free-account access.
The platform requires an explicit model and `GROQ_API_KEY`; model/account access
must be verified separately before a live demo. No billing changes are made.
Durable processing and delivery remain subsequent stages.
