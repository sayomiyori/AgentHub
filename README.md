# AgentHub

AI platform with **RAG** (PostgreSQL + **pgvector**), **Celery** workers, **Gemini** embeddings and LLM, optional **agents** with tools, **MCP** server (SSE) and **multi-provider** LLM routing, plus **cost tracking** and **Prometheus** metrics.

## Architecture

```
                    +------------------+
                    |   FastAPI :8014  |
                    |  /api/v1/*       |
                    |  /mcp (SSE)      |
                    |  /metrics        |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
 +---------------+   +---------------+   +---------------+
 |  PostgreSQL   |   |    Redis      |   | MCP clients   |
 |  + pgvector   |   | semantic cache|   | (optional ext)|
 |  + llm_usage  |   | Celery broker |   +---------------+
 +---------------+   +---------------+
         ^
         |
 +---------------+
 | Celery worker |
 | (embeddings)  |
 +---------------+
```

Data flow (direct RAG query): **upload** → chunk → **Gemini** embed → store vectors → **query** → embed question → **semantic cache** (Redis) or **vector search** → rerank → **LLM** (Gemini / Anthropic / OpenAI) → answer with citations.

## Features

| Area | Description |
|------|-------------|
| **RAG** | Chunking, Gemini `embedding-001`, cosine retrieval, rerank, cited answers |
| **Agents** | JSON tool protocol: knowledge base, web search, calculator, datetime, **MCP tools** |
| **MCP** | SSE server at `/mcp/sse`; optional external MCP servers via `MCP_SERVERS` |
| **Multi-provider** | `LLMFactory`: Gemini (default), Anthropic, OpenAI; per-request `provider` / `model` |
| **Cost control** | `llm_usage_records` table; `GET /api/v1/usage/stats` |
| **Semantic cache** | Redis, LSH bucket + cosine ≥ 0.95, TTL 24h |
| **Observability** | Prometheus `/metrics` |

## Quick start

```bash
cp .env.example .env   # set GEMINI_API_KEY
docker compose up --build -d
```

App: `http://localhost:8014` · Health: `GET http://localhost:8014/health`

### Upload a document

```bash
curl.exe -s -X POST "http://localhost:8014/api/v1/documents" ^
  -F "file=@C:\path\to\file.txt"
```

### Query (RAG, Gemini)

```bash
curl.exe -s -X POST "http://localhost:8014/api/v1/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is this document about?\",\"use_agent\":false,\"provider\":\"gemini\",\"model\":\"models/gemini-2.5-flash\"}"
```

**PowerShell** often breaks JSON encoding; prefer **`curl.exe`** (ships with Windows 10+):

```powershell
curl.exe -s -X POST "http://localhost:8014/api/v1/query" -H "Content-Type: application/json" --data-raw "{\"question\":\"What is FastAPI?\",\"use_agent\":false}"
```

Or with JSON in single quotes (no escaping of inner double quotes):

```powershell
curl.exe -s -X POST "http://localhost:8014/api/v1/query" -H "Content-Type: application/json" --data-raw '{"question":"What is FastAPI?","use_agent":false}'
```

If you use `Invoke-RestMethod`, define `$body` first, then call it in **one** command (parameters are not valid on their own lines):

```powershell
$body = @{ question = "What is FastAPI?"; use_agent = $false } | ConvertTo-Json -Compress -Depth 5
Invoke-RestMethod -Method POST -Uri "http://localhost:8014/api/v1/query" -ContentType "application/json" -Body $body
```

If `Invoke-RestMethod` reports **500**, `$resp1` is **not** updated (the number you print may be from an earlier run). Check `docker compose logs app --tail=80` for the traceback.

### Query (agent)

```bash
curl.exe -s -X POST "http://localhost:8014/api/v1/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Summarize the docs\",\"use_agent\":true}"
```

### Usage stats (cost tracking)

```bash
curl.exe -s "http://localhost:8014/api/v1/usage/stats"
```

Example response:

```json
{
  "total_cost_usd": 0.0123,
  "total_tokens": 45000,
  "cost_by_provider": {"gemini": 0.01},
  "cost_by_model": {"models/gemini-2.5-flash": 0.01},
  "cost_by_day": [{"day": "2026-03-27", "cost_usd": 0.0123}]
}
```

### Prometheus metrics

```bash
curl.exe -s "http://localhost:8014/metrics"
```

Relevant series: `llm_requests_total`, `llm_tokens_used_total`, `llm_cost_usd_total`, `semantic_cache_hit_ratio`, `rag_retrieval_duration_seconds`, `mcp_tool_calls_total`, …

## API reference (curl)

| Method | Path | Description |
|--------|------|-------------|
| **GET** | `/health` | Liveness |
| **GET** | `/metrics` | Prometheus text exposition |
| **POST** | `/api/v1/documents` | Multipart upload (`file`) |
| **GET** | `/api/v1/documents` | List documents |
| **GET** | `/api/v1/documents/{id}` | Document metadata |
| **DELETE** | `/api/v1/documents/{id}` | Delete document |
| **POST** | `/api/v1/query` | JSON: `question`, optional `conversation_id`, `top_k`, `use_agent`, `provider`, `model` |
| **GET** | `/api/v1/conversations` | List conversations |
| **GET** | `/api/v1/conversations/{id}/messages` | Messages |
| **GET** | `/api/v1/usage/stats` | Aggregated LLM cost and tokens |

## MCP

### Local MCP server (this app)

- SSE: `GET http://localhost:8014/mcp/sse`
- Messages: `POST http://localhost:8014/mcp/messages/?session_id=...`

Tools: `search_documents`, `list_documents`. Resource template: `document://{document_id}`.

### External MCP servers

Set env (JSON array):

```env
MCP_SERVERS=[{"name":"local-docs","url":"http://127.0.0.1:8014/mcp/sse"}]
```

Or YAML file path in `MCP_SERVERS_CONFIG`. On startup, AgentHub connects over SSE and merges tools into the agent (prefixed names like `agenthub__search_documents`).

## Screenshots

Place images under `docs/images/` (examples):

- `docs/images/agent-rag-query.png` — query with citations
- `docs/images/agent-calculator.png` — agent tool use
- `docs/images/conversation-history.png` — threaded chat

Add a Grafana dashboard panel for `/metrics` to visualize `semantic_cache_hit_ratio` and `llm_cost_usd_total`.

## Environment

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini LLM + embeddings |
| `DATABASE_URL` | PostgreSQL |
| `REDIS_URL` | Celery + semantic cache |
| `LLM_PROVIDER` | Default: `gemini` |
| `LLM_MODEL` | e.g. `models/gemini-2.5-flash` |
| `EMBEDDING_MODEL` | e.g. `models/embedding-001` |
| `SEMANTIC_CACHE_ENABLED` | `true` / `false` |
| `MCP_SERVERS` | JSON list of `{name, url}` |
| `LLM_FALLBACK_PROVIDER` | Fallback if primary fails |

## Development

```bash
pip install -r requirements.txt
ruff check .
python -m mypy app/models/llm_usage.py app/services/usage_tracker.py app/metrics.py app/cache/semantic_cache.py
pytest tests/
```

CI (GitHub Actions): Ruff, Mypy (subset), pytest, Docker build, PostgreSQL + Redis services.
