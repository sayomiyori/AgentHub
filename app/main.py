from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.api.internal.telegram import router as telegram_router
from app.api.v1.ai_settings import router as ai_settings_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.platform_reads import router as platform_reads_router
from app.api.v1.query import router as query_router
from app.api.v1.usage import router as usage_router
from app.config import get_settings
from app.db.platform import require_platform_schema, require_resource_schema
from app.db.session import Base, SessionLocal, engine
from app.mcp.client import mcp_client_manager
from app.mcp.server import create_mcp_starlette_app
from app.metrics import chunks_total, documents_total, metrics_payload
from app.models.chunk import Chunk  # noqa: F401
from app.models.conversation import Conversation, Message  # noqa: F401
from app.models.document import Document  # noqa: F401
from app.models.llm_usage import LLMUsageRecord  # noqa: F401
from app.platform.config import get_platform_settings
from app.platform.resource_access import platform_active

settings = get_settings()


def _refresh_storage_gauges() -> None:
    db = SessionLocal()
    try:
        documents_total.set(db.query(Document).count())
        chunks_total.set(db.query(Chunk).count())
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    platform = get_platform_settings()
    if platform_active(platform):
        require_platform_schema(engine)
    else:
        require_resource_schema(engine)
        Base.metadata.create_all(bind=engine)
    _refresh_storage_gauges()
    if not platform_active(platform):
        await mcp_client_manager.startup()
    yield
    await mcp_client_manager.shutdown()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.mount("/mcp", create_mcp_starlette_app())

app.include_router(telegram_router)
app.include_router(platform_reads_router)
app.include_router(ai_settings_router)
app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")
app.include_router(usage_router, prefix="/api/v1")
for resource_router in (documents_router, query_router, conversations_router, usage_router):
    app.include_router(resource_router, prefix="/api/v1/tenants/{tenant_id}")


@app.middleware("http")
async def close_unscoped_resources(request: Request, call_next):
    if platform_active(get_platform_settings()) and (
        request.url.path in ("/metrics", "/mcp") or request.url.path.startswith("/mcp/")
        or any(request.url.path == f"/api/v1/{resource}"
               or request.url.path.startswith(f"/api/v1/{resource}/")
               for resource in ("documents", "query", "conversations", "usage"))
    ):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    _refresh_storage_gauges()
    data, ctype = metrics_payload()
    return Response(content=data, media_type=ctype)
