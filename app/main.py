from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response

from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router
from app.api.v1.usage import router as usage_router
from app.config import get_settings
from app.db.session import Base, SessionLocal, engine
from app.mcp.client import mcp_client_manager
from app.mcp.server import create_mcp_starlette_app
from app.metrics import chunks_total, documents_total, metrics_payload
from app.models.chunk import Chunk  # noqa: F401
from app.models.conversation import Conversation, Message  # noqa: F401
from app.models.document import Document  # noqa: F401
from app.models.llm_usage import LLMUsageRecord  # noqa: F401

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
    Base.metadata.create_all(bind=engine)
    _refresh_storage_gauges()
    await mcp_client_manager.startup()
    yield
    await mcp_client_manager.shutdown()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.mount("/mcp", create_mcp_starlette_app())

app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")
app.include_router(usage_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    _refresh_storage_gauges()
    data, ctype = metrics_payload()
    return Response(content=data, media_type=ctype)
