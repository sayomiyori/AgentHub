from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router
from app.config import get_settings
from app.db.session import Base, engine
from app.mcp.client import mcp_client_manager
from app.mcp.server import create_mcp_starlette_app
from app.models.chunk import Chunk  # noqa: F401
from app.models.conversation import Conversation, Message  # noqa: F401
from app.models.document import Document  # noqa: F401

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    await mcp_client_manager.startup()
    yield
    await mcp_client_manager.shutdown()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.mount("/mcp", create_mcp_starlette_app())

app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
