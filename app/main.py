from fastapi import FastAPI

from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router
from app.config import get_settings
from app.db.session import Base, engine
from app.models.chunk import Chunk  # noqa: F401
from app.models.conversation import Conversation, Message  # noqa: F401
from app.models.document import Document  # noqa: F401

settings = get_settings()
app = FastAPI(title=settings.app_name, debug=settings.debug)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
