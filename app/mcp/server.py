from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from app.db.session import SessionLocal
from app.models.document import Document
from app.services.rag.embedder import GeminiEmbedder
from app.services.rag.reranker import SimpleReranker
from app.services.rag.retriever import PgVectorRetriever


def _read_document_text(file_path: str, content_type: str) -> str:
    path = Path(file_path)
    if content_type in {"txt", "md"}:
        return path.read_text(encoding="utf-8")
    if content_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported content type: {content_type}")


def create_mcp_starlette_app() -> Starlette:
    """MCP server with SSE transport at GET /sse (mount app at /mcp → /mcp/sse)."""
    mcp = FastMCP(
        name="agenthub",
        instructions="AgentHub knowledge base: search documents and read full document text.",
        sse_path="/sse",
        message_path="/messages/",
    )

    @mcp.tool(name="search_documents", description="Semantic search over uploaded documents (RAG).")
    def search_documents(query: str, top_k: int = 5) -> list[dict[str, str | float]]:
        db = SessionLocal()
        try:
            embedder = GeminiEmbedder()
            retriever = PgVectorRetriever()
            reranker = SimpleReranker()
            q_emb = embedder.embed_query(query.strip())
            candidates = retriever.retrieve(db=db, query_embedding=q_emb, top_k=max(top_k * 3, 10))
            reranked = reranker.rerank(candidates, top_k=top_k)
            out: list[dict[str, str | float]] = []
            for c in reranked:
                out.append(
                    {
                        "document_title": str(c.get("document_title", "")),
                        "chunk_text": str(c.get("chunk_text", "")),
                        "relevance_score": float(c.get("rerank_score", c.get("score", 0.0))),
                    }
                )
            return out
        finally:
            db.close()

    @mcp.tool(name="list_documents", description="List uploaded documents and processing status.")
    def mcp_list_documents() -> list[dict[str, str | int]]:
        db = SessionLocal()
        try:
            rows = db.query(Document).order_by(Document.id.desc()).limit(500).all()
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "status": d.upload_status.value,
                    "chunk_count": d.chunk_count,
                }
                for d in rows
            ]
        finally:
            db.close()

    @mcp.resource("document://{document_id}")
    def document_resource(document_id: str) -> str:
        db = SessionLocal()
        try:
            doc = db.get(Document, int(document_id))
            if not doc:
                return ""
            return _read_document_text(doc.file_path, doc.content_type.value)
        except (ValueError, TypeError):
            return ""
        finally:
            db.close()

    return mcp.sse_app()
