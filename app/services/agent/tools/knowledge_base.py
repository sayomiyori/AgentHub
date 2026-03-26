from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.services.agent.tools.base import BaseTool
from app.services.rag_pipeline import RAGPipeline


class KnowledgeBaseTool(BaseTool):
    name = "search_knowledge_base"
    description = "Search uploaded documents for relevant information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, db: Session, rag_pipeline: RAGPipeline) -> None:
        self.db = db
        self.rag_pipeline = rag_pipeline

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", 5))
        query_embedding = self.rag_pipeline.embedder.embed_query(query)
        candidates = self.rag_pipeline.retriever.retrieve(
            db=self.db, query_embedding=query_embedding, top_k=max(top_k * 3, 10)
        )
        reranked = self.rag_pipeline.reranker.rerank(candidates, top_k=top_k)
        results = [
            {
                "chunk_id": c["chunk_id"],
                "document_title": c["document_title"],
                "chunk_text": c["chunk_text"],
                "score": float(c.get("rerank_score", c.get("score", 0.0))),
            }
            for c in reranked
        ]
        return {"query": query, "results": results}
