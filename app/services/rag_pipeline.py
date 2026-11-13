import hashlib
from uuid import UUID

from sqlalchemy.orm import Session

from app.cache.semantic_cache import get_cached_rag_answer, set_cached_rag_answer
from app.config import get_settings
from app.metrics import rag_retrieval_timer
from app.models.document import Document
from app.services.rag.embedder import GeminiEmbedder
from app.services.rag.generator import AnswerGenerator
from app.services.rag.reranker import SimpleReranker
from app.services.rag.retriever import PgVectorRetriever


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = GeminiEmbedder()
        self.retriever = PgVectorRetriever()
        self.reranker = SimpleReranker()
        self.generator = AnswerGenerator()

    def ask(
        self,
        db: Session,
        question: str,
        top_k: int = 5,
        *,
        provider: str | None = None,
        model: str | None = None,
        conversation_id: int | None = None,
        request_id: str | None = None,
        tenant_id: UUID | None = None,
    ) -> dict:
        query_embedding = self.embedder.embed_query(question)
        corpus = (db.query(Document.id, Document.content_revision, Document.upload_status, Document.chunk_count)
                  .filter(Document.tenant_id == tenant_id).order_by(Document.id).all())
        settings = get_settings()
        # Old in-flight writers retain their old corpus namespace after a committed change.
        digest = hashlib.sha256(repr((corpus, settings.embedding_model, settings.llm_provider,
                                      settings.llm_model, settings.llm_fallback_provider,
                                      settings.llm_fallback_model)).encode()).hexdigest()
        namespace = f"{tenant_id or 'legacy'}:{digest}"

        cached = get_cached_rag_answer(
            question,
            query_embedding,
            top_k=top_k,
            provider=provider,
            model=model,
            namespace=namespace,
        )
        if cached is not None:
            return {**cached, "semantic_cache_hit": True, "tokens_used": 0, "cost_usd": 0.0}

        with rag_retrieval_timer():
            candidates = self.retriever.retrieve(db=db, query_embedding=query_embedding, top_k=max(top_k * 3, 10),
                                                 tenant_id=tenant_id)
            reranked = self.reranker.rerank(candidates, top_k=top_k)
        answer, used_chunks, usage = self.generator.generate(
            question=question,
            chunks=reranked,
            provider=provider,
            model=model,
            db=db,
            conversation_id=conversation_id,
            request_id=request_id,
            tenant_id=tenant_id,
        )
        out = {
            "answer": answer,
            "sources": used_chunks,
            "tokens_used": usage.tokens_used,
            "cost_usd": usage.cost_usd,
            "semantic_cache_hit": False,
        }
        set_cached_rag_answer(
            question,
            query_embedding,
            top_k=top_k,
            provider=provider,
            model=model,
            namespace=namespace,
            payload={
                "answer": answer,
                "sources": used_chunks,
                "tokens_used": usage.tokens_used,
                "cost_usd": usage.cost_usd,
                "semantic_cache_hit": False,
            },
        )
        return out
