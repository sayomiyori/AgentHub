from sqlalchemy.orm import Session

from app.cache.semantic_cache import get_cached_rag_answer, set_cached_rag_answer
from app.metrics import rag_retrieval_timer
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
    ) -> dict:
        query_embedding = self.embedder.embed_query(question)

        cached = get_cached_rag_answer(
            question,
            query_embedding,
            top_k=top_k,
            provider=provider,
            model=model,
        )
        if cached is not None:
            return {**cached, "semantic_cache_hit": True}

        with rag_retrieval_timer():
            candidates = self.retriever.retrieve(db=db, query_embedding=query_embedding, top_k=max(top_k * 3, 10))
            reranked = self.reranker.rerank(candidates, top_k=top_k)
        answer, used_chunks, usage = self.generator.generate(
            question=question,
            chunks=reranked,
            provider=provider,
            model=model,
            db=db,
            conversation_id=conversation_id,
            request_id=request_id,
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
            payload={
                "answer": answer,
                "sources": used_chunks,
                "tokens_used": usage.tokens_used,
                "cost_usd": usage.cost_usd,
                "semantic_cache_hit": False,
            },
        )
        return out
