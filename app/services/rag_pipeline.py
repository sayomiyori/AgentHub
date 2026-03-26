from sqlalchemy.orm import Session

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

    def ask(self, db: Session, question: str, top_k: int = 5) -> dict:
        query_embedding = self.embedder.embed_query(question)
        candidates = self.retriever.retrieve(db=db, query_embedding=query_embedding, top_k=max(top_k * 3, 10))
        reranked = self.reranker.rerank(candidates, top_k=top_k)
        answer, used_chunks, usage = self.generator.generate(question=question, chunks=reranked)
        return {
            "answer": answer,
            "sources": used_chunks,
            "tokens_used": usage.tokens_used,
            "cost_usd": usage.cost_usd,
        }
