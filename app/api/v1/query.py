from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.conversation import Conversation, Message, MessageRole
from app.services.agent.orchestrator import AgentOrchestrator
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])
pipeline = RAGPipeline()
orchestrator = AgentOrchestrator()


class QueryRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_agent: bool = True
    provider: str | None = Field(
        default=None,
        description='LLM provider: "openai", "anthropic", or "gemini" (default from settings).',
    )
    model: str | None = Field(
        default=None,
        description="Model id (e.g. gpt-4o-mini, claude-3-5-haiku-20241022, models/gemini-2.5-flash).",
    )


def _get_or_create_conversation(db: Session, question: str, conversation_id: int | None) -> Conversation:
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation:
            return conversation
    conversation = Conversation(title=question[:60] or "New conversation")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.post("")
def ask_question(payload: QueryRequest, db: Session = Depends(get_db)) -> dict:
    conversation = _get_or_create_conversation(db, payload.question, payload.conversation_id)

    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content=payload.question,
        tokens_used=0,
        cost_usd=0,
        sources=[],
    )
    db.add(user_message)
    db.flush()

    if payload.use_agent:
        result = orchestrator.run(
            db=db,
            question=payload.question,
            conversation_id=conversation.id,
            top_k=payload.top_k,
            provider=payload.provider,
            model=payload.model,
        )
        source_payload = [
            {
                "chunk_id": s.get("chunk_id"),
                "document_title": s.get("document_title"),
                "score": float(s.get("score", 0.0)),
            }
            for s in result.get("sources", [])
        ]
        response_sources = [
            {
                "document_title": s.get("document_title"),
                "chunk_text_preview": s.get("chunk_text_preview", ""),
                "score": float(s.get("score", 0.0)),
            }
            for s in result.get("sources", [])
        ]
    else:
        result = pipeline.ask(
            db=db,
            question=payload.question,
            top_k=payload.top_k,
            provider=payload.provider,
            model=payload.model,
        )
        source_payload = [
            {
                "chunk_id": item["chunk_id"],
                "document_title": item["document_title"],
                "score": float(item["rerank_score"]),
            }
            for item in result["sources"]
        ]
        response_sources = [
            {
                "document_title": item["document_title"],
                "chunk_text_preview": item["chunk_text"][:220],
                "score": float(item["rerank_score"]),
            }
            for item in result["sources"]
        ]

    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.assistant,
        content=result["answer"],
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        sources=source_payload,
    )
    db.add(assistant_message)
    db.commit()

    return {
        "answer": result["answer"],
        "sources": response_sources,
        "tokens_used": result["tokens_used"],
        "cost_usd": result["cost_usd"],
        "conversation_id": conversation.id,
        "tools_called": result.get("tools_called", []),
    }
