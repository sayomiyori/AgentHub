from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.conversation import Conversation, Message, MessageRole
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("")
def ask_question(payload: QueryRequest, db: Session = Depends(get_db)) -> dict:
    if payload.conversation_id:
        conversation = db.get(Conversation, payload.conversation_id)
    else:
        conversation = Conversation(title=payload.question[:60] or "New conversation")
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.user,
        content=payload.question,
        tokens_used=0,
        cost_usd=0,
        sources=[],
    )
    db.add(user_message)

    result = pipeline.ask(db=db, question=payload.question, top_k=payload.top_k)
    source_payload = [
        {
            "chunk_id": item["chunk_id"],
            "document_title": item["document_title"],
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

    response_sources = [
        {
            "document_title": item["document_title"],
            "chunk_text_preview": item["chunk_text"][:220],
            "score": float(item["rerank_score"]),
        }
        for item in result["sources"]
    ]
    return {
        "answer": result["answer"],
        "sources": response_sources,
        "tokens_used": result["tokens_used"],
        "cost_usd": result["cost_usd"],
        "conversation_id": conversation.id,
    }
