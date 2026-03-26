from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.conversation import Conversation, Message

router = APIRouter(prefix="/conversations", tags=["conversations"])


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    tokens_used: int
    cost_usd: float
    sources: list[dict]

    class Config:
        from_attributes = True


@router.get("")
def list_conversations(db: Session = Depends(get_db)) -> list[dict]:
    conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in conversations]


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(conversation_id: int, db: Session = Depends(get_db)) -> list[Message]:
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.id.asc()).all()
