from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.conversation import Conversation, Message
from app.platform.resource_access import read_scope

router = APIRouter(prefix="/conversations", tags=["conversations"])


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    role: str
    content: str
    tokens_used: int
    cost_usd: float
    sources: list[dict]


@router.get("")
def list_conversations(tenant_id: UUID | None = Depends(read_scope), offset: int = Query(0, ge=0),
                       limit: int = Query(50, ge=1, le=100), db: Session = Depends(get_db)) -> list[dict]:
    query = db.query(Conversation).filter(Conversation.tenant_id == tenant_id)
    if tenant_id is None:
        conversations = query.order_by(Conversation.created_at.desc()).all()
    else:
        conversations = query.order_by(Conversation.id).offset(offset).limit(limit).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in conversations]


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(conversation_id: int, tenant_id: UUID | None = Depends(read_scope),
                 offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                 db: Session = Depends(get_db)) -> list[Message]:
    conversation = (db.query(Conversation)
                    .filter(Conversation.id == conversation_id, Conversation.tenant_id == tenant_id).first())
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    query = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.id.asc())
    return query.all() if tenant_id is None else query.offset(offset).limit(limit).all()
