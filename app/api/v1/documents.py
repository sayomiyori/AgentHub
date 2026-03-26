from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.models.document import ContentType, Document, UploadStatus
from app.workers.embed_worker import process_document

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


class DocumentOut(BaseModel):
    id: int
    title: str
    content_type: str
    upload_status: str
    chunk_count: int

    class Config:
        from_attributes = True


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in {"txt", "md", "pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: txt, md, pdf")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid4()}_{file.filename}"
    path = upload_dir / saved_name
    path.write_bytes(await file.read())

    document = Document(
        title=file.filename or "untitled",
        content_type=ContentType(ext),
        upload_status=UploadStatus.pending,
        file_path=str(path),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    process_document.delay(document.id)
    return {"document_id": document.id, "status": "processing"}


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return db.query(Document).order_by(Document.created_at.desc()).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)) -> dict:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"status": "deleted"}
