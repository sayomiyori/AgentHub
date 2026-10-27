from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.models.document import ContentType, Document, UploadStatus
from app.platform.resource_access import read_scope, write_scope
from app.workers.embed_worker import process_document

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    content_type: str
    upload_status: str
    chunk_count: int


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...), tenant_id: UUID | None = Depends(write_scope),
                          db: Session = Depends(get_db)) -> dict:
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in {"txt", "md", "pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: txt, md, pdf")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    content = bytearray()
    while chunk := await file.read(65536):
        if len(content) + len(chunk) > 10 * 1024 * 1024:
            raise HTTPException(413, "Document exceeds 10 MiB")
        content.extend(chunk)
    if not content:
        raise HTTPException(400, "Empty document")
    try:
        if ext == "pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError()
        else:
            content.decode("utf-8")
    except (ValueError, UnicodeError):
        raise HTTPException(400, "Invalid document content") from None
    saved_name = f"{uuid4()}.{ext}"
    path = upload_dir / saved_name

    document = Document(
        title=(file.filename or "untitled")[:255],
        tenant_id=tenant_id,
        content_type=ContentType(ext),
        upload_status=UploadStatus.pending,
        file_path=str(path),
    )
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    try:
        db.add(document)
        db.commit()
    except Exception:
        db.rollback()
        # The database may have committed before its acknowledgement was lost.
        raise
    db.refresh(document)

    try:
        process_document.delay(document.id)
    except Exception:
        # A lost broker acknowledgement does not prove the worker received nothing.
        raise HTTPException(503, "Document queue outcome unknown") from None
    return {"document_id": document.id, "status": "processing"}


@router.get("", response_model=list[DocumentOut])
def list_documents(tenant_id: UUID | None = Depends(read_scope), offset: int = Query(0, ge=0),
                   limit: int = Query(50, ge=1, le=100), db: Session = Depends(get_db)) -> list[Document]:
    query = db.query(Document).filter(Document.tenant_id == tenant_id)
    if tenant_id is None:
        return query.order_by(Document.created_at.desc()).all()
    return query.order_by(Document.id).offset(offset).limit(limit).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, tenant_id: UUID | None = Depends(read_scope),
                 db: Session = Depends(get_db)) -> Document:
    doc = db.query(Document).filter(Document.id == document_id, Document.tenant_id == tenant_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}")
def delete_document(document_id: int, tenant_id: UUID | None = Depends(write_scope),
                    db: Session = Depends(get_db)) -> dict:
    doc = (db.query(Document).filter(Document.id == document_id, Document.tenant_id == tenant_id)
           .with_for_update().first())
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    path = Path(doc.file_path).resolve()
    if path.is_relative_to(Path(settings.upload_dir).resolve()):
        path.unlink(missing_ok=True)
    return {"status": "deleted"}
