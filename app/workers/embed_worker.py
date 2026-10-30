import hashlib
from io import BytesIO, StringIO
from pathlib import Path
from uuid import uuid4

from celery import Celery
from pypdf import PdfReader

from app.config import get_settings
from app.db.session import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document, UploadStatus
from app.services.rag.chunker import TextChunker
from app.services.rag.embedder import GeminiEmbedder

settings = get_settings()
celery_app = Celery("agenthub", broker=settings.redis_url, backend=settings.redis_url)


def _read_document_text(file_path: str, content_type: str) -> str:
    return _parse_document_bytes(Path(file_path).read_bytes(), content_type)


def _parse_document_bytes(source: bytes, content_type: str) -> str:
    if content_type in {"txt", "md"}:
        return StringIO(source.decode("utf-8"), newline=None).read()
    if content_type == "pdf":
        reader = PdfReader(BytesIO(source))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported content type: {content_type}")


@celery_app.task(name="process_document")
def process_document(document_id: int) -> None:
    db = SessionLocal()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    embedder = GeminiEmbedder()
    try:
        document = db.query(Document).filter(Document.id == document_id).with_for_update().first()
        if not document or document.upload_status == UploadStatus.ready:
            return

        if document.tenant_id is not None:
            embedder.require_provider()

        document.upload_status = UploadStatus.processing
        db.flush()

        source = Path(document.file_path).read_bytes()
        source_digest = hashlib.sha256(source).hexdigest()
        text = _parse_document_bytes(source, document.content_type.value)
        provenance = {"provider": "gemini" if embedder.client is not None else "offline_sha256",
                      "model": embedder.model if embedder.client is not None else "sha256-v1", "dimensions": 1536}
        chunks = chunker.split(text)
        embeddings = []
        batch_size = 512
        for i in range(0, len(chunks), batch_size):
            embeddings.extend(embedder.embed_texts(chunks[i : i + batch_size]))

        try:
            current_digest = hashlib.sha256(Path(document.file_path).read_bytes()).hexdigest()
        except OSError:
            raise ValueError("Document source unavailable after indexing") from None
        if current_digest != source_digest:
            raise ValueError("Document source changed during indexing")

        chunk_rows = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_rows.append(
                Chunk(
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=idx,
                    embedding=embedding,
                    meta={"source": document.title, "chunk_index": idx,
                          "source_sha256": source_digest, "embedding": provenance},
                    token_count=max(len(chunk_text) // 4, 1),
                )
            )
        if chunk_rows:
            db.bulk_save_objects(chunk_rows)

        document.chunk_count = len(chunk_rows)
        document.content_revision = uuid4()
        document.upload_status = UploadStatus.ready
        db.commit()
    except Exception:
        db.rollback()
        # A retry can finish after rollback releases the original lock.
        document = (
            db.query(Document).filter(Document.id == document_id)
            .populate_existing().with_for_update().first()
        )
        if document and document.upload_status != UploadStatus.ready:
            document.upload_status = UploadStatus.failed
            db.commit()
        raise
    finally:
        db.close()
