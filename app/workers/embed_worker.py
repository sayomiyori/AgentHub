from pathlib import Path

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
    path = Path(file_path)
    if content_type in {"txt", "md"}:
        return path.read_text(encoding="utf-8")
    if content_type == "pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported content type: {content_type}")


@celery_app.task(name="process_document")
def process_document(document_id: int) -> None:
    db = SessionLocal()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    embedder = GeminiEmbedder()
    try:
        document = db.get(Document, document_id)
        if not document:
            return

        document.upload_status = UploadStatus.processing
        db.commit()

        text = _read_document_text(document.file_path, document.content_type.value)
        chunks = chunker.split(text)
        embeddings = []
        batch_size = 512
        for i in range(0, len(chunks), batch_size):
            embeddings.extend(embedder.embed_texts(chunks[i : i + batch_size]))

        chunk_rows = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_rows.append(
                Chunk(
                    document_id=document.id,
                    text=chunk_text,
                    chunk_index=idx,
                    embedding=embedding,
                    meta={"source": document.title, "chunk_index": idx},
                    token_count=max(len(chunk_text) // 4, 1),
                )
            )
        if chunk_rows:
            db.bulk_save_objects(chunk_rows)

        document.chunk_count = len(chunk_rows)
        document.upload_status = UploadStatus.ready
        db.commit()
    except Exception:
        document = db.get(Document, document_id)
        if document:
            document.upload_status = UploadStatus.failed
            db.commit()
        raise
    finally:
        db.close()
