"""Read-only aggregate inventory of deterministic fallback embeddings."""
import json
import struct

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.services.rag.embedder import GeminiEmbedder


def classify_vector(content: str, vector: list[float]) -> bool:
    if len(vector) != 1536:
        return False
    fallback = GeminiEmbedder.__new__(GeminiEmbedder)._fallback_embedding(content)
    return all(struct.pack("!f", actual) == struct.pack("!f", expected)
               for actual, expected in zip(vector, fallback, strict=True))


def inventory(engine) -> dict[str, int]:
    counts = {"total": 0, "fallback_matches": 0, "unknown": 0,
              "documents_with_fallback": 0, "documents_with_unknown": 0}
    previous_document = None
    seen_fallback = seen_unknown = False
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout = '30s'"))
        rows = connection.execution_options(stream_results=True).execute(
            text("SELECT document_id, text, embedding::text FROM chunks ORDER BY document_id, id")
        )
        for document_id, content, serialized in rows:
            if document_id != previous_document:
                previous_document = document_id
                seen_fallback = seen_unknown = False
            match = classify_vector(content, json.loads(serialized))
            counts["total"] += 1
            counts["fallback_matches" if match else "unknown"] += 1
            if match and not seen_fallback:
                counts["documents_with_fallback"] += 1
                seen_fallback = True
            if not match and not seen_unknown:
                counts["documents_with_unknown"] += 1
                seen_unknown = True
    return counts


def main() -> int:
    engine = None
    counts = None
    try:
        engine = create_engine(get_settings().database_url)
        counts = inventory(engine)
    except Exception:
        pass
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                counts = None
    if counts is None:
        print("Embedding inventory failed; no result is available.")
        return 1
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
