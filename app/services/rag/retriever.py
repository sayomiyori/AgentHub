import json
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.rag.embedder import EmbeddingUnavailable

MAX_TENANT_SEARCH_CHUNKS = 10000
MAX_TENANT_SEARCH_DOCUMENTS = 10000


class PgVectorRetriever:
    def retrieve(self, db: Session, query_embedding: list[float], top_k: int = 5,
                 tenant_id: UUID | None = None, embedding_model: str | None = None) -> list[dict]:
        vector_literal = "[" + ",".join(f"{x:.9g}" for x in query_embedding) + "]"
        if tenant_id is not None:
            if not embedding_model:
                raise EmbeddingUnavailable()
            receipt = json.dumps({"provider": "gemini", "model": embedding_model, "dimensions": 1536})
            rows = db.execute(text("""
                WITH scoped_documents AS MATERIALIZED (
                    SELECT id, title, upload_status, chunk_count
                    FROM documents WHERE tenant_id = CAST(:tenant_id AS uuid)
                    LIMIT :document_fetch_limit
                ), corpus AS MATERIALIZED (
                    SELECT c.id AS chunk_id, c.document_id, c.text AS chunk_text,
                           c.metadata, c.embedding, c.chunk_index,
                           d.title AS document_title, d.upload_status
                    FROM chunks c JOIN scoped_documents d ON d.id = c.document_id
                    LIMIT :corpus_fetch_limit
                ), compatibility AS (
                    SELECT (SELECT count(*) FROM scoped_documents) <= :document_limit
                    AND (SELECT count(*) FROM corpus) <= :corpus_limit AND NOT EXISTS (
                        SELECT 1 FROM corpus
                        WHERE metadata->'embedding' IS DISTINCT FROM CAST(:receipt AS jsonb)
                           OR upload_status IS DISTINCT FROM 'ready'
                    ) AND NOT EXISTS (
                        SELECT 1 FROM scoped_documents d
                        LEFT JOIN corpus c ON c.document_id = d.id
                        GROUP BY d.id, d.chunk_count, d.upload_status
                        HAVING (count(c.chunk_id) > 0 OR d.upload_status = 'ready') AND (
                            d.chunk_count <> count(c.chunk_id)
                            OR count(DISTINCT c.chunk_index) <> count(c.chunk_id)
                            OR (count(c.chunk_id) > 0 AND (
                                min(c.chunk_index) <> 0 OR max(c.chunk_index) <> count(c.chunk_id) - 1
                            ))
                        )
                    ) AS ok
                )
                SELECT compatibility.ok AS corpus_compatible, hits.*
                FROM compatibility LEFT JOIN LATERAL (
                    SELECT chunk_id, document_id, chunk_text, metadata, document_title,
                           (1 - (embedding <=> CAST(:query_embedding AS vector))) AS score
                    FROM corpus WHERE compatibility.ok
                    ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :top_k
                ) hits ON true
            """), {"tenant_id": tenant_id, "receipt": receipt,
                   "query_embedding": vector_literal, "top_k": top_k,
                   "corpus_limit": MAX_TENANT_SEARCH_CHUNKS,
                   "corpus_fetch_limit": MAX_TENANT_SEARCH_CHUNKS + 1,
                   "document_limit": MAX_TENANT_SEARCH_DOCUMENTS,
                   "document_fetch_limit": MAX_TENANT_SEARCH_DOCUMENTS + 1}).mappings().all()
            if not rows or rows[0]["corpus_compatible"] is not True:
                raise EmbeddingUnavailable()
            return [{key: value for key, value in row.items() if key != "corpus_compatible"}
                    for row in rows if row["chunk_id"] is not None]
        sql = text(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.text AS chunk_text,
                c.metadata,
                d.title AS document_title,
                (1 - (c.embedding <=> CAST(:query_embedding AS vector))) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.tenant_id IS NOT DISTINCT FROM CAST(:tenant_id AS uuid)
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        )
        rows = db.execute(sql, {"query_embedding": vector_literal, "top_k": top_k,
                                "tenant_id": tenant_id}).mappings().all()
        if rows:
            return [dict(row) for row in rows]

        # Safety fallback: return recent chunks when vector search is unavailable/empty.
        fallback_sql = text(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.text AS chunk_text,
                c.metadata,
                d.title AS document_title,
                0.0 AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.tenant_id IS NOT DISTINCT FROM CAST(:tenant_id AS uuid)
            ORDER BY c.id DESC
            LIMIT :top_k
            """
        )
        fallback_rows = db.execute(fallback_sql, {"top_k": top_k, "tenant_id": tenant_id}).mappings().all()
        return [dict(row) for row in fallback_rows]
