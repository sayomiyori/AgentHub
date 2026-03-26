from sqlalchemy import text
from sqlalchemy.orm import Session


class PgVectorRetriever:
    def retrieve(self, db: Session, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        vector_literal = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
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
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        )
        rows = db.execute(sql, {"query_embedding": vector_literal, "top_k": top_k}).mappings().all()
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
            ORDER BY c.id DESC
            LIMIT :top_k
            """
        )
        fallback_rows = db.execute(fallback_sql, {"top_k": top_k}).mappings().all()
        return [dict(row) for row in fallback_rows]
