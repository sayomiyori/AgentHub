class SimpleReranker:
    def rerank(self, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []

        # Balance relevance and diversity by slightly penalizing repeated documents.
        seen_docs: dict[int, int] = {}
        scored: list[dict] = []
        for chunk in chunks:
            doc_id = chunk["document_id"]
            penalty = 0.03 * seen_docs.get(doc_id, 0)
            chunk["rerank_score"] = max(float(chunk["score"]) - penalty, 0.0)
            seen_docs[doc_id] = seen_docs.get(doc_id, 0) + 1
            scored.append(chunk)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]
