from app.models.usage import UsageMetrics
from app.services.llm.factory import LLMFactory


class AnswerGenerator:
    def __init__(self) -> None:
        self.factory = LLMFactory()

    def _build_context(self, chunks: list[dict]) -> str:
        rows = []
        for c in chunks:
            rows.append(
                f"[chunk_id={c['chunk_id']}] {c['document_title']} (score={c['rerank_score']:.4f}):\n{c['chunk_text']}"
            )
        return "\n\n".join(rows)

    def _fallback_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "I do not know based on the available context."
        top = chunks[:2]
        snippets = []
        for item in top:
            snippets.append(f"[chunk_id={item['chunk_id']}] {item['chunk_text'][:260]}")
        return (
            "Model generation is temporarily unavailable. "
            "Here are the most relevant context snippets:\n\n"
            + "\n\n".join(snippets)
            + f"\n\nQuestion: {question}"
        )

    def generate(
        self,
        question: str,
        chunks: list[dict],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[str, list[dict], UsageMetrics]:
        context = self._build_context(chunks)
        system_prompt = (
            "You are a RAG assistant. Answer only from provided context. "
            "If data is missing, say you do not know. Add citations by chunk_id."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context chunks:\n{context}\n\n"
            "Return concise answer in plain text with inline references like [chunk_id=12]."
        )

        try:
            resp = self.factory.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                provider=provider,
                model=model,
                temperature=0.2,
            )
            answer = resp.content or ""
            tokens_used = resp.usage.input_tokens + resp.usage.output_tokens
            cost_usd = resp.usage.cost_usd
        except Exception:
            answer = self._fallback_answer(question, chunks)
            tokens_used = 0
            cost_usd = 0.0

        return answer, chunks, UsageMetrics(tokens_used=tokens_used, cost_usd=round(cost_usd, 8))
