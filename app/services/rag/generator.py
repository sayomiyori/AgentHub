from anthropic import Anthropic
from google import genai
from openai import OpenAI

from app.config import get_settings
from app.models.usage import UsageMetrics


class AnswerGenerator:
    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

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

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, list[dict], UsageMetrics]:
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

        if self.provider == "gemini" and self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                )
                answer = getattr(response, "text", "") or ""
                usage = getattr(response, "usage_metadata", None)
                tokens_used = int(getattr(usage, "total_token_count", 0) or 0)
            except Exception:
                answer = self._fallback_answer(question, chunks)
                tokens_used = 0
        elif self.provider == "anthropic" and self.anthropic_client:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=700,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            answer = ""
            for block in response.content:
                if getattr(block, "type", "") == "text":
                    answer += block.text
            tokens_used = int((response.usage.input_tokens or 0) + (response.usage.output_tokens or 0))
        else:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer = response.choices[0].message.content or ""
                tokens_used = int(response.usage.total_tokens if response.usage else 0)
            except Exception:
                answer = self._fallback_answer(question, chunks)
                tokens_used = 0
        cost_usd = round(tokens_used * 0.000001, 6)
        return answer, chunks, UsageMetrics(tokens_used=tokens_used, cost_usd=cost_usd)
