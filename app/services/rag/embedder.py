import hashlib

from app.config import get_settings
from google import genai


class GeminiEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.embedding_model
        self.client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    def _fallback_embedding(self, text: str) -> list[float]:
        # Deterministic 1536-dim embedding fallback when Gemini embedding API is unavailable.
        values: list[float] = []
        for i in range(1536):
            digest = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
            int_val = int.from_bytes(digest[:4], byteorder="big", signed=False)
            values.append((int_val / 4294967295.0) * 2 - 1)
        return values

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.client:
            return [self._fallback_embedding(text) for text in texts]

        vectors: list[list[float]] = []
        for text in texts:
            try:
                response = self.client.models.embed_content(model=self.model, contents=text)
                embeddings = getattr(response, "embeddings", None) or []
                if embeddings and getattr(embeddings[0], "values", None):
                    vectors.append(list(embeddings[0].values))
                else:
                    vectors.append(self._fallback_embedding(text))
            except Exception:
                vectors.append(self._fallback_embedding(text))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        if not self.client:
            return self._fallback_embedding(query)

        try:
            response = self.client.models.embed_content(model=self.model, contents=query)
            embeddings = getattr(response, "embeddings", None) or []
            if embeddings and getattr(embeddings[0], "values", None):
                return list(embeddings[0].values)
            return self._fallback_embedding(query)
        except Exception:
            return self._fallback_embedding(query)
