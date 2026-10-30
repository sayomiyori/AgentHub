import hashlib
import math
import struct
import time

from google import genai
from google.genai import types

from app.config import get_settings
from app.metrics import embedding_duration_seconds


class EmbeddingUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Embedding service unavailable")


class GeminiEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.embedding_model
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=20000, retry_options=types.HttpRetryOptions(attempts=1)),
        ) if settings.gemini_api_key else None

    def require_provider(self) -> None:
        if self.client is None:
            raise EmbeddingUnavailable()

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

        return [self._provider_embedding(text) for text in texts]

    def _provider_embedding(self, text: str) -> list[float]:
        try:
            if self.client is None:
                raise ValueError("Provider is not configured")
            if self.model in {"gemini-embedding-001", "models/gemini-embedding-001"}:
                response = self.client.models.embed_content(
                    model=self.model, contents=text, config=types.EmbedContentConfig(output_dimensionality=1536)
                )
            else:
                response = self.client.models.embed_content(model=self.model, contents=text)
            embeddings = response.embeddings
            values = embeddings[0].values if embeddings else None
            if (values is None or len(values) != 1536 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or abs(value) > 3.4028235e38
                for value in values
            )):
                raise ValueError("Invalid embedding")
            stored = [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]
            # Keep float32 cosine norms away from underflow and overflow.
            if max(map(abs, stored)) < 1.0842021724855044e-19 or sum(x * x for x in stored) > 1e38:
                raise ValueError("Invalid embedding norm")
            return list(values)
        except Exception:
            raise EmbeddingUnavailable() from None

    def embed_query(self, query: str) -> list[float]:
        t0 = time.perf_counter()
        try:
            if not self.client:
                return self._fallback_embedding(query)

            return self._provider_embedding(query)
        finally:
            embedding_duration_seconds.observe(time.perf_counter() - t0)
