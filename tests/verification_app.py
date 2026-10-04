"""HTTP smoke entrypoint: production app with only the LLM provider boundary replaced."""
from app.services.llm.base import LLMResponse, LLMUsage
from app.services.llm.gemini import GeminiProvider


def generate(self, messages, **kwargs):
    if any("provider-failure" in message["content"] for message in messages):
        raise TimeoutError("Verification provider timeout")
    return LLMResponse("Verification answer", usage=LLMUsage(10, 5, 0.0001), provider="gemini", model="verification")


GeminiProvider.generate = generate

from app.main import app  # noqa: E402, F401
