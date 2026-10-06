from openai import APIStatusError

from app.config import get_settings
from app.platform.config import get_platform_settings
from app.platform.prompts import SYSTEM_PROMPT
from app.services.llm.base import LLMResponse
from app.services.llm.groq import GroqCompletionError, GroqProvider


class GenerationConfigurationError(ValueError):
    pass


class GenerationError(RuntimeError):
    pass


class GenerationRejectedError(GenerationError):
    pass


class GenerationInvalidResponse(GenerationError):
    pass


def validate_generation_configuration(model: str) -> None:
    settings = get_platform_settings()
    if (
        not settings.telegram_ai_enabled or settings.provider != "groq"
        or not model.strip() or model != model.strip() or len(model) > 128
        or not get_settings().groq_api_key.strip()
    ):
        raise GenerationConfigurationError("Platform generation is not configured")


def generate_platform_answer(question: str, model: str) -> LLMResponse:
    validate_generation_configuration(model)
    if not question.strip() or len(question) > 4096:
        raise GenerationConfigurationError("Invalid admitted question")
    try:
        response = GroqProvider().generate(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question}],
            model=model, timeout=20, max_retries=0, max_completion_tokens=1024,
            require_complete=True,
        )
    except GroqCompletionError:
        raise GenerationInvalidResponse("Invalid platform completion") from None
    except APIStatusError as error:
        if error.status_code in {400, 401, 403, 404, 422}:
            raise GenerationRejectedError("Platform generation rejected") from None
        raise GenerationError("Platform generation failed") from None
    except Exception:
        # Never propagate remote error bodies, request headers or message content.
        raise GenerationError("Platform generation failed") from None
    if len(response.content) > 4096:
        response.content = response.content[:4095] + "…"
    return response
