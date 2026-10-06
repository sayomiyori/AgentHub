"""Controlled external transport, imported only by the verification worker CLI."""
import os
import time

import httpx
from billiard.exceptions import SoftTimeLimitExceeded


def response(request: httpx.Request) -> httpx.Response:
    bot = os.environ["VERIFICATION_BOT_ID"]
    tenant = os.environ["VERIFICATION_TENANT_ID"]
    if request.url.host == "verification-context":
        if request.url.path != f"/internal/v1/bots/{bot}/context":
            return httpx.Response(403)
        return httpx.Response(200, json={
            "bot_id": bot, "tenant_id": tenant, "telegram_bot_id": 123, "is_active": True,
        })
    if str(request.url) != "https://api.groq.com/openai/v1/chat/completions":
        raise AssertionError("Unexpected verification destination")
    if os.getenv("VERIFICATION_MODE") == "hard_timeout":
        # Deliberately block the test transport past both limits. Production
        # uses the cancellable async SDK path and never imports this module.
        try:
            time.sleep(40)
        except SoftTimeLimitExceeded:
            time.sleep(20)
    return httpx.Response(200, json={
        "id": "synthetic", "object": "chat.completion", "created": 1, "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Synthetic answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


class VerificationClient(httpx.AsyncClient):
    def __init__(self, **kwargs):
        kwargs["transport"] = httpx.MockTransport(response)
        super().__init__(**kwargs)


httpx.AsyncClient = VerificationClient

from app.workers.telegram_worker import celery_app  # noqa: E402, F401
