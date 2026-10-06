"""Bounded fresh authorization lookup against a fixed internal origin."""

import asyncio
import json
from typing import Annotated
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from app.platform.config import PlatformSettings
from app.platform.schemas import strict_json


class ContextDenied(Exception):
    pass


class ContextUnavailable(Exception):
    pass


class BotContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    bot_id: UUID
    tenant_id: UUID
    telegram_bot_id: Annotated[StrictInt, Field(gt=0, le=2**63 - 1)]
    is_active: StrictBool


class BotContextClient:
    def __init__(self, config: PlatformSettings, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    async def resolve(self, bot_id: UUID) -> BotContext:
        try:
            async with (
                asyncio.timeout(8),
                httpx.AsyncClient(
                    timeout=httpx.Timeout(5, connect=2),
                    transport=self.transport,
                    follow_redirects=False,
                    trust_env=False,
                    verify=True,
                ) as client,
            ):
                async with client.stream(
                    "GET",
                    f"{self.config.webhook_url.rstrip('/')}/internal/v1/bots/{bot_id}/context",
                    headers={
                        "X-Service-Key": self.config.context_key.get_secret_value(),
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    if response.status_code in {401, 403, 404}:
                        raise ContextDenied()
                    if (
                        response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                    ):
                        raise ContextUnavailable()
                    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                        raise ContextUnavailable()
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if len(body) + len(chunk) > 65536:
                            raise ContextUnavailable()
                        body.extend(chunk)
                    decoded = strict_json(bytes(body))
                    context = BotContext.model_validate_json(json.dumps(decoded))
                    if context.bot_id != bot_id or not context.is_active:
                        raise ContextDenied()
                    return context
        except (httpx.HTTPError, TimeoutError, ValueError, UnicodeError, RecursionError):
            raise ContextUnavailable() from None
