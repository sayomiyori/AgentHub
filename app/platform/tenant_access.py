"""Fresh issuer authorization; no bearer or upstream body is exposed in errors."""

import asyncio
import json
from typing import Literal
from uuid import UUID

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, SecretStr, StrictBool

from app.platform.config import PlatformSettings
from app.platform.schemas import strict_json


class AuthorizedContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: UUID
    tenant_id: UUID
    role: Literal["owner", "member"]
    permission: Literal["ai.read"]
    allowed: StrictBool


class TenantAccessClient:
    def __init__(self, config: PlatformSettings, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    async def authorize(self, tenant_id: UUID, bearer: SecretStr) -> None:
        try:
            async with (
                asyncio.timeout(8),
                httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2), transport=self.transport,
                                  follow_redirects=False, trust_env=False, verify=True) as client,
            ):
                async with client.stream(
                    "POST", f"{self.config.auth_url.rstrip('/')}/api/v1/tenants/{tenant_id}/authorize",
                    headers={"Authorization": "Bearer " + bearer.get_secret_value(), "Accept-Encoding": "identity"},
                    json={"permission": "ai.read"},
                ) as response:
                    if response.status_code == 401:
                        raise HTTPException(401, "Invalid authentication")
                    if response.status_code in {403, 404}:
                        raise HTTPException(404, "Scope not found")
                    if (response.status_code != 200
                        or response.headers.get("content-encoding", "identity") != "identity"
                        or response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                            != "application/json"):
                        raise ValueError()
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if len(body) + len(chunk) > 65536:
                            raise ValueError()
                        body.extend(chunk)
                    context = AuthorizedContext.model_validate_json(json.dumps(strict_json(bytes(body))))
                    if context.tenant_id != tenant_id or not context.allowed:
                        raise ValueError()
        except (httpx.HTTPError, TimeoutError, ValueError, UnicodeError, RecursionError):
            raise HTTPException(503, "Authorization unavailable") from None
