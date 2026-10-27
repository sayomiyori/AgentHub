"""Canonical tenant scope for existing standalone resource routers."""

from typing import Literal
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from pydantic import SecretStr

from app.api.v1.platform_reads import get_access_client
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.tenant_access import TenantAccessClient


def platform_active(config: PlatformSettings) -> bool:
    return config.telegram_ai_enabled or config.read_enabled or getattr(config, "tenant_enabled", False)


def require_standalone() -> None:
    if platform_active(get_platform_settings()):
        raise HTTPException(404, "Not found")


async def _scope(request: Request, config: PlatformSettings, issuer: TenantAccessClient,
                 permission: Literal["ai.read", "ai.configure"]) -> UUID | None:
    value = request.path_params.get("tenant_id")
    if value is None:
        if platform_active(config):
            raise HTTPException(404, "Not found")
        return None
    if not config.tenant_enabled:
        raise HTTPException(404, "Not found")
    try:
        tenant_id = UUID(str(value))
    except ValueError:
        raise HTTPException(422, "Invalid tenant ID") from None
    values = request.headers.getlist("authorization")
    parts = values[0].split(" ") if len(values) == 1 else []
    if (len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]
        or len(parts[1]) > 8192 or not parts[1].isascii()
        or any(c.isspace() or not c.isprintable() for c in parts[1])):
        raise HTTPException(401, "Invalid authentication")
    await issuer.authorize(tenant_id, SecretStr(parts[1]), permission, forbidden_status=403)
    return tenant_id


async def read_scope(request: Request, config: PlatformSettings = Depends(get_platform_settings),
                     issuer: TenantAccessClient = Depends(get_access_client)) -> UUID | None:
    return await _scope(request, config, issuer, "ai.read")


async def write_scope(request: Request, config: PlatformSettings = Depends(get_platform_settings),
                      issuer: TenantAccessClient = Depends(get_access_client)) -> UUID | None:
    return await _scope(request, config, issuer, "ai.configure")
