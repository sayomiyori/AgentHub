from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.platform.bot_context import BotContextClient, ContextDenied, ContextUnavailable
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.reads import JobPage, UsageView, list_jobs, read_usage
from app.platform.tenant_access import TenantAccessClient


def get_access_client(config: PlatformSettings = Depends(get_platform_settings)) -> TenantAccessClient:
    return TenantAccessClient(config)


def get_bot_client(config: PlatformSettings = Depends(get_platform_settings)) -> BotContextClient:
    return BotContextClient(config)


async def authorize_scope(
    tenant_id: UUID, bot_id: UUID, request: Request,
    config: PlatformSettings = Depends(get_platform_settings),
    issuer: TenantAccessClient = Depends(get_access_client),
    bots: BotContextClient = Depends(get_bot_client),
) -> None:
    if not config.read_enabled:
        raise HTTPException(404, "Not found")
    values = request.headers.getlist("authorization")
    parts = values[0].split(" ") if len(values) == 1 else []
    if (len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]
        or len(parts[1]) > 8192 or not parts[1].isascii()
        or any(c.isspace() or not c.isprintable() for c in parts[1])):
        raise HTTPException(401, "Invalid authentication")
    await issuer.authorize(tenant_id, SecretStr(parts[1]))
    try:
        context = await bots.resolve(bot_id)
        if context.tenant_id != tenant_id:
            raise ContextDenied()
    except ContextDenied:
        raise HTTPException(404, "Scope not found") from None
    except ContextUnavailable:
        raise HTTPException(503, "Authorization unavailable") from None


router = APIRouter(prefix="/api/v1/tenants/{tenant_id}/bots/{bot_id}/ai", tags=["platform AI"],
                   dependencies=[Depends(authorize_scope)])


@router.get("/jobs", response_model=JobPage)
def jobs(tenant_id: UUID, bot_id: UUID, cursor: UUID | None = None,
         limit: int = Query(50, ge=1, le=100), db: Session = Depends(get_db)) -> JobPage:
    return list_jobs(db, tenant_id, bot_id, cursor, limit)


@router.get("/usage", response_model=UsageView)
def usage(tenant_id: UUID, bot_id: UUID, start: datetime, end: datetime,
          db: Session = Depends(get_db)) -> UsageView:
    if (start.utcoffset() is None or end.utcoffset() is None
        or not timedelta(0) < end - start <= timedelta(days=31)):
        raise HTTPException(422, "A timezone-aware interval of at most 31 days is required")
    return read_usage(db, tenant_id, bot_id, start, end)
