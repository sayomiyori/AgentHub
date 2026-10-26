import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import SecretStr, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.v1.platform_reads import get_access_client, get_bot_client
from app.db.session import get_db
from app.platform.ai_settings import SettingsConflict, SettingsInput, SettingsView, save_settings, settings_view
from app.platform.bot_context import BotContextClient, ContextDenied, ContextUnavailable
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.schemas import strict_json
from app.platform.tenant_access import AuthorizedContext, TenantAccessClient

router = APIRouter(prefix="/api/v1/tenants/{tenant_id}", tags=["AI settings"])


async def access(tenant_id: UUID, request: Request,
                 config: PlatformSettings = Depends(get_platform_settings),
                 issuer: TenantAccessClient = Depends(get_access_client),
                 bots: BotContextClient = Depends(get_bot_client)) -> AuthorizedContext:
    if not config.read_enabled:
        raise HTTPException(404, "Not found")
    if request.query_params:
        raise HTTPException(400, "Invalid scope")
    values = request.headers.getlist("authorization")
    parts = values[0].split(" ") if len(values) == 1 else []
    if (len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1] or len(parts[1]) > 8192
            or not parts[1].isascii() or any(c.isspace() or not c.isprintable() for c in parts[1])):
        raise HTTPException(401, "Invalid authentication")
    context = await issuer.authorize(tenant_id, SecretStr(parts[1]),
                                     "ai.read" if request.method == "GET" else "ai.configure")
    if "bot_id" in request.path_params:
        try:
            bot = UUID(str(request.path_params['bot_id']))
        except ValueError:
            raise HTTPException(422, "Invalid scope") from None
        try:
            resolved = await bots.resolve(bot)
            if resolved.tenant_id != tenant_id:
                raise ContextDenied()
        except ContextDenied:
            raise HTTPException(404, "Scope not found") from None
        except ContextUnavailable:
            raise HTTPException(503, "Authorization unavailable") from None
    return context


async def input_body(request: Request, _context: AuthorizedContext = Depends(access)) -> SettingsInput:
    if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/json':
        raise HTTPException(400, "Invalid settings")
    try:
        async with asyncio.timeout(5):
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > 16384:
                    raise ValueError()
                raw.extend(chunk)
        return SettingsInput.model_validate(strict_json(bytes(raw)))
    except (ValueError, ValidationError, TimeoutError, UnicodeError, RecursionError):
        raise HTTPException(400, "Invalid settings") from None


def view(db: Session, tenant: UUID, bot: UUID | None, config: PlatformSettings) -> SettingsView:
    try:
        return settings_view(db, tenant, bot, config)
    except (ValueError, SQLAlchemyError):
        raise HTTPException(503, "Configuration unavailable") from None


def save(db: Session, tenant: UUID, bot: UUID | None, context: AuthorizedContext,
         value: SettingsInput, config: PlatformSettings) -> SettingsView:
    try:
        save_settings(db, tenant, bot, context.user_id, value)
        result = view(db, tenant, bot, config)
        db.commit()
        return result
    except SettingsConflict:
        db.rollback()
        raise HTTPException(409, "Settings changed; reload before saving") from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(503, "Configuration unavailable") from None
    except Exception:
        db.rollback()
        raise


@router.get('/ai/settings', response_model=SettingsView)
def tenant_settings(tenant_id: UUID, _context: AuthorizedContext = Depends(access),
                    config: PlatformSettings = Depends(get_platform_settings),
                    db: Session = Depends(get_db)) -> SettingsView:
    return view(db, tenant_id, None, config)


@router.get('/bots/{bot_id}/ai/settings', response_model=SettingsView)
def bot_settings(tenant_id: UUID, bot_id: UUID, _context: AuthorizedContext = Depends(access),
                 config: PlatformSettings = Depends(get_platform_settings),
                    db: Session = Depends(get_db)) -> SettingsView:
    return view(db, tenant_id, bot_id, config)


@router.put('/ai/settings', response_model=SettingsView)
def put_tenant_settings(tenant_id: UUID, context: AuthorizedContext = Depends(access),
                        value: SettingsInput = Depends(input_body),
                        config: PlatformSettings = Depends(get_platform_settings),
                        db: Session = Depends(get_db)) -> SettingsView:
    return save(db, tenant_id, None, context, value, config)


@router.put('/bots/{bot_id}/ai/settings', response_model=SettingsView)
def put_bot_settings(tenant_id: UUID, bot_id: UUID, context: AuthorizedContext = Depends(access),
                     value: SettingsInput = Depends(input_body),
                        config: PlatformSettings = Depends(get_platform_settings),
                     db: Session = Depends(get_db)) -> SettingsView:
    return save(db, tenant_id, bot_id, context, value, config)
