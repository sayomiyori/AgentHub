"""Authenticate raw bytes, authorize scope and durably admit; no generation here."""

import hashlib
import hmac
import json
import logging
import re
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from app.db.session import SessionLocal
from app.platform.bot_context import BotContextClient, ContextDenied, ContextUnavailable
from app.platform.config import PlatformSettings, get_platform_settings
from app.platform.jobs import JobConflict, enqueue_job, persist_admission
from app.platform.schemas import IngressEnvelope, JobReceipt, strict_json

router = APIRouter(prefix="/internal/v1/telegram", tags=["internal Telegram"])
MAX_BODY_BYTES = 1024 * 1024
logger = logging.getLogger(__name__)


def get_context_client(config: PlatformSettings = Depends(get_platform_settings)) -> BotContextClient:  # noqa: B008
    return BotContextClient(config)


def get_session_factory() -> Callable[[], Session]:
    return SessionLocal


def get_job_notifier():
    return enqueue_job


@router.post("/updates", response_model=JobReceipt, status_code=202)
async def telegram_updates(
    request: Request,
    response: Response,
    config: PlatformSettings = Depends(get_platform_settings),  # noqa: B008
    context_client: BotContextClient = Depends(get_context_client),  # noqa: B008
    sessions: Callable[[], Session] = Depends(get_session_factory),  # noqa: B008
    notifier=Depends(get_job_notifier),
) -> JobReceipt:  # noqa: B008
    if not config.telegram_ai_enabled:
        raise HTTPException(503, "Telegram AI is disabled")
    signatures = request.headers.getlist("X-Webhook-Signature")
    if len(signatures) != 1 or not re.fullmatch(r"sha256=[0-9a-f]{64}", signatures[0]):
        raise HTTPException(401, "Invalid signature")
    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Telegram event is too large")
            body.extend(chunk)
    except ClientDisconnect:
        raise HTTPException(400, "Incomplete Telegram event") from None
    expected = "sha256=" + hmac.new(config.ingress_key.get_secret_value().encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signatures[0], expected):
        raise HTTPException(401, "Invalid signature")
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json"
        or request.headers.get("content-encoding", "identity") != "identity"
    ):
        raise HTTPException(415, "Unsupported Telegram content")
    try:
        decoded = strict_json(bytes(body))
        envelope = IngressEnvelope.model_validate_json(json.dumps(decoded))
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(422, "Invalid Telegram event") from None
    try:
        context = await context_client.resolve(envelope.bot_id)
        if context.tenant_id != envelope.tenant_id:
            raise ContextDenied()
    except ContextDenied:
        raise HTTPException(403, "Bot context denied") from None
    except ContextUnavailable:
        raise HTTPException(503, "Bot context unavailable") from None
    digest = hashlib.sha256(
        json.dumps(envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    try:
        receipt, created = await run_in_threadpool(persist_admission, sessions, envelope, digest)
    except JobConflict:
        raise HTTPException(409, "Telegram event conflict") from None
    except SQLAlchemyError:
        raise HTTPException(503, "Job storage unavailable") from None
    # A lost notification is safe: the durable pending row is the recovery source.
    try:
        await run_in_threadpool(notifier, receipt.job_id)
    except Exception:
        logger.warning("Telegram job notification unavailable")
    response.status_code = 202 if created else 200
    return receipt
