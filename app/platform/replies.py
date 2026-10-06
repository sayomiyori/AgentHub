"""Idempotent signed publication; a receipt confirms admission, not sending."""

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.telegram_reply import TelegramReplyOutbox
from app.platform.config import get_platform_settings
from app.platform.jobs import _db_now, enqueue_platform_task
from app.platform.processor import _check_schema, _session_factory
from app.platform.schemas import strict_json

logger = logging.getLogger(__name__)


class AnswerReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: UUID
    delivery_id: UUID
    state: Literal["pending", "processing", "succeeded", "failed", "unknown", "cancelled"]


@dataclass(frozen=True)
class ReplyClaim:
    outbox_id: UUID
    claim_id: UUID
    tenant_id: UUID
    bot_id: UUID
    envelope: dict


def _release(row: TelegramReplyOutbox) -> None:
    row.claim_id = None
    row.lease_until = None


def claim_reply(db: Session, outbox_id: UUID) -> ReplyClaim | None:
    row = db.scalar(select(TelegramReplyOutbox).where(
        TelegramReplyOutbox.id == outbox_id, TelegramReplyOutbox.state == "pending",
        TelegramReplyOutbox.next_attempt_at <= func.clock_timestamp(),
    ).with_for_update(skip_locked=True))
    if row is None:
        return None
    now = _db_now(db)
    row.updated_at = now
    if row.attempts >= 10:
        row.state = "failed"
        row.error_code = "attempts_exhausted"
        return None
    row.state = "processing"
    row.claim_id = uuid4()
    row.lease_until = now + timedelta(seconds=60)
    row.attempts += 1
    row.error_code = None
    return ReplyClaim(row.id, row.claim_id, row.tenant_id, row.bot_id, row.envelope)


def finish_reply(db: Session, claim: ReplyClaim, *, state: str, code: str | None,
                 delivery_id: UUID | None = None, retry_after: int = 0) -> bool:
    row = db.scalar(select(TelegramReplyOutbox).where(
        TelegramReplyOutbox.id == claim.outbox_id, TelegramReplyOutbox.tenant_id == claim.tenant_id,
        TelegramReplyOutbox.bot_id == claim.bot_id, TelegramReplyOutbox.claim_id == claim.claim_id,
        TelegramReplyOutbox.state == "processing", TelegramReplyOutbox.lease_until > func.clock_timestamp(),
    ).with_for_update())
    now = _db_now(db)
    if row is None or row.lease_until is None or row.lease_until <= now:
        return False
    row.state = "failed" if state == "pending" and row.attempts >= 10 else state
    row.error_code = code
    row.delivery_id = delivery_id
    row.next_attempt_at = now + timedelta(seconds=min(max(2**row.attempts, retry_after), 60))
    row.updated_at = now
    _release(row)
    return True


async def _post(claim: ReplyClaim) -> tuple[str, str | None, UUID | None, int]:
    config = get_platform_settings()
    body = json.dumps(claim.envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    signature = hmac.new(config.reply_key.get_secret_value().encode("ascii"), body, hashlib.sha256).hexdigest()
    async with (
        asyncio.timeout(8),
        httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2), trust_env=False, follow_redirects=False) as client,
    ):
        async with client.stream("POST", f"{config.webhook_url.rstrip('/')}/internal/v1/telegram/answers",
            content=body, headers={"Content-Type": "application/json", "Accept-Encoding": "identity",
                                  "X-Webhook-Signature": f"sha256={signature}"}) as response:
            status = response.status_code
            if status == 429:
                value = response.headers.get("retry-after", "")
                delay = min(int(value), 60) if value.isascii() and value.isdecimal() and len(value) <= 9 else 0
                return "pending", "rate_limited", None, delay
            if status >= 500:
                return "pending", "remote_unavailable", None, 0
            if status in {400, 401, 403, 404, 415, 422}:
                return "failed", "publication_rejected", None, 0
            if status not in {200, 202, 409}:
                return "pending", "invalid_receipt", None, 0
            try:
                media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if (response.headers.get("content-encoding", "identity") != "identity"
                        or media_type != "application/json"):
                    raise ValueError("Invalid receipt encoding")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if len(data) + len(chunk) > 65536:
                        raise ValueError("Receipt too large")
                    data.extend(chunk)
                decoded = strict_json(bytes(data))
            except (httpx.HTTPError, TimeoutError, ValueError, UnicodeError, RecursionError, asyncio.CancelledError):
                if status == 409:
                    return "failed", "publication_conflict", None, 0
                raise
            if status == 409:
                if decoded == {"detail": "ingress_publication_not_ready"}:
                    return "pending", "ingress_not_ready", None, 0
                return "failed", "publication_conflict", None, 0
            receipt = AnswerReceipt.model_validate_json(json.dumps(decoded))
            if str(receipt.event_id) != claim.envelope["event_id"] or receipt.delivery_id.int == 0:
                raise ValueError("Receipt identity mismatch")
            return "published", None, receipt.delivery_id, 0


def publish_reply(outbox_id: UUID, *, factory: Callable[[], Session] | None = None) -> None:
    if not get_platform_settings().telegram_ai_enabled:
        return
    create_session = _session_factory(factory)
    with create_session() as db:
        _check_schema(db)
        claim = claim_reply(db, outbox_id)
        db.commit()
    if claim is None:
        return
    try:
        state, code, delivery_id, delay = asyncio.run(_post(claim))
    except Exception:
        state, code, delivery_id, delay = "pending", "publication_uncertain", None, 0
    try:
        with create_session() as db:
            finish_reply(db, claim, state=state, code=code, delivery_id=delivery_id, retry_after=delay)
            db.commit()
    except Exception:
        logger.warning("Telegram answer receipt persistence unavailable")


def enqueue_reply(outbox_id: UUID) -> None:
    enqueue_platform_task("platform.publish_telegram_reply", outbox_id, "telegram_replies")


def recover_replies_once(batch_size: int = 100, *, factory: Callable[[], Session] | None = None,
                         notify: Callable[[UUID], None] = enqueue_reply) -> int:
    if not 1 <= batch_size <= 100:
        raise ValueError("Invalid scanner batch size")
    if not get_platform_settings().telegram_ai_enabled:
        return 0
    with _session_factory(factory)() as db:
        _check_schema(db)
        rows = db.scalars(select(TelegramReplyOutbox).where(or_(
            (TelegramReplyOutbox.state == "pending") & (TelegramReplyOutbox.next_attempt_at <= func.clock_timestamp()),
            (TelegramReplyOutbox.state == "processing") & (TelegramReplyOutbox.lease_until <= func.clock_timestamp()),
        )).order_by(TelegramReplyOutbox.next_attempt_at, TelegramReplyOutbox.id).limit(batch_size)
            .with_for_update(skip_locked=True))
        now, due = _db_now(db), []
        for row in rows:
            if row.attempts >= 10:
                row.state = "failed"
                row.error_code = "attempts_exhausted"
                row.updated_at = now
                _release(row)
            elif row.state == "processing":
                row.state = "pending"
                row.error_code = "claim_expired"
                row.next_attempt_at = now + timedelta(seconds=min(2**row.attempts, 60))
                row.updated_at = now
                _release(row)
            else:
                due.append(row.id)
        db.commit()
    submitted = 0
    for outbox_id in due:
        try:
            notify(outbox_id)
            submitted += 1
        except Exception:
            logger.warning("Telegram answer notification unavailable")
            break
    return submitted
