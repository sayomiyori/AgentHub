"""PostgreSQL admission; unique identities arbitrate concurrent writers."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.telegram_job import TelegramAIJob
from app.models.telegram_reply import TelegramReplyOutbox
from app.models.telegram_usage import TelegramAIUsage
from app.platform.config import get_platform_settings
from app.platform.schemas import IngressEnvelope, JobReceipt
from app.services.llm.base import LLMResponse


class JobConflict(Exception):
    pass


@dataclass(frozen=True)
class JobClaim:
    job_id: UUID
    claim_id: UUID
    tenant_id: UUID
    bot_id: UUID
    event_id: UUID
    envelope: dict
    model: str


def _release(job: TelegramAIJob, now: datetime) -> None:
    job.claim_id = None
    job.lease_until = None
    job.updated_at = now


def _db_now(db: Session) -> datetime:
    return db.execute(select(func.clock_timestamp())).scalar_one()


def claim_job(db: Session, job_id: UUID) -> JobClaim | None:
    config = get_platform_settings()
    if not config.telegram_ai_enabled:
        return None
    job = db.scalar(select(TelegramAIJob).where(
        TelegramAIJob.id == job_id, TelegramAIJob.state == "pending",
        TelegramAIJob.next_attempt_at <= func.clock_timestamp(),
    ).with_for_update(skip_locked=True))
    if job is None:
        return None
    now = _db_now(db)
    if job.attempts >= 5 or job.call_started_at is not None:
        job.state = "unknown" if job.call_started_at is not None else "failed"
        job.error_code = "attempts_exhausted"
        _release(job, now)
        return None
    job.state = "processing"
    job.claim_id = uuid4()
    job.lease_until = now + timedelta(seconds=60)
    job.attempts += 1
    job.provider = job.provider or config.provider
    job.model = job.model or config.model
    job.updated_at = now
    job.error_code = None
    return JobClaim(job.id, job.claim_id, job.tenant_id, job.bot_id,
                    job.event_id, job.envelope, job.model)


def _owned_job(db: Session, claim: JobClaim) -> TelegramAIJob | None:
    job = db.scalar(select(TelegramAIJob).where(
        TelegramAIJob.id == claim.job_id, TelegramAIJob.tenant_id == claim.tenant_id,
        TelegramAIJob.bot_id == claim.bot_id, TelegramAIJob.claim_id == claim.claim_id,
        TelegramAIJob.state == "processing", TelegramAIJob.lease_until > func.clock_timestamp(),
    ).with_for_update())
    # Recheck after any row-lock wait, using the database clock again.
    if job is not None and job.lease_until is not None and job.lease_until > _db_now(db):
        return job
    return None


def mark_call_started(db: Session, claim: JobClaim) -> bool:
    job = _owned_job(db, claim)
    if job is None or job.call_started_at is not None:
        return False
    now = _db_now(db)
    job.call_started_at = now
    job.updated_at = now
    return True


def finish_job(db: Session, claim: JobClaim, *, state: str, code: str, started: bool) -> bool:
    job = _owned_job(db, claim)
    if job is None or (job.call_started_at is not None) != started:
        return False
    now = _db_now(db)
    job.state = "failed" if state == "pending" and job.attempts >= 5 else state
    job.error_code = code
    job.next_attempt_at = now + timedelta(seconds=min(2**job.attempts, 60))
    _release(job, now)
    return True


def complete_job(db: Session, claim: JobClaim, result: LLMResponse) -> bool:
    job = _owned_job(db, claim)
    if job is None or job.call_started_at is None:
        return False
    if result.provider != job.provider or result.model != job.model:
        raise ValueError("Completion scope mismatch")
    now = _db_now(db)
    db.add(TelegramAIUsage(
        job_id=job.id, tenant_id=job.tenant_id, bot_id=job.bot_id,
        provider=result.provider, model=result.model, input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens, estimated_cost_usd=Decimal(str(result.usage.cost_usd)),
    ))
    answer_event_id = uuid4()
    db.add(TelegramReplyOutbox(
        id=uuid4(), job_id=job.id, tenant_id=job.tenant_id, bot_id=job.bot_id,
        envelope={
            "event_id": str(answer_event_id), "event_type": "telegram.answer.created",
            "schema_version": 1, "occurred_at": now.astimezone(UTC).isoformat(),
            "tenant_id": str(job.tenant_id), "bot_id": str(job.bot_id),
            "correlation_id": job.envelope["correlation_id"],
            "idempotency_key": f"telegram-answer:{job.event_id}",
            "payload": {"ingress_event_id": str(job.event_id), "job_id": str(job.id), "text": result.content},
        },
    ))
    job.answer = result.content
    job.state = "completed"
    job.error_code = None
    _release(job, now)
    db.flush()
    return True


def recover_due_jobs(db: Session, batch_size: int = 100) -> list[UUID]:
    if not 1 <= batch_size <= 100:
        raise ValueError("Invalid scanner batch size")
    jobs = db.scalars(select(TelegramAIJob).where(or_(
        (TelegramAIJob.state == "pending") & (TelegramAIJob.next_attempt_at <= func.clock_timestamp()),
        (TelegramAIJob.state == "processing") & (TelegramAIJob.lease_until <= func.clock_timestamp()),
    )).order_by(TelegramAIJob.next_attempt_at, TelegramAIJob.id).limit(batch_size)
        .with_for_update(skip_locked=True))
    now = _db_now(db)
    due = []
    for job in jobs:
        if job.state == "processing":
            job.state = ("unknown" if job.call_started_at is not None else
                         "failed" if job.attempts >= 5 else "pending")
            job.error_code = "claim_expired"
            job.next_attempt_at = now + timedelta(seconds=min(2**job.attempts, 60))
            _release(job, now)
        elif job.call_started_at is not None or job.attempts >= 5:
            job.state = "unknown" if job.call_started_at is not None else "failed"
            job.error_code = "attempts_exhausted"
            _release(job, now)
        else:
            due.append(job.id)
    return due


def admit_job(db: Session, envelope: IngressEnvelope, digest: str) -> tuple[TelegramAIJob, bool]:
    job_id = uuid4()
    created = db.scalar(
        insert(TelegramAIJob)
        .values(
            id=job_id,
            event_id=envelope.event_id,
            tenant_id=envelope.tenant_id,
            bot_id=envelope.bot_id,
            update_id=envelope.payload.update_id,
            envelope=envelope.model_dump(mode="json"),
            digest=digest,
        )
        .on_conflict_do_nothing()
        .returning(TelegramAIJob)
    )
    if created is not None:
        return created, True
    matches = list(
        db.scalars(
            select(TelegramAIJob).where(
                TelegramAIJob.tenant_id == envelope.tenant_id,
                TelegramAIJob.bot_id == envelope.bot_id,
                or_(
                    TelegramAIJob.event_id == envelope.event_id,
                    (TelegramAIJob.bot_id == envelope.bot_id) & (TelegramAIJob.update_id == envelope.payload.update_id),
                )
            )
        )
    )
    if len(matches) != 1:
        raise JobConflict()
    existing = matches[0]
    if (
        existing.event_id != envelope.event_id
        or existing.tenant_id != envelope.tenant_id
        or existing.bot_id != envelope.bot_id
        or existing.update_id != envelope.payload.update_id
        or existing.digest != digest
    ):
        raise JobConflict()
    return existing, False


def persist_admission(
    factory: Callable[[], Session], envelope: IngressEnvelope, digest: str
) -> tuple[JobReceipt, bool]:
    with factory() as db:
        job, created = admit_job(db, envelope, digest)
        receipt = JobReceipt.model_validate(dict(event_id=job.event_id, job_id=job.id, state=job.state))
        db.commit()
        return receipt, created


def enqueue_job(job_id: UUID) -> None:
    enqueue_platform_task("platform.process_telegram_job", job_id, "telegram_ai")


def enqueue_platform_task(name: str, identity: UUID, queue: str) -> None:
    from celery import Celery

    from app.config import get_settings

    # Dedicated queue avoids exposing unfinished jobs to the legacy embedding worker.
    app = Celery("telegram_admission", broker=get_settings().redis_url)
    app.conf.update(
        broker_connection_timeout=2,
        task_publish_retry=False,
        broker_transport_options={
            "socket_connect_timeout": 2,
            "socket_timeout": 2,
            "retry_on_timeout": False,
            "max_retries": 0,
        },
    )
    try:
        app.send_task(name, args=[str(identity)], queue=queue, retry=False)
    finally:
        app.close()
