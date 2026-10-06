"""PostgreSQL admission; unique identities arbitrate concurrent writers."""

from collections.abc import Callable
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.telegram_job import TelegramAIJob
from app.platform.schemas import IngressEnvelope, JobReceipt


class JobConflict(Exception):
    pass


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
        app.send_task("platform.process_telegram_job", args=[str(job_id)], queue="telegram_ai", retry=False)
    finally:
        app.close()
