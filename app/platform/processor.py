import asyncio
import json
import logging
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.db.platform import require_platform_schema
from app.platform.bot_context import BotContextClient, ContextDenied, ContextUnavailable
from app.platform.config import get_platform_settings
from app.platform.generation import (
    GenerationConfigurationError,
    GenerationInvalidResponse,
    GenerationRejectedError,
    generate_platform_answer,
    validate_generation_configuration,
)
from app.platform.jobs import (
    JobClaim,
    claim_job,
    complete_job,
    enqueue_job,
    finish_job,
    mark_call_started,
    recover_due_jobs,
)
from app.platform.schemas import IngressEnvelope

logger = logging.getLogger(__name__)


def _session_factory(factory: Callable[[], Session] | None) -> Callable[[], Session]:
    if factory is not None:
        return factory
    from app.db.session import SessionLocal
    return SessionLocal


def _check_schema(db: Session) -> None:
    bind = db.get_bind()
    if not isinstance(bind, Engine):
        raise RuntimeError("Platform engine is required")
    require_platform_schema(bind)


def _finish(factory: Callable[[], Session], claim: JobClaim, state: str, code: str, started: bool) -> None:
    try:
        with factory() as db:
            finish_job(db, claim, state=state, code=code, started=started)
            db.commit()
    except Exception:
        # The persisted marker/lease remains the authority for scanner recovery.
        logger.warning("Telegram job outcome persistence unavailable")


def process_job(job_id: UUID, *, factory: Callable[[], Session] | None = None) -> None:
    config = get_platform_settings()
    if not config.telegram_ai_enabled:
        return
    create_session = _session_factory(factory)
    with create_session() as db:
        _check_schema(db)
        claim = claim_job(db, job_id)
        db.commit()
    if claim is None:
        return
    started = False
    try:
        envelope = IngressEnvelope.model_validate_json(json.dumps(claim.envelope))
        if (envelope.tenant_id != claim.tenant_id or envelope.bot_id != claim.bot_id
                or envelope.event_id != claim.event_id):
            raise GenerationConfigurationError("Invalid stored job scope")
        validate_generation_configuration(claim.model)
        context = asyncio.run(BotContextClient(config).resolve(claim.bot_id))
        if context.tenant_id != claim.tenant_id:
            raise ContextDenied()
        with create_session() as db:
            if not mark_call_started(db, claim):
                return
            db.commit()
        started = True
        result = generate_platform_answer(envelope.payload.question, claim.model)
        with create_session() as db:
            complete_job(db, claim, result)
            db.commit()
    except ContextUnavailable:
        _finish(create_session, claim, "pending", "context_unavailable", started)
    except ContextDenied:
        _finish(create_session, claim, "failed", "context_denied", started)
    except (GenerationConfigurationError, GenerationInvalidResponse, GenerationRejectedError):
        _finish(create_session, claim, "failed", "generation_rejected", started)
    except ValueError:
        _finish(create_session, claim, "unknown" if started else "failed", "invalid_job", started)
    except Exception:
        _finish(create_session, claim, "unknown" if started else "pending",
                "generation_uncertain" if started else "precall_unavailable", started)


def recover_jobs_once(
    batch_size: int = 100, *, factory: Callable[[], Session] | None = None,
    notify: Callable[[UUID], None] = enqueue_job,
) -> int:
    if not get_platform_settings().telegram_ai_enabled:
        return 0
    with _session_factory(factory)() as db:
        _check_schema(db)
        ids = recover_due_jobs(db, batch_size)
        db.commit()
    submitted = 0
    for job_id in ids:
        try:
            notify(job_id)
            submitted += 1
        except Exception:
            logger.warning("Telegram job notification unavailable")
            break
    return submitted
