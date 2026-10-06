import logging
from uuid import UUID

from celery import Celery

from app.config import get_settings
from app.platform.processor import process_job

logger = logging.getLogger(__name__)
celery_app = Celery("telegram_platform", broker=get_settings().redis_url)
celery_app.conf.update(task_default_queue="telegram_ai", worker_prefetch_multiplier=1)


@celery_app.task(name="platform.process_telegram_job", soft_time_limit=25, time_limit=30,
                 acks_late=True, reject_on_worker_lost=True, max_retries=0, ignore_result=True)
def process_telegram_job(job_id: str) -> None:
    try:
        parsed = UUID(job_id)
    except (TypeError, ValueError, AttributeError):
        logger.warning("Invalid Telegram job notification")
        return
    try:
        process_job(parsed)
    except Exception:
        logger.warning("Telegram job processing unavailable")
