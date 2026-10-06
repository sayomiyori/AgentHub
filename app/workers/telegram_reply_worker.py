import logging
from uuid import UUID

from celery import Celery

from app.config import get_settings
from app.platform.replies import publish_reply

logger = logging.getLogger(__name__)
celery_app = Celery("telegram_replies", broker=get_settings().redis_url)
celery_app.conf.update(task_default_queue="telegram_replies", worker_prefetch_multiplier=1)


@celery_app.task(name="platform.publish_telegram_reply", soft_time_limit=25, time_limit=30,
                 acks_late=True, reject_on_worker_lost=True, max_retries=0, ignore_result=True)
def publish_telegram_reply(outbox_id: str) -> None:
    try:
        parsed = UUID(outbox_id)
    except (TypeError, ValueError, AttributeError):
        logger.warning("Invalid Telegram answer notification")
        return
    try:
        publish_reply(parsed)
    except Exception:
        logger.warning("Telegram answer publication unavailable")
