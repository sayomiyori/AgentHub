from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.llm_usage import LLMUsageRecord

logger = logging.getLogger(__name__)


def record_llm_call(
    db: Session,
    *,
    conversation_id: int | None,
    message_id: int | None,
    request_id: str | None,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    total = input_tokens + output_tokens
    row = LLMUsageRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        request_id=request_id,
        provider=provider.lower(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cost_usd=cost_usd,
    )
    db.add(row)
    try:
        db.flush()
    except Exception as exc:
        logger.warning("Failed to flush LLM usage record: %s", exc)


def attach_message_id(db: Session, request_id: str, message_id: int) -> None:
    if not request_id:
        return
    db.query(LLMUsageRecord).filter(LLMUsageRecord.request_id == request_id).update(
        {LLMUsageRecord.message_id: message_id},
        synchronize_session=False,
    )
    try:
        db.flush()
    except Exception as exc:
        logger.warning("Failed to attach message_id to usage rows: %s", exc)


def get_usage_stats(db: Session) -> dict[str, Any]:
    total_cost = float(db.query(func.sum(LLMUsageRecord.cost_usd)).scalar() or 0)
    total_tokens = int(db.query(func.sum(LLMUsageRecord.total_tokens)).scalar() or 0)

    by_provider_rows = (
        db.query(LLMUsageRecord.provider, func.sum(LLMUsageRecord.cost_usd))
        .group_by(LLMUsageRecord.provider)
        .all()
    )
    cost_by_provider = {p: float(c or 0) for p, c in by_provider_rows}

    by_model_rows = (
        db.query(LLMUsageRecord.model, func.sum(LLMUsageRecord.cost_usd))
        .group_by(LLMUsageRecord.model)
        .all()
    )
    cost_by_model = {m: float(c or 0) for m, c in by_model_rows}

    day_expr = func.date_trunc("day", LLMUsageRecord.created_at)
    by_day_rows = (
        db.query(day_expr, func.sum(LLMUsageRecord.cost_usd))
        .group_by(day_expr)
        .order_by(day_expr)
        .all()
    )
    cost_by_day: list[dict[str, Any]] = []
    for day, cost in by_day_rows:
        if day is None:
            continue
        day_str = day.date().isoformat() if hasattr(day, "date") else str(day)
        cost_by_day.append({"day": day_str, "cost_usd": float(cost or 0)})

    return {
        "total_cost_usd": float(total_cost or 0),
        "total_tokens": int(total_tokens or 0),
        "cost_by_provider": cost_by_provider,
        "cost_by_model": cost_by_model,
        "cost_by_day": cost_by_day,
    }
