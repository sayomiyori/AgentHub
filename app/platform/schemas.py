"""Version-one ingress contract; no Telegram or provider credentials."""

import json
from datetime import timedelta
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

TelegramId = Annotated[StrictInt, Field(ge=-(2**63), le=2**63 - 1)]


def strict_json(body: bytes) -> object:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(body.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


class MessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    update_id: TelegramId
    chat_id: TelegramId
    message_id: Annotated[StrictInt, Field(gt=0, le=2**63 - 1)]
    question: Annotated[StrictStr, Field(min_length=1, max_length=4096)]

    @field_validator("question")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Empty question")
        return value


class IngressEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: UUID
    event_type: Literal["telegram.message.received"]
    schema_version: Literal[1]
    occurred_at: AwareDatetime
    tenant_id: UUID
    bot_id: UUID
    correlation_id: UUID
    idempotency_key: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    payload: MessagePayload

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Invalid version")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "IngressEnvelope":
        if self.idempotency_key != f"telegram:{self.bot_id}:{self.payload.update_id}":
            raise ValueError("Invalid identity")
        if self.occurred_at.utcoffset() is None or self.occurred_at.utcoffset() != timedelta(0):
            raise ValueError("UTC timestamp required")
        return self


class JobReceipt(BaseModel):
    event_id: UUID
    job_id: UUID
    state: Literal["pending", "processing", "completed", "failed", "unknown"]
