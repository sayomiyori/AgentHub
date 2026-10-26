"""Scoped model precedence and atomic versioned changes; never executes a job."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from sqlalchemy import Update, func, update
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import Session

from app.models.ai_settings import AIModelSetting, AIModelSettingAudit
from app.platform.config import MODEL_IDENTIFIER, PlatformSettings


class SettingsConflict(Exception):
    pass


class SettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str | None
    expected_revision: StrictInt = Field(ge=0, lt=2147483647)

    @field_validator("model")
    @classmethod
    def valid_model(cls, value: str | None) -> str | None:
        if value is not None and not MODEL_IDENTIFIER.fullmatch(value):
            raise ValueError("Invalid model identifier")
        return value


class SavedSetting(BaseModel):
    model: str | None
    revision: int
    updated_at: datetime | None


class EffectiveSetting(BaseModel):
    scope: Literal["platform", "tenant", "bot"]
    provider: Literal["groq"] = "groq"
    model: str | None


class SettingsView(BaseModel):
    tenant_id: UUID
    bot_id: UUID | None
    target: Literal["tenant", "bot"]
    saved: SavedSetting
    effective: EffectiveSetting
    processing_enabled: bool


def effective_model(db: Session, tenant: UUID, bot: UUID | None, config: PlatformSettings) -> EffectiveSetting:
    for target, subject in (("bot", bot), ("tenant", tenant)):
        if subject is None:
            continue
        row = db.get(AIModelSetting, (tenant, target, subject))
        if row is not None and row.model is not None:
            return EffectiveSetting(scope="bot" if target == "bot" else "tenant", model=row.model)
    if config.model and not MODEL_IDENTIFIER.fullmatch(config.model):
        raise ValueError("Invalid platform model")
    return EffectiveSetting(scope="platform", model=config.model or None)


def settings_view(db: Session, tenant: UUID, bot: UUID | None, config: PlatformSettings) -> SettingsView:
    row = db.get(AIModelSetting, (tenant, "tenant" if bot is None else "bot", tenant if bot is None else bot))
    return SettingsView(tenant_id=tenant, bot_id=bot, target="tenant" if bot is None else "bot",
                        saved=SavedSetting(model=row.model if row else None, revision=row.revision if row else 0,
                                           updated_at=row.updated_at if row else None),
                        effective=effective_model(db, tenant, bot, config),
                        processing_enabled=config.telegram_ai_enabled)


def save_settings(db: Session, tenant: UUID, bot: UUID | None, actor: UUID, value: SettingsInput) -> None:
    key = dict(tenant_id=tenant, target="tenant" if bot is None else "bot", subject_id=tenant if bot is None else bot)
    values = dict(model=value.model, updated_by=actor, updated_at=func.clock_timestamp())
    statement: Insert | Update
    if value.expected_revision == 0:
        statement = insert(AIModelSetting).values(**key, **values, revision=1).on_conflict_do_nothing()
    else:
        statement = update(AIModelSetting).where(
            AIModelSetting.tenant_id == tenant, AIModelSetting.target == key['target'],
            AIModelSetting.subject_id == key['subject_id'], AIModelSetting.revision == value.expected_revision,
        ).values(**values, revision=value.expected_revision + 1)
    changed = db.execute(statement.returning(AIModelSetting.revision, AIModelSetting.updated_at)).first()
    if changed is None:
        raise SettingsConflict()
    db.execute(insert(AIModelSettingAudit).values(**key, model=value.model, revision=changed.revision,
                                                updated_by=actor, updated_at=changed.updated_at))
    db.expire_all()
