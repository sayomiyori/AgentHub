from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKeyConstraint, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.platform import PlatformBase


class AIModelSetting(PlatformBase):
    __tablename__ = "ai_model_settings"
    tenant_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    target: Mapped[str] = mapped_column(String(6), primary_key=True)
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    model: Mapped[str | None] = mapped_column(String(128))
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_by: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    __table_args__ = (
        CheckConstraint("target IN ('tenant','bot') AND (target <> 'tenant' OR subject_id = tenant_id)",
                        name="ck_ai_model_scope"),
        CheckConstraint("revision > 0", name="ck_ai_model_revision"),
        CheckConstraint("model IS NULL OR model ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$'",
                        name="ck_ai_model_identifier"),
    )


class AIModelSettingAudit(PlatformBase):
    __tablename__ = "ai_model_settings_audit"
    tenant_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    target: Mapped[str] = mapped_column(String(6), primary_key=True)
    subject_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str | None] = mapped_column(String(128))
    updated_by: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(["tenant_id", "target", "subject_id"],
                             ["ai_model_settings.tenant_id", "ai_model_settings.target",
                              "ai_model_settings.subject_id"],
                             ondelete="RESTRICT", name="fk_ai_model_audit_scope"),
        CheckConstraint("revision > 0", name="ck_ai_model_audit_revision"),
        CheckConstraint("model IS NULL OR model ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$'",
                        name="ck_ai_model_audit_identifier"),
    )
