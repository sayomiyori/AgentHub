"""Persist tenant defaults, bot overrides and immutable revision receipts."""

import sqlalchemy as sa
from alembic import op

revision = "003_ai_settings"
down_revision = "002_telegram_ai"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_model_settings",
        sa.Column("tenant_id", sa.UUID(), primary_key=True),
        sa.Column("target", sa.String(6), primary_key=True),
        sa.Column("subject_id", sa.UUID(), primary_key=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("updated_by", sa.UUID(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("target IN ('tenant','bot') AND (target <> 'tenant' OR subject_id = tenant_id)",
                           name="ck_ai_model_scope"),
        sa.CheckConstraint("revision > 0", name="ck_ai_model_revision"),
        sa.CheckConstraint("model IS NULL OR model ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$'",
                           name="ck_ai_model_identifier"),
    )
    op.create_table(
        "ai_model_settings_audit",
        sa.Column("tenant_id", sa.UUID(), primary_key=True),
        sa.Column("target", sa.String(6), primary_key=True),
        sa.Column("subject_id", sa.UUID(), primary_key=True),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("updated_by", sa.UUID(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id", "target", "subject_id"],
                                ["ai_model_settings.tenant_id", "ai_model_settings.target",
                              "ai_model_settings.subject_id"],
                                ondelete="RESTRICT", name="fk_ai_model_audit_scope"),
        sa.CheckConstraint("revision > 0", name="ck_ai_model_audit_revision"),
        sa.CheckConstraint("model IS NULL OR model ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$'",
                           name="ck_ai_model_audit_identifier"),
    )


def downgrade() -> None:
    op.drop_table("ai_model_settings_audit")
    op.drop_table("ai_model_settings")
