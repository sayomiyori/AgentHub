"""Explicit frozen schema snapshot."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "002_telegram_ai"
down_revision = "001_legacy_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_ai_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("event_id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("bot_id", sa.UUID(), nullable=False),
        sa.Column("update_id", sa.BigInteger(), nullable=False),
        sa.Column("envelope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("digest", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("claim_id", sa.UUID(), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("call_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "(state = 'processing' AND claim_id IS NOT NULL AND lease_until IS NOT NULL) "
            "OR (state <> 'processing' AND claim_id IS NULL AND lease_until IS NULL)",
            name="ck_telegram_ai_claim",
        ),
        sa.CheckConstraint(
            "state <> 'completed' OR (answer IS NOT NULL AND length(answer) BETWEEN 1 AND 4096 "
            "AND provider IS NOT NULL AND model IS NOT NULL)",
            name="ck_telegram_ai_result",
        ),
        sa.CheckConstraint(
            "state IN ('pending','processing','completed','failed','unknown')", name="ck_telegram_ai_state"
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_telegram_ai_attempts"),
        sa.CheckConstraint("length(digest) = 64", name="ck_telegram_ai_digest"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("bot_id", "update_id", name="uq_telegram_ai_bot_update"),
        sa.UniqueConstraint("event_id", name="uq_telegram_ai_event"),
        sa.UniqueConstraint("id", "tenant_id", "bot_id", name="uq_telegram_ai_job_scope"),
    )
    op.create_index("ix_telegram_ai_due", "telegram_ai_jobs", ["state", "next_attempt_at", "id"], unique=False)
    op.create_index("ix_telegram_ai_scope", "telegram_ai_jobs", ["tenant_id", "bot_id", "id"], unique=False)
    op.create_table(
        "telegram_ai_usage",
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("bot_id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(precision=14, scale=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND estimated_cost_usd >= 0 "
            "AND estimated_cost_usd < 'Infinity'::numeric",
            name="ck_telegram_ai_usage_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id", "bot_id"],
            ["telegram_ai_jobs.id", "telegram_ai_jobs.tenant_id", "telegram_ai_jobs.bot_id"],
            name="fk_telegram_ai_usage_scope",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(
        "ix_telegram_ai_usage_scope", "telegram_ai_usage", ["tenant_id", "bot_id", "created_at"], unique=False
    )
    op.create_table(
        "telegram_reply_outbox",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("bot_id", sa.UUID(), nullable=False),
        sa.Column("envelope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("state", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("claim_id", sa.UUID(), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_id", sa.UUID(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "(state = 'processing' AND claim_id IS NOT NULL AND lease_until IS NOT NULL) "
            "OR (state <> 'processing' AND claim_id IS NULL AND lease_until IS NULL)",
            name="ck_telegram_reply_claim",
        ),
        sa.CheckConstraint("state <> 'published' OR delivery_id IS NOT NULL", name="ck_telegram_reply_delivery"),
        sa.CheckConstraint(
            "state IN ('pending','processing','published','failed','cancelled')", name="ck_telegram_reply_state"
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_telegram_reply_attempts"),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id", "bot_id"],
            ["telegram_ai_jobs.id", "telegram_ai_jobs.tenant_id", "telegram_ai_jobs.bot_id"],
            name="fk_telegram_reply_scope",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", name="uq_telegram_reply_job"),
    )
    op.create_index("ix_telegram_reply_due", "telegram_reply_outbox", ["state", "next_attempt_at", "id"], unique=False)
    op.create_index("ix_telegram_reply_scope", "telegram_reply_outbox", ["tenant_id", "bot_id", "id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_telegram_reply_scope", table_name="telegram_reply_outbox")
    op.drop_index("ix_telegram_reply_due", table_name="telegram_reply_outbox")
    op.drop_table("telegram_reply_outbox")
    op.drop_index("ix_telegram_ai_usage_scope", table_name="telegram_ai_usage")
    op.drop_table("telegram_ai_usage")
    op.drop_index("ix_telegram_ai_scope", table_name="telegram_ai_jobs")
    op.drop_index("ix_telegram_ai_due", table_name="telegram_ai_jobs")
    op.drop_table("telegram_ai_jobs")
