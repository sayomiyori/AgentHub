"""Keep historical standalone data unassigned; scope new resource records."""

import sqlalchemy as sa
from alembic import op

revision = "003_tenant_resources"
down_revision = "002_telegram_ai"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("documents", "conversations", "llm_usage_records"):
        op.add_column(table, sa.Column("tenant_id", sa.UUID(), nullable=True))
        op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
    op.add_column("documents", sa.Column("content_revision", sa.UUID(), nullable=False,
                                        server_default=sa.text("gen_random_uuid()")))


def downgrade() -> None:
    op.drop_column("documents", "content_revision")
    for table in ("llm_usage_records", "conversations", "documents"):
        op.drop_index(f"ix_{table}_tenant_id", table_name=table)
        op.drop_column(table, "tenant_id")
