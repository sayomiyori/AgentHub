"""Merge independently applied tenant resource and AI settings histories."""

revision = "004_merge_ai_resources"
down_revision = ("003_tenant_resources", "003_ai_settings")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
