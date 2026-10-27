from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import DeclarativeBase


class PlatformBase(DeclarativeBase):
    """Separate metadata prevents legacy create_all from creating platform data."""


PLATFORM_REVISION = "004_merge_ai_resources"


def require_resource_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        schema = inspect(connection)
        for table, required in (("documents", {"tenant_id", "content_revision"}),
                                ("conversations", {"tenant_id"}), ("llm_usage_records", {"tenant_id"})):
            if schema.has_table(table) and not required.issubset({x["name"] for x in schema.get_columns(table)}):
                raise RuntimeError("Resource migrations are required")


def require_platform_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        if not inspect(connection).has_table("alembic_version"):
            raise RuntimeError("Platform migrations are required")
        revisions = set(connection.execute(text("SELECT version_num FROM alembic_version")).scalars())
        if revisions != {PLATFORM_REVISION}:
            raise RuntimeError("Platform migrations are required")
