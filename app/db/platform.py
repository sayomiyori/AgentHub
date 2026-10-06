from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import DeclarativeBase


class PlatformBase(DeclarativeBase):
    """Separate metadata prevents legacy create_all from creating platform data."""


PLATFORM_REVISION = "002_telegram_ai"


def require_platform_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        if not inspect(connection).has_table("alembic_version"):
            raise RuntimeError("Platform migrations are required")
        revisions = set(connection.execute(text("SELECT version_num FROM alembic_version")).scalars())
        if revisions != {PLATFORM_REVISION}:
            raise RuntimeError("Platform migrations are required")
