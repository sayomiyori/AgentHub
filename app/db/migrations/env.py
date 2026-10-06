from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.config import get_settings
from app.db.platform import PlatformBase
from app.db.session import Base
from app.models import (  # noqa: F401
    chunk,
    conversation,
    document,
    llm_usage,
    telegram_job,
    telegram_reply,
    telegram_usage,
)

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
metadata = [Base.metadata, PlatformBase.metadata]


def run_migrations() -> None:
    url = get_settings().database_url
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=metadata, literal_binds=True, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations()
