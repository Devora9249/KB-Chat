"""Alembic environment: async engine, URL from DATABASE_URL (never from alembic.ini)."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from kc.db.engine import create_engine
from kc.db.models import Base
from kc.db.settings import load_db_settings

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout / output buffer without connecting (`--sql`)."""
    context.configure(
        url=load_db_settings().database_url.get_secret_value(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_engine(load_db_settings())
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_sync)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    # Selector loop: psycopg async does not support Windows' default Proactor loop.
    asyncio.run(_run_async(), loop_factory=asyncio.SelectorEventLoop)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
