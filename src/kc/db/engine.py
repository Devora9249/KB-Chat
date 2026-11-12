"""Async SQLAlchemy engine and session factory.

Create ONE engine per process (it owns the connection pool) and pass it, or the session
factory, to the components that need it (dependency injection), instead of importing a
global. Creating the engine does not connect; the first query does.

Windows note: psycopg's async mode does not support the default Proactor event loop.
When running async DB code locally on Windows use:
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
Linux (Docker, CI) is not affected.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kc.db.settings import DbSettings


def create_engine(settings: DbSettings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url.get_secret_value(),
        echo=settings.db_echo_sql,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,  # transparently replace connections the server has dropped
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: objects stay readable after commit without a new query,
    # which avoids surprise lazy-loads (not allowed in async code).
    return async_sessionmaker(engine, expire_on_commit=False)
