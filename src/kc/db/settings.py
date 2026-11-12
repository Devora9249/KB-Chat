"""Database settings, read from environment variables or the local `.env` file.

The connection URL contains a password, so it is stored as SecretStr: it is masked in
repr/logs and must be read explicitly with `.get_secret_value()`.
"""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_SCHEME = "postgresql+psycopg://"


class DbSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,  # never echo DATABASE_URL (with its password) in errors
    )

    database_url: SecretStr = Field(validation_alias="DATABASE_URL")
    db_echo_sql: bool = Field(default=False, validation_alias="DB_ECHO_SQL")
    db_pool_size: int = Field(default=5, ge=1, validation_alias="DB_POOL_SIZE")

    @field_validator("database_url")
    @classmethod
    def _require_async_psycopg_driver(cls, value: SecretStr) -> SecretStr:
        # We use SQLAlchemy's async engine with psycopg 3; other drivers would fail later
        # with a less clear error, so fail fast here (without echoing the secret).
        if not value.get_secret_value().startswith(REQUIRED_SCHEME):
            raise ValueError(f"DATABASE_URL must start with '{REQUIRED_SCHEME}'")
        return value


def load_db_settings() -> DbSettings:
    """Load settings from the environment / `.env`.

    mypy cannot see that pydantic-settings fills required fields from the environment,
    so the single type-ignore lives here instead of at every call site.
    """
    return DbSettings()  # type: ignore[call-arg]
