"""Centralized configuration for the modular backend example."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any, Literal

from pydantic import Field, field_validator, PostgresDsn, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App Settings
    app_name: str = Field("EDINAI Modular Backend", env="APP_NAME")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("production", env="ENVIRONMENT")

    # Database Configuration
    database_url: PostgresDsn = Field(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/inai",
        env="DATABASE_URL",
    )
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(10, env="DATABASE_MAX_OVERFLOW")
    database_pool_recycle: int = Field(3600, env="DATABASE_POOL_RECYCLE")
    database_pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")

    # Security
    secret_key: str = Field("your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(525600, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 1 year (365 * 24 * 60)
    access_token_expire_days: int = Field(365, env="ACCESS_TOKEN_EXPIRE_DAYS")  # 1 year

    # CORS
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
