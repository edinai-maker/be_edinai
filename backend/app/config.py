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

    # API Settings
    api_v1_prefix: str = "/api/v1"

    # File Uploads
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "application/pdf"]

    # AWS S3 Configuration
    aws_access_key_id: str = Field("", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("ap-south-1", env="AWS_REGION")
    aws_s3_bucket_name: str = Field("edinai-storage", env="AWS_S3_BUCKET_NAME")
    s3_enabled: bool = Field(True, env="S3_ENABLED")

    # Model Config (initial, later override below)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @validator("database_url", pre=True)
    def assemble_db_connection(cls, v: str | PostgresDsn) -> str | PostgresDsn:
        if isinstance(v, str) and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg2://", 1)
        return v

    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    # Token / auth / misc settings
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    algorithm: str = Field("HS256", env="ALGORITHM")
    allowed_email_domains_raw: str = Field("gmail.com", env="ALLOWED_EMAIL_DOMAINS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")
    fernet_key: Optional[str] = Field(None, env="FERNET_KEY")
    default_language: str = Field("English", env="DEFAULT_LANGUAGE")
    default_lecture_duration: int = Field(45, env="DEFAULT_LECTURE_DURATION")
    dev_admin_email: Optional[str] = Field("dev_admin@inai.dev", env="DEV_ADMIN_EMAIL")
    dev_admin_password: Optional[str] = Field("DevAdmin@123", env="DEV_ADMIN_PASSWORD")
    dev_admin_name: str = Field("Dev Admin", env="DEV_ADMIN_NAME")
    dev_admin_package: str = Field("trial", env="DEV_ADMIN_PACKAGE")
    dev_admin_expiry_days: int = Field(365, env="DEV_ADMIN_EXPIRY_DAYS")
    password_reset_url: Optional[str] = Field(None, env="PASSWORD_RESET_URL")

    # 🔹 SMTP / Email Settings (yahi naya add kiya gaya hai)
    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: int = Field(587, env="SMTP_PORT")
    email_sender: Optional[str] = Field(None, env="EMAIL_SENDER")
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(True, env="SMTP_USE_TLS")
    smtp_timeout: int = Field(30, env="SMTP_TIMEOUT")

    public_base_url: Optional[str] = Field(
        None,
        env="PUBLIC_BASE_URL",
        description="Base URL (e.g., https://example.com) used when constructing absolute media links",
    )
    gcp_tts_credentials_path: Optional[str] = Field(
        "/opt/app/json.production",
        env="GCP_TTS_CREDENTIALS_PATH",
        description="Absolute path to the Google Cloud Text-to-Speech service account JSON.",
    )
    groq_api_key: Optional[str] = Field(
        None,
        env="GROQ_API_KEY",
        description="API key for Groq AI service",
    )
    runway_api_key: Optional[str] = Field(
        None,
        env="RUNWAY_API_KEY",
        description="API key for Runway text-to-video service",
    )
    runway_enabled: bool = Field(
        False,
        env="RUNWAY_ENABLED",
        description="Toggle Runway media generation features on/off",
    )
    runway_text_to_video_url: str = Field(
        "https://api.runwayml.com/v1/text_to_video",
        env="RUNWAY_TEXT_TO_VIDEO_URL",
        description="Endpoint for Runway text-to-video generations",
    )
    runway_tasks_base_url: str = Field(
        "https://api.runwayml.com/v1/tasks",
        env="RUNWAY_TASKS_BASE_URL",
        description="Base endpoint for Runway task polling",
    )
    runway_api_version: str = Field(
        "2024-11-06",
        env="RUNWAY_API_VERSION",
        description="X-Runway-Version header value",
    )
    runway_text_to_video_model: str = Field(
        "veo3",
        env="RUNWAY_TEXT_TO_VIDEO_MODEL",
        description="Default Runway text-to-video model identifier",
    )
    runway_text_to_video_ratio: str = Field(
        "854:480",
        env="RUNWAY_TEXT_TO_VIDEO_RATIO",
        description="Default Runway video aspect ratio",
    )
    runway_text_to_video_duration: int = Field(
        60,
        env="RUNWAY_TEXT_TO_VIDEO_DURATION",
        description="Default Runway video duration in seconds",
    )
    runway_text_to_image_url: str = Field(
        "https://api.runwayml.com/v1/text_to_image",
        env="RUNWAY_TEXT_TO_IMAGE_URL",
        description="Endpoint for Runway text/image-to-image generations",
    )
    runway_text_to_image_model: str = Field(
        "gen4_image",
        env="RUNWAY_TEXT_TO_IMAGE_MODEL",
        description="Default Runway text-to-image model identifier",
    )
    runway_text_to_image_ratio: str = Field(
        "1360:768",
        env="RUNWAY_TEXT_TO_IMAGE_RATIO",
        description="Default Runway image aspect ratio",
    )

    topic_extract_max_workers: int = Field(1, env="TOPIC_EXTRACT_MAX_WORKERS")
    topic_extract_queue_limit: int = Field(0, env="TOPIC_EXTRACT_QUEUE_LIMIT")
    topic_extract_queue_timeout_seconds: int = Field(
        0,
        env="TOPIC_EXTRACT_QUEUE_TIMEOUT_SECONDS",
    )
    topic_extract_queue_backend: Literal["memory", "redis"] = Field(
        "memory",
        env="TOPIC_EXTRACT_QUEUE_BACKEND",
    )
    topic_extract_queue_poll_interval_ms: int = Field(
        500,
        env="TOPIC_EXTRACT_QUEUE_POLL_INTERVAL_MS",
    )
    topic_extract_queue_lease_seconds: int = Field(
        900,
        env="TOPIC_EXTRACT_QUEUE_LEASE_SECONDS",
    )
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, env="REDIS_SSL")

    # Final model_config (server env file)
    model_config = SettingsConfigDict(
        env_file="/opt/app/env.production",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @property
    def allowed_email_domains(self) -> List[str]:
        return [
            domain.strip().lower()
            for domain in self.allowed_email_domains_raw.split(",")
            if domain.strip()
        ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: List[str] | str) -> List[str]:
        if isinstance(value, str):
            if value.strip() == "*":
                return ["*"]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance so values are computed once."""
    return Settings()


settings = get_settings()