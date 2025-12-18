"""SQLAlchemy database configuration and session management."""
#database.py
from __future__ import annotations

from collections.abc import Generator
import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
import time
import random

from .config import settings

def _normalize_database_url(raw_url) -> str:
    """Convert database URL to SQLAlchemy format if needed."""
    if hasattr(raw_url, 'unicode_string'):
        # Handle Pydantic's PostgresDsn type
        url_str = raw_url.unicode_string()
    else:
        # Handle string URLs
        url_str = str(raw_url)
        
    # Ensure we're using psycopg2 driver
    if url_str.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url_str[len("postgresql://"):]
    if url_str.startswith("postgres://"):
        return "postgresql+psycopg2://" + url_str[len("postgres://"):]
    return url_str


logger = logging.getLogger(__name__)
_CHAPTER_SCHEMA_LOCK_ID = 0x4CAFEBABE


def _retry_on_deadlock(max_retries: int = 3, base_delay: float = 0.5):
    """Decorator to retry on database deadlock with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if "deadlock detected" not in str(e).lower():
                        raise
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(
                        "Deadlock detected in %s, retrying in %.2fs (attempt %d/%d)",
                        func.__name__, delay, attempt + 1, max_retries
                    )
                    time.sleep(delay)
        return wrapper
    return decorator

# Create database engine with connection pooling
database_url = _normalize_database_url(settings.database_url)
engine = create_engine(
    database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_recycle=settings.database_pool_recycle,
    pool_timeout=settings.database_pool_timeout,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """Provide a transactional scope for database operations."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@_retry_on_deadlock(max_retries=3, base_delay=0.5)
def _ensure_chapter_material_schema() -> None:
    """Ensure chapter_materials has columns required by the ORM model."""

    with engine.begin() as connection:
        # Acquire an advisory lock so only one worker performs schema updates at a time.
        connection.execute(
            text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": _CHAPTER_SCHEMA_LOCK_ID}
        )
        try:
            inspector = inspect(connection)
            if "chapter_materials" not in inspector.get_table_names():
                return

            columns = {col["name"]: col for col in inspector.get_columns("chapter_materials")}
            statements: list[str] = []

            optional_columns = {}

            for column_name, ddl in optional_columns.items():
                if column_name not in columns:
                    logger.info("Adding missing %s column to chapter_materials table", column_name)
                    statements.append(ddl)

            if "is_global" not in columns:
                logger.info("Adding missing is_global column to chapter_materials table")
                statements.append(
                    "ALTER TABLE chapter_materials ADD COLUMN is_global BOOLEAN DEFAULT FALSE"
                )

            if "updated_at" not in columns:
                logger.info("Adding missing updated_at column to chapter_materials table")
                statements.append(
                    "ALTER TABLE chapter_materials ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW()"
                )
            else:
                updated_default = columns["updated_at"].get("default")
                if not updated_default:
                    logger.info("Setting default for chapter_materials.updated_at")
                    statements.append(
                        "ALTER TABLE chapter_materials ALTER COLUMN updated_at SET DEFAULT NOW()"
                    )

            created_col = columns.get("created_at")
            if created_col and not created_col.get("default"):
                logger.info("Setting default for chapter_materials.created_at")
                statements.append(
                    "ALTER TABLE chapter_materials ALTER COLUMN created_at SET DEFAULT NOW()"
                )

            for stmt in statements:
                connection.execute(text(stmt))

            # Backfill any NULL timestamps and enforce NOT NULL on updated_at.
            connection.execute(
                text("UPDATE chapter_materials SET created_at = NOW() WHERE created_at IS NULL")
            )
            connection.execute(
                text(
                    "UPDATE chapter_materials SET updated_at = COALESCE(updated_at, created_at, NOW())"
                )
            )
            connection.execute(
                text("ALTER TABLE chapter_materials ALTER COLUMN updated_at SET NOT NULL")
            )
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": _CHAPTER_SCHEMA_LOCK_ID}
            )

@_retry_on_deadlock(max_retries=3, base_delay=0.5)
def _ensure_lecture_gen_core_columns() -> None:
    """Ensure essential columns exist on lecture_gen table."""
    inspector = inspect(engine)
    if "lecture_gen" not in inspector.get_table_names():
        return
    statements = [
        "ALTER TABLE lecture_gen ADD COLUMN IF NOT EXISTS lecture_title VARCHAR(255) DEFAULT '' NOT NULL",
        "ALTER TABLE lecture_gen ADD COLUMN IF NOT EXISTS lecture_link VARCHAR(512) DEFAULT '' NOT NULL",
        "ALTER TABLE lecture_gen ADD COLUMN IF NOT EXISTS lecture_shared BOOLEAN DEFAULT FALSE NOT NULL",
    ]
    with engine.begin() as connection:
        for stmt in statements:
            connection.execute(text(stmt))
    # Populate human-friendly defaults for legacy rows that received blank values.
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE lecture_gen
                SET lecture_title = COALESCE(
                    NULLIF(lecture_title, ''),
                    NULLIF(lecture_data->>'title', ''),
                    CONCAT('Lecture ', lecture_uid::text)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE lecture_gen
                SET lecture_link = COALESCE(
                    NULLIF(lecture_link, ''),
                    NULLIF(lecture_data->>'lecture_url', ''),
                    CONCAT('/lectures/general/lecture/', lecture_uid::text, '.json')
                )
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE lecture_gen
                SET lecture_shared = COALESCE(lecture_shared, FALSE)
                """
            )
        )

@_retry_on_deadlock(max_retries=3, base_delay=0.5)
def _ensure_lecture_gen_timestamps() -> None:
    """Ensure lecture_gen has timestamp columns required by ORM."""

    inspector = inspect(engine)
    if "lecture_gen" not in inspector.get_table_names():
        return

    columns = {col["name"]: col for col in inspector.get_columns("lecture_gen")}
    statements: list[str] = []

    if "created_at" not in columns:
        logger.info("Adding missing created_at column to lecture_gen table")
        statements.append(
            "ALTER TABLE lecture_gen ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()"
        )
    else:
        created_default = columns["created_at"].get("default")
        if not created_default:
            logger.info("Setting default for lecture_gen.created_at")
            statements.append(
                "ALTER TABLE lecture_gen ALTER COLUMN created_at SET DEFAULT NOW()"
            )

    if "updated_at" not in columns:
        logger.info("Adding missing updated_at column to lecture_gen table")
        statements.append(
            "ALTER TABLE lecture_gen ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW()"
        )
    else:
        updated_default = columns["updated_at"].get("default")
        if not updated_default:
            logger.info("Setting default for lecture_gen.updated_at")
            statements.append(
                "ALTER TABLE lecture_gen ALTER COLUMN updated_at SET DEFAULT NOW()"
            )
    if "lecture_data" not in columns:
        logger.info("Adding missing lecture_data column to lecture_gen table")
        statements.append(
            "ALTER TABLE lecture_gen ADD COLUMN lecture_data JSONB"
        )

    if statements:
        with engine.begin() as connection:
            for stmt in statements:
                connection.execute(text(stmt))

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE lecture_gen SET created_at = NOW() WHERE created_at IS NULL")
        )
        connection.execute(
            text(
                "UPDATE lecture_gen SET updated_at = COALESCE(updated_at, created_at, NOW())"
            )
        )
        connection.execute(
            text("ALTER TABLE lecture_gen ALTER COLUMN created_at SET NOT NULL")
        )
        connection.execute(
            text("ALTER TABLE lecture_gen ALTER COLUMN updated_at SET NOT NULL")
        )


@_retry_on_deadlock(max_retries=3, base_delay=0.5)
def _ensure_administrator_timestamps() -> None:
    """Ensure administrators table has timestamp defaults/columns."""

    inspector = inspect(engine)
    if "administrators" not in inspector.get_table_names():
        return

    columns = {col["name"]: col for col in inspector.get_columns("administrators")}
    statements: list[str] = []

    if "created_at" not in columns:
        logger.info("Adding missing created_at column to administrators table")
        statements.append(
            "ALTER TABLE administrators ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()"
        )
    else:
        created_default = columns["created_at"].get("default")
        if not created_default:
            logger.info("Setting default for administrators.created_at")
            statements.append(
                "ALTER TABLE administrators ALTER COLUMN created_at SET DEFAULT NOW()"
            )

    if "updated_at" not in columns:
        logger.info("Adding missing updated_at column to administrators table")
        statements.append(
            "ALTER TABLE administrators ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW()"
        )
    else:
        updated_default = columns["updated_at"].get("default")
        if not updated_default:
            logger.info("Setting default for administrators.updated_at")
            statements.append(
                "ALTER TABLE administrators ALTER COLUMN updated_at SET DEFAULT NOW()"
            )

    if statements:
        with engine.begin() as connection:
            for stmt in statements:
                connection.execute(text(stmt))

    with engine.begin() as connection:
        connection.execute(
            text("UPDATE administrators SET created_at = NOW() WHERE created_at IS NULL")
        )
        connection.execute(
            text(
                "UPDATE administrators SET updated_at = COALESCE(updated_at, created_at, NOW())"
            )
        )
        connection.execute(
            text("ALTER TABLE administrators ALTER COLUMN created_at SET NOT NULL")
        )
        connection.execute(
            text("ALTER TABLE administrators ALTER COLUMN updated_at SET NOT NULL")
        )


def init_db() -> None:
    """Create database tables if they do not exist and align schema."""

    # Import models that should be registered with SQLAlchemy metadata.
    import app.models.chapter_material  # noqa: F401  (ensure model is imported)

    # Base.metadata.create_all(bind=engine)
    try:
        # _ensure_chapter_material_timestamps()
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        logger.warning(
            "Skipping SQLAlchemy metadata creation during startup due to database error: %s",
            exc,
        )
        return
    try:
        _ensure_chapter_material_schema()
        _ensure_lecture_gen_core_columns()
        _ensure_lecture_gen_timestamps()
        _ensure_administrator_timestamps()
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("Failed to ensure updated_at column on chapter_materials")