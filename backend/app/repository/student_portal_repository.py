"""Psycopg helpers for student portal accounts and profiles."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from psycopg import OperationalError

from ..postgres import get_pg_cursor

logger = logging.getLogger(__name__)

_ACCOUNT_COLUMNS: List[str] = [
    "id",
    "enrollment_number",
    "password_hash",
    "created_at",
    "last_login",
]

_PROFILE_COLUMNS: List[str] = [
    "id",
    "first_name",
    "middle_name",
    "last_name",
    "class_stream",
    "division",
    "class_head",
    "enrollment_number",
    "mobile_number",
    "parents_number",
    "email",
    "photo_path",
    "created_at",
]


def _ensure_chat_message_columns() -> None:
    """Ensure chat message table has attachment and timestamp columns."""

    columns_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'student_chat_messages'
    """

    with get_pg_cursor() as cur:
        cur.execute(columns_query)
        existing_columns = {row["column_name"] for row in cur.fetchall()}

    alterations: List[str] = []

    if "id" not in existing_columns:
        alterations.append(
            "ALTER TABLE student_chat_messages "
            "ADD COLUMN id BIGSERIAL PRIMARY KEY"
        )
    if "created_at" not in existing_columns:
        alterations.append(
            "ALTER TABLE student_chat_messages "
            "ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
    if "attachment_path" not in existing_columns:
        alterations.append("ALTER TABLE student_chat_messages ADD COLUMN attachment_path TEXT")
    if "attachment_name" not in existing_columns:
        alterations.append("ALTER TABLE student_chat_messages ADD COLUMN attachment_name TEXT")
    if "attachment_mime_type" not in existing_columns:
        alterations.append("ALTER TABLE student_chat_messages ADD COLUMN attachment_mime_type TEXT")
    if "attachment_size" not in existing_columns:
        alterations.append("ALTER TABLE student_chat_messages ADD COLUMN attachment_size BIGINT")

    if not alterations:
        return

    with get_pg_cursor(dict_rows=False) as cur:
        for statement in alterations:
            cur.execute(statement)


try:
    _ensure_chat_message_columns()
except OperationalError as exc:  # pragma: no cover - defensive startup guard
    logger.warning("Skipping chat column auto-migration due to DB error: %s", exc)


def _row_to_account(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row.get(column) for column in _ACCOUNT_COLUMNS}


def _row_to_profile(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row.get(column) for column in _PROFILE_COLUMNS}


def get_roster_entry(enrollment_number: str) -> Optional[Dict[str, Any]]:
    query = (
        "SELECT admin_id, enrollment_number, first_name,middle_name, last_name, std, division, auto_password "
        "FROM student_roster_entries "
        "WHERE LOWER(TRIM(enrollment_number)) = LOWER(TRIM(%(enrollment_number)s)) LIMIT 1"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"enrollment_number": enrollment_number})
        return cur.fetchone()


def update_roster_auto_password(enrollment_number: str, auto_password: str) -> None:
    query = (
        "UPDATE student_roster_entries SET auto_password = %(auto_password)s "
        "WHERE LOWER(TRIM(enrollment_number)) = LOWER(TRIM(%(enrollment_number)s))"
    )
    params = {"enrollment_number": enrollment_number, "auto_password": auto_password}
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)


def create_student_account(*, enrollment_number: str, password_hash: str) -> Dict[str, Any]:
    params = {
        "enrollment_number": enrollment_number,
        "password_hash": password_hash,
        "created_at": datetime.utcnow(),
    }
    query = (
        "INSERT INTO student_accounts (enrollment_number, password_hash, created_at) "
        "VALUES (%(enrollment_number)s, %(password_hash)s, %(created_at)s) "
        f"RETURNING {', '.join(_ACCOUNT_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        account = _row_to_account(cur.fetchone())

    if account is None:
        raise RuntimeError("Failed to create student account")
    return account


def bulk_upsert_student_accounts(credentials: List[Dict[str, Any]]) -> None:
    if not credentials:
        return

    payload = [
        {
            "enrollment_number": cred["enrollment_number"],
            "password_hash": cred["password_hash"],
            "created_at": cred.get("created_at", datetime.utcnow()),
        }
        for cred in credentials
    ]

    query = (
        "INSERT INTO student_accounts (enrollment_number, password_hash, created_at) "
        "VALUES (%(enrollment_number)s, %(password_hash)s, %(created_at)s) "
        "ON CONFLICT (enrollment_number) DO UPDATE SET password_hash = EXCLUDED.password_hash"
    )

    with get_pg_cursor(dict_rows=False) as cur:
        cur.executemany(query, payload)


def upsert_student_account(*, enrollment_number: str, password_hash: str) -> Dict[str, Any]:
    params = {
        "enrollment_number": enrollment_number,
        "password_hash": password_hash,
        "created_at": datetime.utcnow(),
    }
    query = (
        "INSERT INTO student_accounts (enrollment_number, password_hash, created_at) "
        "VALUES (%(enrollment_number)s, %(password_hash)s, %(created_at)s) "
        "ON CONFLICT (enrollment_number) DO UPDATE SET password_hash = EXCLUDED.password_hash "
        f"RETURNING {', '.join(_ACCOUNT_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        account = _row_to_account(cur.fetchone())

    if account is None:
        raise RuntimeError("Failed to upsert student account")
    return account


def get_student_account_by_enrollment(enrollment_number: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ACCOUNT_COLUMNS)} FROM student_accounts "
        "WHERE LOWER(enrollment_number) = LOWER(%(enrollment_number)s) LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"enrollment_number": enrollment_number})
        return _row_to_account(cur.fetchone())


def update_student_last_login(account_id: int, when: datetime) -> Optional[Dict[str, Any]]:
    query = (
        "UPDATE student_accounts SET last_login = %(last_login)s "
        "WHERE id = %(account_id)s RETURNING " + ", ".join(_ACCOUNT_COLUMNS)
    )
    params = {"account_id": account_id, "last_login": when}
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return _row_to_account(cur.fetchone())


def update_student_password(enrollment_number: str, password_hash: str) -> Optional[Dict[str, Any]]:
    query = (
        "UPDATE student_accounts SET password_hash = %(password_hash)s "
        "WHERE LOWER(enrollment_number) = LOWER(%(enrollment_number)s) "
        "RETURNING " + ", ".join(_ACCOUNT_COLUMNS)
    )
    params = {"enrollment_number": enrollment_number, "password_hash": password_hash}
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return _row_to_account(cur.fetchone())


def get_student_profile_by_enrollment(enrollment_number: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_PROFILE_COLUMNS)} FROM student_profiles "
        "WHERE LOWER(enrollment_number) = LOWER(%(enrollment_number)s) LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"enrollment_number": enrollment_number})
        return _row_to_profile(cur.fetchone())

def delete_student_profile_by_enrollment(enrollment_number: str) -> bool:
    query = (
        "DELETE FROM student_profiles "
        "WHERE LOWER(enrollment_number) = LOWER(%(enrollment_number)s) "
        "RETURNING id"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"enrollment_number": enrollment_number})
        return cur.fetchone() is not None


def create_student_profile(**fields: Any) -> Dict[str, Any]:
    payload = {**fields}
    payload.setdefault("created_at", datetime.utcnow())

    columns = [
        "first_name",
        "middle_name",
        "last_name",
        "class_stream",
        "division",
        "class_head",
        "enrollment_number",
        "mobile_number",
        "parents_number",
        "email",
        "photo_path",
        "created_at",
    ]

    column_clause = ", ".join(columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)

    query = (
        f"INSERT INTO student_profiles ({column_clause}) "
        f"VALUES ({placeholders}) RETURNING {', '.join(_PROFILE_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, payload)
        profile = _row_to_profile(cur.fetchone())

    if profile is None:
        raise RuntimeError("Failed to create student profile")
    return profile


def update_student_profile(profile_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_student_profile_by_id(profile_id)

    allowed_columns = {
        "first_name",
        "middle_name",
        "last_name",
        "class_stream",
        "division",
        "class_head",
        "mobile_number",
        "parents_number",
        "email",
        "photo_path",
    }
    invalid = set(fields.keys()) - allowed_columns
    if invalid:
        raise ValueError(f"Invalid student profile columns: {', '.join(sorted(invalid))}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]
    params = {"profile_id": profile_id, **fields}

    query = (
        f"UPDATE student_profiles SET {', '.join(assignments)} "
        "WHERE id = %(profile_id)s RETURNING " + ", ".join(_PROFILE_COLUMNS)
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return _row_to_profile(cur.fetchone())


def get_student_profile_by_id(profile_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_PROFILE_COLUMNS)} FROM student_profiles "
        "WHERE id = %(profile_id)s"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"profile_id": profile_id})
        return _row_to_profile(cur.fetchone())


def get_student_roster_context(enrollment_number: str) -> Optional[Dict[str, Any]]:
    query = """
        SELECT
            r.admin_id,
            r.enrollment_number,
            r.std,
            r.division,
            r.first_name AS roster_first_name,
            r.last_name AS roster_last_name,
            p.first_name AS profile_first_name,
            p.class_stream AS profile_class_stream,
            p.division AS profile_division,
            p.photo_path
        FROM student_roster_entries r
        LEFT JOIN student_profiles p
            ON LOWER(TRIM(p.enrollment_number)) = LOWER(TRIM(r.enrollment_number))
        WHERE LOWER(TRIM(r.enrollment_number)) = LOWER(TRIM(%(enrollment_number)s))
        LIMIT 1
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"enrollment_number": enrollment_number})
        return cur.fetchone()


def list_classmates(
    *,
    admin_id: int,
    std: str,
    division: Optional[str],
    exclude_enrollment: str,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            r.enrollment_number,
            r.std,
            r.division,
            r.first_name AS roster_first_name,
            r.last_name AS roster_last_name,
            p.first_name AS profile_first_name,
            p.photo_path
        FROM student_roster_entries r
        LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
        WHERE r.admin_id = %(admin_id)s
          AND r.std = %(std)s
          AND COALESCE(r.division, '') = COALESCE(%(division)s, '')
          AND r.enrollment_number <> %(exclude_enrollment)s
        ORDER BY COALESCE(p.first_name, r.first_name, r.enrollment_number)
    """

    params = {
        "admin_id": admin_id,
        "std": std,
        "division": division,
        "exclude_enrollment": exclude_enrollment,
    }

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()

def list_students_for_class(
    *,
    admin_id: int,
    std: str,
    division: Optional[str],
) -> List[Dict[str, Any]]:
    """Return roster/profile details for students in a class/division."""

    where_clauses = [
        "r.admin_id = %(admin_id)s",
        "(r.std = %(std)s OR p.class_stream ILIKE %(std_like)s)",
    ]

    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "std": std,
        "std_like": f"%{std}%",
    }

    if division:
        where_clauses.append(
            "(COALESCE(r.division, '') = %(division_exact)s OR COALESCE(p.division, '') ILIKE %(division_like)s)"
        )
        params.update(
            {
                "division_exact": division,
                "division_like": f"%{division}%",
            }
        )

    query = f"""
        SELECT
            r.enrollment_number,
            r.std,
            r.division,
            r.first_name AS roster_first_name,
            r.last_name AS roster_last_name,
            p.first_name AS profile_first_name,
            p.last_name AS profile_last_name,
            p.class_stream,
            p.division AS profile_division,
            p.photo_path
        FROM student_roster_entries r
        LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
        WHERE {' AND '.join(where_clauses)}
        ORDER BY COALESCE(p.first_name, r.first_name, r.enrollment_number)
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def fetch_latest_peer_messages(
    *,
    admin_id: int,
    current_enrollment: str,
    peer_enrollments: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not peer_enrollments:
        return {}

    query = """
        WITH ranked AS (
            SELECT
                CASE
                    WHEN sender_enrollment = %(current)s THEN receiver_enrollment
                    ELSE sender_enrollment
                END AS peer_enrollment,
                message,
                attachment_name,
                attachment_path,
                attachment_mime_type,
                sender_enrollment,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY CASE
                        WHEN sender_enrollment = %(current)s THEN receiver_enrollment
                        ELSE sender_enrollment
                    END
                    ORDER BY created_at DESC
                ) AS rn
            FROM student_chat_messages
            WHERE admin_id = %(admin_id)s
              AND (
                    (sender_enrollment = %(current)s AND receiver_enrollment = ANY(%(peers)s))
                 OR (receiver_enrollment = %(current)s AND sender_enrollment = ANY(%(peers)s))
              )
        )
        SELECT peer_enrollment, message, attachment_name, attachment_path, attachment_mime_type, sender_enrollment, created_at
        FROM ranked
        WHERE rn = 1
    """

    params = {
        "admin_id": admin_id,
        "current": current_enrollment,
        "peers": list(peer_enrollments),
    }

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return {row["peer_enrollment"]: row for row in rows}


def fetch_chat_messages(
    *,
    admin_id: int,
    enrollment_a: str,
    enrollment_b: str,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            id,
            sender_enrollment,
            receiver_enrollment,
            message,
            attachment_path,
            attachment_name,
            attachment_mime_type,
            attachment_size,
            created_at
        FROM student_chat_messages
        WHERE admin_id = %(admin_id)s
          AND (
                (sender_enrollment = %(enrollment_a)s AND receiver_enrollment = %(enrollment_b)s)
             OR (sender_enrollment = %(enrollment_b)s AND receiver_enrollment = %(enrollment_a)s)
          )
        ORDER BY created_at ASC
        LIMIT %(limit)s
    """

    params = {
        "admin_id": admin_id,
        "enrollment_a": enrollment_a,
        "enrollment_b": enrollment_b,
        "limit": limit,
    }

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def insert_chat_message(
    *,
    admin_id: int,
    sender_enrollment: str,
    receiver_enrollment: str,
    message: Optional[str],
    attachment_path: Optional[str] = None,
    attachment_name: Optional[str] = None,
    attachment_mime_type: Optional[str] = None,
    attachment_size: Optional[int] = None,
) -> Dict[str, Any]:
    query = """
        INSERT INTO student_chat_messages (
            admin_id,
            sender_enrollment,
            receiver_enrollment,
            message,
            attachment_path,
            attachment_name,
            attachment_mime_type,
            attachment_size
        )
        VALUES (%(admin_id)s, %(sender)s, %(receiver)s, %(message)s, %(attachment_path)s, %(attachment_name)s, %(attachment_mime_type)s, %(attachment_size)s)
        RETURNING
            id,
            sender_enrollment,
            receiver_enrollment,
            message,
            attachment_path,
            attachment_name,
            attachment_mime_type,
            attachment_size,
            created_at
    """

    params = {
        "admin_id": admin_id,
        "sender": sender_enrollment,
        "receiver": receiver_enrollment,
        "message": message or "",
        "attachment_path": attachment_path,
        "attachment_name": attachment_name,
        "attachment_mime_type": attachment_mime_type,
        "attachment_size": attachment_size,
    }

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()
