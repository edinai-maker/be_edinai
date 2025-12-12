"""Repository helpers for recording lecture shares to students."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor


def _ensure_table_exists() -> None:
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS lecture_shares (
            id SERIAL PRIMARY KEY,
            lecture_id TEXT NOT NULL,
            student_enrollment TEXT NOT NULL,
            std TEXT,
            division TEXT,
            subject TEXT,
            shared_by TEXT,
            share_message TEXT,
            shared_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_lecture_shares_lecture
        ON lecture_shares (lecture_id)
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(create_table_sql)
        cur.execute(create_index_sql)


_ensure_table_exists()


def record_shares(
    *,
    lecture_id: str,
    std: Optional[str],
    subject: Optional[str],
    division: Optional[str],
    shared_by: Optional[str],
    share_message: Optional[str],
    enrollments: Iterable[str],
) -> int:
    """Persist share records for each enrollment. Returns number of rows inserted."""

    rows: List[Dict[str, Optional[str]]] = []
    for enrollment in enrollments:
        enrollment_number = (enrollment or "").strip()
        if not enrollment_number:
            continue
        rows.append(
            {
                "lecture_id": lecture_id,
                "student_enrollment": enrollment_number,
                "std": std,
                "division": division,
                "subject": subject,
                "shared_by": shared_by,
                "share_message": share_message,
            }
        )

    if not rows:
        return 0

    insert_sql = """
        INSERT INTO lecture_shares (
            lecture_id,
            student_enrollment,
            std,
            division,
            subject,
            shared_by,
            share_message
        ) VALUES (
            %(lecture_id)s,
            %(student_enrollment)s,
            %(std)s,
            %(division)s,
            %(subject)s,
            %(shared_by)s,
            %(share_message)s
        )
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.executemany(insert_sql, rows)

    return len(rows)


def list_shares_for_lecture(lecture_id: str) -> List[Dict[str, object]]:
    """Return all share records for a lecture."""

    query = """
        SELECT
            id,
            lecture_id,
            student_enrollment,
            std,
            division,
            subject,
            shared_by,
            share_message,
            shared_at
        FROM lecture_shares
        WHERE lecture_id = %(lecture_id)s
        ORDER BY shared_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"lecture_id": lecture_id})
        return cur.fetchall()

def delete_shares_for_lecture(*, lecture_id: str) -> int:
    """Delete share records for a lecture and return removed row count."""

    query = """
        DELETE FROM lecture_shares
        WHERE lecture_id = %(lecture_id)s
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"lecture_id": lecture_id})
        return cur.rowcount or 0


def list_shared_lectures(
    *,
    admin_id: int,
    std: Optional[str] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Return lectures shared for the given admin with optional filters.

        NOTE:
        - The product flow currently records shared lectures in the
        ``student_portal_videos`` table (video_url column).
        - This repository previously looked only at ``lecture_gen`` +
        ``lecture_shares``. To align with the actual sharing behaviour
        and ensure the `/lectures/shared` API reflects what teachers
        share to students, we now derive the shared lecture list from
        ``student_portal_videos`` when possible.
        """

    filters: List[str] = ["spv.admin_id = %(admin_id)s"]
    params: Dict[str, object] = {"admin_id": admin_id}

    if std:
        filters.append("LOWER(spv.std) = LOWER(%(std)s)")
        params["std"] = std

    if subject:
        filters.append("LOWER(spv.subject) = LOWER(%(subject)s)")
        params["subject"] = subject

    where_clause = " AND ".join(filters)

    # We attempt to derive a lecture_id from the student_portal_videos.video_url
    # for lecture-style URLs while keeping a safe fallback to the row id. Two
    # primary patterns are supported:
    # - Legacy JSON lectures: `/lectures/<std>/<subject>/<lecture_id>.json`
    # - Recorded videos on storage: `/lectures/<admin_id>/<lecture_id>/<file>.mp4`

    query = f"""
        SELECT
COALESCE(
                NULLIF(
                    CASE
                        -- Legacy JSON lecture URLs: take the third segment
                        -- after "/lectures/" (e.g. "142.json") and strip
                        -- the extension.
                        WHEN split_part(split_part(spv.video_url, '/lectures/', 2), '/', 3) LIKE '%%.json'
                        THEN split_part(
                            split_part(
                                split_part(spv.video_url, '/lectures/', 2),
                                '/',
                                3
                            ),
                            '.',
                            1
                        )
                        -- Storage-backed recordings: `/lectures/<admin>/<lecture_id>/<file>`.
                        -- Here we treat the second segment after "/lectures/"
                        -- as the lecture_id when it looks numeric.
                        WHEN split_part(split_part(spv.video_url, '/lectures/', 2), '/', 2) ~ '^[0-9]+$'
                        THEN split_part(
                            split_part(spv.video_url, '/lectures/', 2),
                            '/',
                            2
                        )
                        ELSE NULL
                    END,
                    ''
                ),
                spv.id::text
            ) AS lecture_id,
            spv.title AS title,
            spv.std,
            spv.subject,
            spv.video_url AS lecture_url,
            spv.created_at AS last_shared_at,
            spv.created_at AS lecture_updated_at
        FROM student_portal_videos spv
        WHERE {where_clause}
        ORDER BY spv.created_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()