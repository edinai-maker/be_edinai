"""Repository helpers for managing student portal videos and engagements."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor


def _ensure_tables_exist() -> None:
    create_videos_sql = """
        CREATE TABLE IF NOT EXISTS student_portal_videos (
            id SERIAL PRIMARY KEY,
            admin_id INTEGER NOT NULL,
            std TEXT,
            subject TEXT,
            title TEXT NOT NULL,
            description TEXT,
            chapter_name TEXT,
            duration_seconds INTEGER,
            video_url TEXT,
            thumbnail_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            total_watch_time_seconds INTEGER NOT NULL DEFAULT 0,
            total_watch_count INTEGER NOT NULL DEFAULT 0,
            total_likes INTEGER NOT NULL DEFAULT 0,
            total_comments INTEGER NOT NULL DEFAULT 0,
            total_subscribers INTEGER NOT NULL DEFAULT 0
        )
    """

    create_engagement_sql = """
        CREATE TABLE IF NOT EXISTS student_portal_video_engagement (
            video_id INTEGER NOT NULL REFERENCES student_portal_videos(id) ON DELETE CASCADE,
            enrollment_number TEXT NOT NULL,
            liked BOOLEAN NOT NULL DEFAULT FALSE,
            subscribed BOOLEAN NOT NULL DEFAULT FALSE,
            watch_duration_seconds INTEGER NOT NULL DEFAULT 0,
            last_watched_at TIMESTAMPTZ,
            PRIMARY KEY (video_id, enrollment_number)
        )
    """

    create_comments_sql = """
        CREATE TABLE IF NOT EXISTS student_portal_video_comments (
            id SERIAL PRIMARY KEY,
            video_id INTEGER NOT NULL REFERENCES student_portal_videos(id) ON DELETE CASCADE,
            enrollment_number TEXT,
            comment TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            like_count INTEGER NOT NULL DEFAULT 0
        )
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(create_videos_sql)
        cur.execute(create_engagement_sql)
        cur.execute(create_comments_sql)


_ensure_tables_exist()


def _row_to_video(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row.get("id"),
        "admin_id": row.get("admin_id"),
        "std": row.get("std"),
        "subject": row.get("subject"),
        "title": row.get("title"),
        "description": row.get("description"),
        "chapter_name": row.get("chapter_name"),
        "duration_seconds": row.get("duration_seconds"),
        "video_url": row.get("video_url"),
        "thumbnail_url": row.get("thumbnail_url"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "total_watch_time_seconds": row.get("total_watch_time_seconds"),
        "total_watch_count": row.get("total_watch_count"),
        "total_likes": row.get("total_likes"),
        "total_comments": row.get("total_comments"),
        "total_subscribers": row.get("total_subscribers"),
    }


def create_video(
    *,
    admin_id: int,
    std: Optional[str],
    subject: Optional[str],
    title: str,
    description: Optional[str],
    chapter_name: Optional[str],
    duration_seconds: Optional[int],
    video_url: Optional[str],
    thumbnail_url: Optional[str],
) -> Dict[str, Any]:
    payload = {
        "admin_id": admin_id,
        "std": std,
        "subject": subject,
        "title": title,
        "description": description,
        "chapter_name": chapter_name,
        "duration_seconds": duration_seconds,
        "video_url": video_url,
        "thumbnail_url": thumbnail_url,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    columns = ", ".join(payload.keys())
    values = ", ".join(f"%({column})s" for column in payload.keys())

    query = f"""
        INSERT INTO student_portal_videos ({columns})
        VALUES ({values})
        RETURNING *
    """

    with get_pg_cursor() as cur:
        cur.execute(query, payload)
        return _row_to_video(cur.fetchone())  # type: ignore[arg-type]

def _normalize_url_for_match(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    return text.lower()


def delete_videos_by_identifiers(
    *,
    admin_id: int,
    canonical_urls: Iterable[str],
    filenames: Iterable[str],
) -> int:
    canonical_set = {
        _normalize_url_for_match(url)
        for url in canonical_urls
        if isinstance(url, str) and url.strip()
    }
    filename_set = {
        Path(name).name.lower()
        for name in filenames
        if isinstance(name, str) and name.strip()
    }

    if not canonical_set and not filename_set:
        return 0

    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT id, video_url
            FROM student_portal_videos
            WHERE admin_id = %(admin_id)s
            """,
            {"admin_id": admin_id},
        )
        rows = cur.fetchall()

    ids_to_delete: List[int] = []
    for row in rows:
        video_id = row.get("id")
        video_url = row.get("video_url")
        normalized_url = _normalize_url_for_match(video_url)
        filename = Path(str(video_url or "")).name.lower()

        if normalized_url in canonical_set or filename in filename_set:
            if isinstance(video_id, int):
                ids_to_delete.append(video_id)

    if not ids_to_delete:
        return 0

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(
            """
            DELETE FROM student_portal_videos
            WHERE id = ANY(%(ids)s)
            """,
            {"ids": ids_to_delete},
        )
        return cur.rowcount or 0

def ensure_sample_videos(admin_id: int, std: Optional[str], samples: Iterable[Dict[str, Any]]) -> None:
    params: Dict[str, Any] = {"admin_id": admin_id}
    where_clauses = ["admin_id = %(admin_id)s"]
    normalized_std = (std or "").strip()
    if normalized_std:
        params["std"] = normalized_std
        where_clauses.append("(std = %(std)s OR std IS NULL)")
    else:
        where_clauses.append("(std IS NULL OR std = '')")

    existing_rows: List[Dict[str, Any]] = []
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    shared_key = "__shared__"

    select_query = """
        SELECT id, title, std
        FROM student_portal_videos
        WHERE {where}
    """.format(where=" AND ".join(where_clauses))

    with get_pg_cursor() as cur:
        cur.execute(select_query, params)
        existing_rows = cur.fetchall()

    for row in existing_rows:
        row_std_key = (row.get("std") or "").strip() or shared_key
        lookup[(row.get("title"), row_std_key)] = row

    target_std_key = normalized_std or shared_key

    def _prepare_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subject": sample.get("subject"),
            "description": sample.get("description"),
            "chapter_name": sample.get("chapter_name"),
            "duration_seconds": sample.get("duration_seconds"),
            "video_url": sample.get("video_url"),
            "thumbnail_url": sample.get("thumbnail_url"),
            "updated_at": datetime.utcnow(),
        }

    for sample in samples:
        title = sample.get("title", "Untitled")
        existing = lookup.get((title, target_std_key)) or lookup.get((title, shared_key))

        if existing:
            payload = _prepare_payload(sample)
            payload["id"] = existing["id"]
            update_query = """
                UPDATE student_portal_videos
                SET subject = %(subject)s,
                    description = %(description)s,
                    chapter_name = %(chapter_name)s,
                    duration_seconds = %(duration_seconds)s,
                    video_url = %(video_url)s,
                    thumbnail_url = %(thumbnail_url)s,
                    updated_at = %(updated_at)s
                WHERE id = %(id)s
            """
            with get_pg_cursor() as cur:
                cur.execute(update_query, payload)
            continue

        created = create_video(
            admin_id=admin_id,
            std=std,
            subject=sample.get("subject"),
            title=title,
            description=sample.get("description"),
            chapter_name=sample.get("chapter_name"),
            duration_seconds=sample.get("duration_seconds"),
            video_url=sample.get("video_url"),
            thumbnail_url=sample.get("thumbnail_url"),
        )
        lookup[(title, (created.get("std") or "").strip() or shared_key)] = {
            "id": created.get("id"),
            "title": title,
            "std": created.get("std"),
        }


def get_video(video_id: int) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM student_portal_videos WHERE id = %(video_id)s"
    with get_pg_cursor() as cur:
        cur.execute(query, {"video_id": video_id})
        return _row_to_video(cur.fetchone())  # type: ignore[arg-type]


def get_video_with_engagement(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
    video_id: int,
) -> Optional[Dict[str, Any]]:
    params = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
        "video_id": video_id,
    }

    std_clause = ""
    if std:
        
        normalized_std = std.strip()
        params["std"] = normalized_std
        params["std_regex"] = rf"(^|[,\s]){normalized_std}([,\s]|$)"
        std_clause = " AND (v.std = %(std)s OR v.std ~ %(std_regex)s OR v.std IS NULL)"

    query = f"""
        SELECT
            v.*,
            e.liked AS user_liked,
            e.subscribed AS user_subscribed,
            e.watch_duration_seconds AS user_watch_duration_seconds,
            e.last_watched_at AS user_last_watched_at
        FROM student_portal_videos v
        LEFT JOIN student_portal_video_engagement e
          ON e.video_id = v.id AND e.enrollment_number = %(enrollment)s
        WHERE v.id = %(video_id)s
          AND v.admin_id = %(admin_id)s
          {std_clause}
        LIMIT 1
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    if not row:
        return None

    base = _row_to_video(row)
    if base is None:
        return None
    base.update(
        {
            "user_liked": row.get("user_liked", False),
            "user_subscribed": row.get("user_subscribed", False),
            "user_watch_duration_seconds": row.get("user_watch_duration_seconds", 0),
            "user_last_watched_at": row.get("user_last_watched_at"),
        }
    )
    return base


def list_videos_for_student(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
    limit: int,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
        "limit": limit,
    }

    std_clause = ""
    if std:
        # params["std"] = std
        # std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"
        normalized_std = std.strip()
        params["std"] = normalized_std
        params["std_regex"] = rf"(^|[,\s]){normalized_std}([,\s]|$)"
        std_clause = " AND (v.std = %(std)s OR v.std ~ %(std_regex)s OR v.std IS NULL)"

    query = f"""
        SELECT
            v.*, 
            e.liked AS user_liked,
            e.subscribed AS user_subscribed,
            e.watch_duration_seconds AS user_watch_duration_seconds,
            e.last_watched_at AS user_last_watched_at
        FROM student_portal_videos v
        LEFT JOIN student_portal_video_engagement e
          ON e.video_id = v.id AND e.enrollment_number = %(enrollment)s
        WHERE v.admin_id = %(admin_id)s
          {std_clause}
        ORDER BY v.created_at DESC
        LIMIT %(limit)s
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    videos: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        base.update(
            {
                "user_liked": row.get("user_liked", False),
                "user_subscribed": row.get("user_subscribed", False),
                "user_watch_duration_seconds": row.get("user_watch_duration_seconds", 0),
                "user_last_watched_at": row.get("user_last_watched_at"),
            }
        )
        videos.append(base)
    return videos
def list_all_videos(
    *,
    admin_id: int,
    std: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"admin_id": admin_id}
    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (std = %(std)s OR std IS NULL)"
    query = f"""
        SELECT *
        FROM student_portal_videos
        WHERE admin_id = %(admin_id)s
          {std_clause}
        ORDER BY created_at DESC
    """
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    videos: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        videos.append(base)
    return videos
def list_all_videos_for_student(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
    }
    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"
    query = f"""
        SELECT
            v.*,
            e.liked AS user_liked,
            e.subscribed AS user_subscribed,
            e.watch_duration_seconds AS user_watch_duration_seconds,
            e.last_watched_at AS user_last_watched_at
        FROM student_portal_videos v
        LEFT JOIN student_portal_video_engagement e
          ON e.video_id = v.id AND e.enrollment_number = %(enrollment)s
        WHERE v.admin_id = %(admin_id)s
          {std_clause}
        ORDER BY v.created_at DESC
    """
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    videos: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        base.update(
            {
                "user_liked": row.get("user_liked", False),
                "user_subscribed": row.get("user_subscribed", False),
                "user_watch_duration_seconds": row.get("user_watch_duration_seconds", 0),
                "user_last_watched_at": row.get("user_last_watched_at"),
            }
        )
        videos.append(base)
    return videos

def list_related_videos(
    *,
    admin_id: int,
    std: Optional[str],
    subject: Optional[str],
    enrollment_number: str,
    exclude_video_id: int,
    limit: int,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
        "video_id": exclude_video_id,
        "limit": limit,
    }

    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"

    subject_order_clause = ""
    if subject:
        params["subject"] = subject
        subject_order_clause = ""
        subject_order_clause = """
            CASE
                WHEN v.subject = %(subject)s THEN 0
                ELSE 1
            END,
        """

    query = f"""
        SELECT
            v.*,
            e.liked AS user_liked,
            e.subscribed AS user_subscribed,
            e.watch_duration_seconds AS user_watch_duration_seconds,
            e.last_watched_at AS user_last_watched_at
        FROM student_portal_videos v
        LEFT JOIN student_portal_video_engagement e
          ON e.video_id = v.id AND e.enrollment_number = %(enrollment)s
        WHERE v.admin_id = %(admin_id)s
          AND v.id <> %(video_id)s
          {std_clause}
        ORDER BY
            {subject_order_clause}
            v.created_at DESC
        LIMIT %(limit)s
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    related: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        base.update(
            {
                "user_liked": row.get("user_liked", False),
                "user_subscribed": row.get("user_subscribed", False),
                "user_watch_duration_seconds": row.get("user_watch_duration_seconds", 0),
                "user_last_watched_at": row.get("user_last_watched_at"),
            }
        )
        related.append(base)
    return related


def list_watched_videos(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
    }

    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"

    limit_clause = ""
    if limit is not None:
        params["limit"] = limit
        limit_clause = " LIMIT %(limit)s"

    query = f"""
        SELECT
            v.*,
            e.watch_duration_seconds AS user_watch_duration_seconds,
            e.liked AS user_liked,
            e.subscribed AS user_subscribed,
            e.last_watched_at AS user_last_watched_at
        FROM student_portal_video_engagement e
        JOIN student_portal_videos v ON v.id = e.video_id
        WHERE e.enrollment_number = %(enrollment)s
          AND v.admin_id = %(admin_id)s
          {std_clause}
        ORDER BY e.last_watched_at DESC NULLS LAST
        {limit_clause}
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    items: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        user_watch_seconds = row.get("user_watch_duration_seconds", 0) or 0
        user_last_watched = row.get("user_last_watched_at")
        user_liked = row.get("user_liked", False)
        user_subscribed = row.get("user_subscribed", False)
        base.update(
            {
                "watch_duration_seconds": user_watch_seconds,
                "last_watched_at": user_last_watched,
                "liked": user_liked,
                "subscribed": user_subscribed,
                "user_watch_duration_seconds": user_watch_seconds,
                "user_last_watched_at": user_last_watched,
                "user_liked": user_liked,
                "user_subscribed": user_subscribed,
            }
        )
        items.append(base)
    return items


def list_subscribed_videos(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
    }

    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"

    query = f"""
        SELECT
            v.*,
            e.watch_duration_seconds,
            e.liked,
            e.subscribed,
            e.last_watched_at
        FROM student_portal_video_engagement e
        JOIN student_portal_videos v ON v.id = e.video_id
        WHERE e.enrollment_number = %(enrollment)s
          AND v.admin_id = %(admin_id)s
          AND e.subscribed = TRUE
          {std_clause}
        ORDER BY COALESCE(e.last_watched_at, v.created_at) DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    items: List[Dict[str, Any]] = []
    for row in rows:
        base = _row_to_video(row)
        if base is None:
            continue
        base.update(
            {
                "watch_duration_seconds": row.get("watch_duration_seconds", 0),
                "liked": row.get("liked", False),
                "subscribed": row.get("subscribed", False),
                "last_watched_at": row.get("last_watched_at"),
            }
        )
        items.append(base)
    return items


def _update_video_totals(video_id: int, **deltas: int) -> None:
    if not deltas:
        return

    assignments = [
        f"{column} = {column} + %({column})s"
        for column in deltas.keys()
    ]
    params = {**deltas, "video_id": video_id}
    query = f"""
        UPDATE student_portal_videos
        SET {', '.join(assignments)}, updated_at = NOW()
        WHERE id = %(video_id)s
    """
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)

def set_video_duration_seconds(*, video_id: int, duration_seconds: int) -> None:
    if duration_seconds <= 0:
        return
    query = """
        UPDATE student_portal_videos
        SET duration_seconds = %(duration_seconds)s,
            updated_at = NOW()
        WHERE id = %(video_id)s
          AND (duration_seconds IS NULL OR duration_seconds <> %(duration_seconds)s)
    """
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"video_id": video_id, "duration_seconds": duration_seconds})

def record_watch_event(
    *,
    video_id: int,
    enrollment_number: str,
    watch_seconds: int,
) -> None:
    if watch_seconds <= 0:
        return

    params = {
        "video_id": video_id,
        "enrollment": enrollment_number,
    }

    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT
                e.watch_duration_seconds,
                v.duration_seconds
            FROM student_portal_videos v
            LEFT JOIN student_portal_video_engagement e
              ON e.video_id = v.id AND e.enrollment_number = %(enrollment)s
            WHERE v.id = %(video_id)s
            """,
            params,
        )
        row = cur.fetchone()

        existing_watch = 0
        duration_seconds = None
        if row:
            existing_watch = int(row.get("watch_duration_seconds") or 0)
            duration_seconds = row.get("duration_seconds")
            try:
                duration_seconds = int(duration_seconds) if duration_seconds is not None else None
            except (TypeError, ValueError):
                duration_seconds = None

        target_watch = existing_watch + int(watch_seconds)
        if duration_seconds and duration_seconds > 0:
            target_watch = min(target_watch, duration_seconds)

        delta_added = target_watch - existing_watch
        if delta_added <= 0:
            return

        now = datetime.utcnow()
        if existing_watch <= 0:
            cur.execute(
                """
                INSERT INTO student_portal_video_engagement (
                    video_id, enrollment_number, watch_duration_seconds, last_watched_at
                ) VALUES (%(video_id)s, %(enrollment)s, %(watch_seconds)s, %(now)s)
                ON CONFLICT (video_id, enrollment_number)
                DO UPDATE SET watch_duration_seconds = EXCLUDED.watch_duration_seconds,
                              last_watched_at = EXCLUDED.last_watched_at
                """,
                {**params, "watch_seconds": target_watch, "now": now},
            )
            _update_video_totals(
                video_id,
                total_watch_count=1,
                total_watch_time_seconds=delta_added,
            )
        else:
            cur.execute(
                """
                UPDATE student_portal_video_engagement
                SET watch_duration_seconds = %(new_watch)s,
                    last_watched_at = %(now)s
                WHERE video_id = %(video_id)s AND enrollment_number = %(enrollment)s
                """,
                {**params, "new_watch": target_watch, "now": now},
            )
            _update_video_totals(video_id, total_watch_time_seconds=delta_added)


def set_like_status(
    *,
    video_id: int,
    enrollment_number: str,
    liked: bool,
) -> Dict[str, bool]:
    params = {
        "video_id": video_id,
        "enrollment": enrollment_number,
    }

    delta = 0
    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT liked FROM student_portal_video_engagement
            WHERE video_id = %(video_id)s AND enrollment_number = %(enrollment)s
            """,
            params,
        )
        existing = cur.fetchone()

        if existing is None:
            cur.execute(
                """
                INSERT INTO student_portal_video_engagement (
                    video_id, enrollment_number, liked, last_watched_at
                ) VALUES (%(video_id)s, %(enrollment)s, %(liked)s, %(now)s)
                """,
                {**params, "liked": liked, "now": datetime.utcnow()},
            )
            delta = 1 if liked else 0
        else:
            previous = bool(existing.get("liked"))
            if previous == liked:
                delta = 0
            else:
                delta = 1 if liked else -1
            cur.execute(
                """
                UPDATE student_portal_video_engagement
                SET liked = %(liked)s
                WHERE video_id = %(video_id)s AND enrollment_number = %(enrollment)s
                """,
                {**params, "liked": liked},
            )

    if delta:
        _update_video_totals(video_id, total_likes=delta)

    return {"liked": liked}


def set_subscription_status(
    *,
    video_id: int,
    enrollment_number: str,
    subscribed: bool,
) -> Dict[str, bool]:
    params = {
        "video_id": video_id,
        "enrollment": enrollment_number,
    }

    delta = 0
    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT subscribed FROM student_portal_video_engagement
            WHERE video_id = %(video_id)s AND enrollment_number = %(enrollment)s
            """,
            params,
        )
        existing = cur.fetchone()

        if existing is None:
            cur.execute(
                """
                INSERT INTO student_portal_video_engagement (
                    video_id, enrollment_number, subscribed, last_watched_at
                ) VALUES (%(video_id)s, %(enrollment)s, %(subscribed)s, %(now)s)
                """,
                {**params, "subscribed": subscribed, "now": datetime.utcnow()},
            )
            delta = 1 if subscribed else 0
        else:
            previous = bool(existing.get("subscribed"))
            if previous == subscribed:
                delta = 0
            else:
                delta = 1 if subscribed else -1
            cur.execute(
                """
                UPDATE student_portal_video_engagement
                SET subscribed = %(subscribed)s
                WHERE video_id = %(video_id)s AND enrollment_number = %(enrollment)s
                """,
                {**params, "subscribed": subscribed},
            )

    if delta:
        _update_video_totals(video_id, total_subscribers=delta)

    return {"subscribed": subscribed}


def add_comment(
    *,
    video_id: int,
    enrollment_number: Optional[str],
    comment: str,
) -> Dict[str, Any]:
    payload = {
        "video_id": video_id,
        "enrollment": enrollment_number,
        "comment": comment.strip(),
        "created_at": datetime.utcnow(),
    }

    query = """
        WITH inserted AS (
            INSERT INTO student_portal_video_comments (video_id, enrollment_number, comment, created_at)
            VALUES (%(video_id)s, %(enrollment)s, %(comment)s, %(created_at)s)
            RETURNING id, video_id, enrollment_number, comment, created_at, like_count
        )
        SELECT
            i.id,
            i.video_id,
            i.enrollment_number,
            i.comment,
            i.created_at,
            i.like_count,
            COALESCE(
                NULLIF(TRIM(CONCAT_WS(' ', p.first_name, p.middle_name)), ''),
                NULLIF(TRIM(CONCAT_WS(' ', r.first_name, r.last_name)), ''),
                NULLIF(i.enrollment_number, '')
            ) AS student_name
        FROM inserted i
        JOIN student_portal_videos v ON v.id = i.video_id
        LEFT JOIN student_profiles p ON LOWER(p.enrollment_number) = LOWER(i.enrollment_number)
        LEFT JOIN student_roster_entries r
            ON LOWER(r.enrollment_number) = LOWER(i.enrollment_number)
           AND r.admin_id = v.admin_id
    """

    with get_pg_cursor() as cur:
        cur.execute(query, payload)
        record = cur.fetchone()

    _update_video_totals(video_id, total_comments=1)
    return record


def list_comments(video_id: int) -> List[Dict[str, Any]]:
    query = """
        SELECT
            c.id,
            c.video_id,
            c.enrollment_number,
            c.comment,
            c.created_at,
            c.like_count,
            COALESCE(
                NULLIF(TRIM(CONCAT_WS(' ', p.first_name, p.middle_name)), ''),
                NULLIF(TRIM(CONCAT_WS(' ', r.first_name, r.last_name)), ''),
                NULLIF(c.enrollment_number, '')
            ) AS student_name
        FROM student_portal_video_comments c
        JOIN student_portal_videos v ON v.id = c.video_id
        LEFT JOIN student_profiles p ON LOWER(p.enrollment_number) = LOWER(c.enrollment_number)
        LEFT JOIN student_roster_entries r
            ON LOWER(r.enrollment_number) = LOWER(c.enrollment_number)
           AND r.admin_id = v.admin_id
        WHERE c.video_id = %(video_id)s
        ORDER BY c.created_at ASC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"video_id": video_id})
        return cur.fetchall()


def get_watch_summary(
    *,
    admin_id: int,
    std: Optional[str],
    enrollment_number: str,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "enrollment": enrollment_number,
    }

    std_clause = ""
    if std:
        params["std"] = std
        std_clause = " AND (v.std = %(std)s OR v.std IS NULL)"

    summary_query = f"""
        SELECT
            COUNT(*) FILTER (WHERE e.watch_duration_seconds > 0) AS watched_videos,
            COALESCE(SUM(e.watch_duration_seconds), 0) AS total_watch_seconds,
            COUNT(*) FILTER (
                WHERE v.duration_seconds IS NOT NULL
                  AND e.watch_duration_seconds >= v.duration_seconds
            ) AS completed_videos,
            COUNT(*) FILTER (WHERE e.watch_duration_seconds > 0) AS total_records
        FROM student_portal_video_engagement e
        JOIN student_portal_videos v ON v.id = e.video_id
        WHERE e.enrollment_number = %(enrollment)s
          AND v.admin_id = %(admin_id)s
          {std_clause}
    """

    with get_pg_cursor() as cur:
        cur.execute(summary_query, params)
        summary = cur.fetchone() or {}

    return {
        "watched_videos": summary.get("watched_videos", 0),
        "total_watch_seconds": summary.get("total_watch_seconds", 0),
        "completed_videos": summary.get("completed_videos", 0),
        "total_records": summary.get("total_records", 0),
    }


def delete_lecture_videos_by_lecture_id(*, admin_id: int, lecture_id: str) -> int:
    params: Dict[str, Any] = {
        "admin_id": admin_id,
        "lecture_id": lecture_id,
    }

    query = """
        DELETE FROM student_portal_videos
        WHERE admin_id = %(admin_id)s
          AND (
            video_url LIKE '%%/lectures/%%/' || %(lecture_id)s || '.json'
            OR video_url LIKE '%%/lectures/%%/' || %(lecture_id)s || '/%%'
          )
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        return cur.rowcount or 0