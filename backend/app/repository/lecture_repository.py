"""
Lecture Repository - Data Storage and Retrieval Layer
Handles all database operations for lecture storage
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.postgres import get_pg_cursor
from app.models.chapter_material import LectureGen

def _slugify(value: Any) -> str:
    """Convert metadata values to slug format for comparisons."""
    return str(value or "").strip().lower().replace(" ", "_")


def _sort_key(value: str) -> Tuple[int, str]:
    """Sort numerically when possible, otherwise lexicographically."""
    try:
        return (0, f"{int(value):02d}")
    except (ValueError, TypeError):
        return (1, (value or "").lower())


def _clone_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of the payload to prevent accidental mutations."""
    return json.loads(json.dumps(payload or {}))


async def create_lecture(
    *,
    title: str,
    language: str,
    style: str,
    duration: int,
    slides: List[Dict[str, Any]],
    context: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    fallback_used: bool = False,
    admin_id: Optional[int] = None,
    material_id: Optional[int] = None,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    sem: Optional[str] = None,
    board: Optional[str] = None,
    lecture_uid: Optional[str] = None,
    lecture_url: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = _default_metadata(metadata)
    lecture_id = lecture_uid or await _generate_lecture_id()
    created_at = datetime.utcnow().isoformat()

    record: Dict[str, Any] = {
        "lecture_id": lecture_id,
        "title": title,
        "language": language,
        "style": style,
        "requested_duration": duration,
        "estimated_duration": len(slides) * 3,
        "total_slides": len(slides),
        "slides": slides,
        "context": context,
        "created_at": created_at,
        "updated_at": created_at,
        "fallback_used": fallback_used,
        "source_text": text,
        "metadata": metadata,
        "play_count": 0,
        "last_played_at": None,
    }

    record["lecture_url"] = lecture_url or _build_default_url(record)

    admin_value = admin_id or metadata.get("admin_id") or 0
    material_value = material_id or metadata.get("material_id") or 0
    subject_value = subject or metadata.get("subject")
    std_value = std or metadata.get("std") or metadata.get("class")
    sem_value = sem or metadata.get("sem")
    board_value = board or metadata.get("board")

    with get_pg_cursor() as cur:
        cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
        existing_row = cur.fetchone()

    params = {
        "admin_id": admin_value,
        "material_id": material_value,
        "lecture_uid": lecture_id,
        "chapter_title": record.get("title") or f"Lecture {lecture_id}",
        "lecture_link": record["lecture_url"],
        "std": std_value,
        "subject": subject_value,
        "sem": sem_value,
        "board": board_value,
        "lecture_data": json.dumps(record),
    }

    if existing_row:
        # Update existing
        query = """
            UPDATE lecture_gen
            SET admin_id = %(admin_id)s,
                material_id = %(material_id)s,
                chapter_title = %(chapter_title)s,
                lecture_link = %(lecture_link)s,
                std = %(std)s,
                subject = %(subject)s,
                sem = %(sem)s,
                board = %(board)s,
                lecture_data = %(lecture_data)s
            WHERE lecture_uid = %(lecture_uid)s
            RETURNING *
        """
    else:
        # Insert new
        query = """
            INSERT INTO lecture_gen (
                admin_id,
                material_id,
                lecture_uid,
                chapter_title,
                lecture_link,
                std,
                subject,
                sem,
                board,
                lecture_data
            )
            VALUES (
                %(admin_id)s,
                %(material_id)s,
                %(lecture_uid)s,
                %(chapter_title)s,
                %(lecture_link)s,
                %(std)s,
                %(subject)s,
                %(sem)s,
                %(board)s,
                %(lecture_data)s
            )
            RETURNING *
        """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        result = cur.fetchone()

    record.setdefault("lecture_id", lecture_id)
    record.setdefault("metadata", metadata)
    record.setdefault("lecture_url", record["lecture_url"])
    if result:
        record["db_record_id"] = result.get("id")
    return record


def _default_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get default metadata or copy provided metadata."""
    return metadata.copy() if metadata else {}


def _metadata_value(metadata: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Optional[str]:
    """Extract value from metadata by trying multiple keys."""
    for key in keys:
        if key in metadata and metadata[key]:
            return metadata[key]
    return default


def _build_default_url(record: Dict[str, Any]) -> Optional[str]:
    """Build default lecture URL from record metadata."""
    metadata = record.get("metadata") or {}
    std_value = _metadata_value(metadata, "std", "class", default="general")
    subject_value = _metadata_value(metadata, "subject", default="lecture")
    lecture_id = record.get("lecture_id")
    if not lecture_id:
        return None
    std_slug = _slugify(std_value)
    subject_slug = _slugify(subject_value)
    return f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"


async def _generate_lecture_id() -> str:
    """Generate next lecture ID by finding max numeric ID."""
    query = """
        SELECT MAX(CAST(lecture_uid AS INTEGER)) as max_id
        FROM lecture_gen
        WHERE lecture_uid ~ '^\\d+$'
    """
    with get_pg_cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()

    max_id = result.get("max_id") if result else None
    next_id = (max_id or 0) + 1
    return str(next_id)


async def get_lecture(lecture_id: str) -> Dict[str, Any]:
    with get_pg_cursor() as cur:
        cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
        row = cur.fetchone()
    if not row or not row.get("lecture_data"):
        raise FileNotFoundError(f"Lecture {lecture_id} not found")

    record = _clone_record(row.get("lecture_data"))
    record.setdefault("lecture_id", row.get("lecture_uid"))
    record.setdefault("metadata", {})
    record.setdefault("lecture_url", row.get("lecture_link"))
    record["db_record_id"] = row.get("id")
    return record


async def update_lecture(lecture_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    with get_pg_cursor() as cur:
        cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
        row = cur.fetchone()
    if not row or not row.get("lecture_data"):
        raise FileNotFoundError(f"Lecture {lecture_id} not found")

    record = _clone_record(row.get("lecture_data"))
    record.update(updates)
    record["updated_at"] = datetime.utcnow().isoformat()

    metadata = record.get("metadata") or {}
    chapter_title = record.get("title") or row.get("chapter_title") or f"Lecture {row.get('lecture_uid')}"
    lecture_link = record.get("lecture_url") or row.get("lecture_link")
    std = metadata.get("std") or metadata.get("class") or row.get("std")
    subject = metadata.get("subject") or row.get("subject")
    sem = metadata.get("sem") or row.get("sem")
    board = metadata.get("board") or row.get("board")

    with get_pg_cursor() as cur:
        cur.execute("""
            UPDATE lecture_gen
            SET chapter_title = %(chapter_title)s,
                lecture_link = %(lecture_link)s,
                std = %(std)s,
                subject = %(subject)s,
                sem = %(sem)s,
                board = %(board)s,
                lecture_data = %(lecture_data)s
            WHERE lecture_uid = %(lecture_uid)s
            RETURNING *
        """, {
            "chapter_title": chapter_title,
            "lecture_link": lecture_link,
            "std": std,
            "subject": subject,
            "sem": sem,
            "board": board,
            "lecture_data": json.dumps(record),
            "lecture_uid": lecture_id,
        })
        result = cur.fetchone()

    record["db_record_id"] = result.get("id")
    return record


async def delete_lectures_by_metadata(
    *,
    std: str,
    subject: str,
    division: Optional[str] = None,
    lecture_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Delete lectures matching metadata filters directly from the database."""

    std_filter = _slugify(std)
    subject_filter = _slugify(subject)
    division_filter = _slugify(division) if division else None

    with get_pg_cursor() as cur:
        cur.execute("""
            SELECT * FROM lecture_gen
            WHERE std = %(std)s AND subject = %(subject)s
        """, {
            "std": std_filter,
            "subject": subject_filter,
        })
        candidate_rows = cur.fetchall()

    deleted: List[Dict[str, Any]] = []
    for row in candidate_rows:
        record = row.get("lecture_data") or {}
        metadata = record.get("metadata") or {}
        division_value = metadata.get("division") or metadata.get("section")
        division_slug = _slugify(division_value) if division_value else None
        if division_filter and division_slug != division_filter:
            continue

        lecture_entry = {
            "lecture_id": row.get("lecture_uid"),
            "title": record.get("title") or row.get("chapter_title"),

            "std": metadata.get("std") or metadata.get("class") or row.get("std"),
            "subject": metadata.get("subject") or row.get("subject"),
            "division": division_value,
            "std_slug": _slugify(metadata.get("std") or metadata.get("class") or row.get("std")),
            "subject_slug": _slugify(metadata.get("subject") or row.get("subject")),
            "division_slug": division_slug,
        }

        with get_pg_cursor() as cur:
            cur.execute("DELETE FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": row.get("lecture_uid")})
        deleted.append(lecture_entry)

    return deleted


async def update_slide(
    lecture_id: str,
    slide_number: int,
    slide_updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Update a specific slide in a lecture."""
    record = await get_lecture(lecture_id)
    slides = record.get("slides") or []

    for slide in slides:
        if slide.get("number") == slide_number:
            slide.update(slide_updates)
            break

    if "narration" in slide_updates:
        record["context"] = "\n\n".join(
            slide.get("narration", "") for slide in slides if slide.get("narration")
        )

    record["slides"] = slides
    return await update_lecture(lecture_id, record)


async def delete_lecture(lecture_id: str) -> bool:
    """Delete a lecture by ID."""
    with get_pg_cursor() as cur:
        cur.execute("SELECT id FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
        row = cur.fetchone()
    
    if not row:
        return False

    with get_pg_cursor() as cur:
        cur.execute("DELETE FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
    return True


async def list_lectures(
    *,
    language: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    division: Optional[str] = None,
    admin_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List lectures with optional filtering."""
    with get_pg_cursor() as cur:
        query = "SELECT * FROM lecture_gen"
        params: Dict[str, Any] = {}

        if admin_id is not None:
            query += " WHERE admin_id = %(admin_id)s"
            params["admin_id"] = admin_id

        query += " ORDER BY created_at DESC"
        cur.execute(query, params)
        rows = cur.fetchall()

    std_filter = _slugify(std) if std else None
    subject_filter = _slugify(subject) if subject else None
    division_filter = _slugify(division) if division else None
    lang_filter = (language or "").lower() if language else None

    summaries: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("lecture_data"):
            continue
        record = _clone_record(row.get("lecture_data"))
        metadata = record.get("metadata") or {}

        if lang_filter and (record.get("language") or "").lower() != lang_filter:
            continue

        std_value = metadata.get("std") or metadata.get("class") or row.get("std") or "general"
        subject_value = metadata.get("subject") or row.get("subject") or "lecture"
        division_value = metadata.get("division") or metadata.get("section")

        if std_filter and _slugify(std_value) != std_filter:
            continue
        if subject_filter and _slugify(subject_value) != subject_filter:
            continue
        if division_filter and _slugify(division_value) != division_filter:
            continue

        summary = {
            "lecture_id": row.get("lecture_uid"),
            "title": record.get("title") or row.get("lecture_title"),
            "language": record.get("language"),
            "total_slides": record.get("total_slides"),
            "estimated_duration": record.get("estimated_duration"),
            "created_at": record.get("created_at"),
            "fallback_used": record.get("fallback_used", False),
            "lecture_url": record.get("lecture_url") or row.get("lecture_link"),
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": _slugify(std_value),
            "subject_slug": _slugify(subject_value),
            "division_slug": _slugify(division_value) if division_value else None,
        }
        summaries.append(summary)

    return summaries[offset : offset + limit]


async def list_played_lectures(admin_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return lectures which have a play_count greater than zero."""
    query = "SELECT * FROM lecture_gen"
    params: Dict[str, Any] = {}

    if admin_id is not None:
        query += " WHERE admin_id = %(admin_id)s"
        params["admin_id"] = admin_id

    query += " ORDER BY created_at DESC"
        
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    played: List[Dict[str, Any]] = []
    for row in rows:
        record = row.get("lecture_data") or {}
        play_count = int(record.get("play_count") or 0)
        if play_count <= 0:
            continue

        played.append(
            {
                "lecture_id": row.get("lecture_uid"),
                "title": record.get("title") or row.get("lecture_title"),
                "language": record.get("language"),
                "play_count": play_count,
                "last_played_at": record.get("last_played_at"),
                "lecture_url": record.get("lecture_url") or row.get("lecture_link"),
            }
        )

    played.sort(key=lambda item: item.get("last_played_at") or "", reverse=True)
    return played


async def get_class_subject_filters() -> Dict[str, Any]:
    """Return normalized class/subject combinations present in the DB."""
    with get_pg_cursor() as cur:
        cur.execute(
            "SELECT std, subject FROM lecture_gen WHERE std IS NOT NULL AND subject IS NOT NULL"
        )
        rows = cur.fetchall()

    class_map: Dict[str, set] = {}
    for row in rows:
        std_value = (row.get("std") or "").strip()
        subject_value = (row.get("subject") or "").strip()
        if not std_value or not subject_value:
            continue
        class_map.setdefault(std_value, set()).add(subject_value)

    normalized_classes: List[Dict[str, Any]] = []
    for entry in sorted(class_map.items(), key=lambda item: _sort_key(item[0])):
        normalized_classes.append(
            {
                "std": entry[0],
                "subject": sorted(entry[1]),
            }
        )

    return {"classes": normalized_classes}


async def lecture_exists(lecture_id: str) -> bool:
    """Check if a lecture exists."""
    with get_pg_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s LIMIT 1",
            {"lecture_uid": lecture_id},
        )
        return cur.fetchone() is not None


async def record_play(lecture_id: str) -> Dict[str, Any]:
    """Increment play count for a lecture."""
    record = await get_lecture(lecture_id)
    play_count = int(record.get("play_count") or 0) + 1
    timestamp = datetime.utcnow().isoformat()

    record.update({"play_count": play_count, "last_played_at": timestamp})
    return await update_lecture(lecture_id, record)


async def get_lecture_stats() -> Dict[str, Any]:
    """Compute aggregate statistics for lectures."""
    with get_pg_cursor() as cur:
        cur.execute("SELECT lecture_data FROM lecture_gen")
        rows = cur.fetchall()

    stats = {
        "total_lectures": 0,
        "by_language": {},
        "fallback_count": 0,
        "total_slides": 0,
    }

    for row in rows:
        record = row.get("lecture_data") or {}
        stats["total_lectures"] += 1
        language = record.get("language", "Unknown")
        stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
        if record.get("fallback_used"):
            stats["fallback_count"] += 1
        stats["total_slides"] += record.get("total_slides", 0) or 0

    return stats


async def get_source_text(lecture_id: str) -> str:
    """Return saved source text for a lecture."""
    record = await get_lecture(lecture_id)
    source_text = record.get("source_text")
    if not source_text:
        raise FileNotFoundError(f"Source text not found for lecture {lecture_id}")
    return source_text


class LectureRepository:
    """Compatibility wrapper preserving the legacy repository interface."""

    def __init__(self, db: Optional[Any] = None) -> None:
        self._db = db

    def _fetch_row(self, lecture_id: str) -> Optional[Any]:
        """Fetch a lecture row from the database by ID."""
        with get_pg_cursor() as cur:
            cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
            return cur.fetchone()

    async def create_lecture(self, **kwargs: Any) -> Dict[str, Any]:
        return await create_lecture(**kwargs)

    async def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        return await get_lecture(lecture_id)

    async def update_lecture(self, lecture_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return await update_lecture(lecture_id, updates)

    async def delete_lectures_by_metadata(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return await delete_lectures_by_metadata(**kwargs)

    async def update_slide(self, lecture_id: str, slide_number: int, slide_updates: Dict[str, Any]) -> Dict[str, Any]:
        return await update_slide(lecture_id, slide_number, slide_updates)

    async def delete_lecture(self, lecture_id: str) -> bool:
        return await delete_lecture(lecture_id)

    async def list_lectures(
        self,
        *,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        std: Optional[str] = None,
        subject: Optional[str] = None,
        division: Optional[str] = None,
        admin_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await list_lectures(
            language=language,
            limit=limit,
            offset=offset,
            std=std,
            subject=subject,
            division=division,
            admin_id=admin_id,
        )

    async def list_played_lectures(self, admin_id: Optional[int] = None) -> List[Dict[str, Any]]:
        return await list_played_lectures(admin_id=admin_id)

    async def get_class_subject_filters(self) -> Dict[str, Any]:
        return await get_class_subject_filters()

    async def lecture_exists(self, lecture_id: str) -> bool:
        return await lecture_exists(lecture_id)

    async def record_play(self, lecture_id: str) -> Dict[str, Any]:
        return await record_play(lecture_id)

    async def get_lecture_stats(self) -> Dict[str, Any]:
        return await get_lecture_stats()

    async def get_source_text(self, lecture_id: str) -> str:
        return await get_source_text(lecture_id)
