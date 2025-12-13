# app/repositories/chapter_material_repository.py

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.schemas.admin_schema import WorkType
from app.utils.file_handler import (
    save_uploaded_file,
    delete_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
    get_file_url, #-------------
)
from app.config import get_settings
from app.postgres import get_pg_cursor, get_connection

logger = logging.getLogger(__name__)

# Constants (keep same as original)
PDF_MAX_SIZE = 50 * 1024 * 1024  # 15MB
DEFAULT_MIN_DURATION = 5
DEFAULT_MAX_DURATION = 180
MAX_ASSISTANT_SUGGESTIONS = 10
DEFAULT_LANGUAGE_CODE = "eng"

LANGUAGE_OUTPUT_RULES: Dict[str, Dict[str, str]] = {
    "eng": {
        "label": "English",
        "instruction": "The PDF language is English. Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in English.",
    },
    "hin": {
        "label": "Hindi (हिंदी)",
        "instruction": "The PDF language is Hindi (हिंदी). Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in Hindi (हिंदी).",
    },
    "guj": {
        "label": "Gujarati (ગુજરાતી)",
        "instruction": "The PDF language is Gujarati (ગુજરાતी). Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in Gujarati (ગુજરાતી).",
    },
}

SUPPORTED_LANGUAGES = [
    {"value": "English", "label": "English"},
    {"value": "Hindi", "label": "हिंदी / Hindi"},
    {"value": "Gujarati", "label": "ગુજરાતી / Gujarati"},
]

DURATION_OPTIONS = [30, 45, 60]

# Cache for is_global column existence check
_IS_GLOBAL_COLUMN_EXISTS = None

def _check_is_global_column_exists() -> bool:
    """Check if is_global column exists in chapter_materials table (cached)."""
    global _IS_GLOBAL_COLUMN_EXISTS
    if _IS_GLOBAL_COLUMN_EXISTS is not None:
        return _IS_GLOBAL_COLUMN_EXISTS
    
    try:
        check_query = "SELECT column_name FROM information_schema.columns WHERE table_name='chapter_materials' AND column_name='is_global'"
        with get_pg_cursor() as cur:
            cur.execute(check_query)
            _IS_GLOBAL_COLUMN_EXISTS = cur.fetchone() is not None
        if _IS_GLOBAL_COLUMN_EXISTS:
            logger.info("is_global column found in chapter_materials table")
        else:
            logger.warning("is_global column NOT found in chapter_materials table - global materials will not be visible")
    except Exception as e:
        logger.warning(f"Error checking for is_global column: {e}")
        _IS_GLOBAL_COLUMN_EXISTS = False
    
    return _IS_GLOBAL_COLUMN_EXISTS

def _reset_is_global_column_cache() -> None:
    """Reset the cache for is_global column check (useful after schema changes)."""
    global _IS_GLOBAL_COLUMN_EXISTS
    _IS_GLOBAL_COLUMN_EXISTS = None


# -------------------------
# Simple DB operations
# -------------------------

def create_chapter_material(
    *,
    admin_id: int,
    std: str,
    subject: str,
    sem: str,
    board: str,
    chapter_number: str,
    chapter_title: Optional[str] = None,
    file_info: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_title = (chapter_title or chapter_number).strip()
    
    # Check if is_global column exists
    has_is_global = _check_is_global_column_exists()
    
    # Build INSERT query based on whether is_global column exists
    if has_is_global:
        query = """
            INSERT INTO chapter_materials (admin_id, std, sem, board, subject, chapter_number, chapter_title, file_name, file_path, file_size, is_global)
            VALUES (%(admin_id)s, %(std)s, %(sem)s, %(board)s, %(subject)s, %(chapter_number)s, %(chapter_title)s, %(file_name)s, %(file_path)s, %(file_size)s, %(is_global)s)
            RETURNING id, admin_id, std, sem, board, subject, chapter_number, chapter_title, file_name, file_path, file_size, is_global, created_at, updated_at
        """
        params = {
            "admin_id": admin_id,
            "std": std.strip(),
            "sem": sem.strip() if sem else "",
            "board": board.strip() if board else "",
            "subject": subject.strip(),
            "chapter_number": chapter_number.strip(),
            "chapter_title": normalized_title,
            "file_name": file_info["filename"],
            "file_path": file_info["s3_url"],
            "file_size": file_info["file_size"],
            "is_global": file_info.get("is_global", False),
        }
    else:
        query = """
            INSERT INTO chapter_materials (admin_id, std, sem, board, subject, chapter_number, chapter_title, file_name, file_path, file_size)
            VALUES (%(admin_id)s, %(std)s, %(sem)s, %(board)s, %(subject)s, %(chapter_number)s, %(chapter_title)s, %(file_name)s, %(file_path)s, %(file_size)s)
            RETURNING id, admin_id, std, sem, board, subject, chapter_number, chapter_title, file_name, file_path, file_size, created_at, updated_at
        """
        params = {
            "admin_id": admin_id,
            "std": std.strip(),
            "sem": sem.strip() if sem else "",
            "board": board.strip() if board else "",
            "subject": subject.strip(),
            "chapter_number": chapter_number.strip(),
            "chapter_title": normalized_title,
            "file_name": file_info["filename"],
            "file_path": file_info["s3_url"],
            "file_size": file_info["file_size"],
        }
    
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        result = cur.fetchone()
    
    return result if result else {}

def update_chapter_material_overrides(
    material_id: int,
    *,
    chapter_title_override: Optional[str] = None,
    topic_title_override: Optional[str] = None,
    video_duration_minutes: Optional[int] = None,
    video_resolution: Optional[str] = None,
) -> Dict[str, Any]:
    updates = {}
    if chapter_title_override is not None:
        updates["chapter_title_override"] = chapter_title_override.strip() or None
    if topic_title_override is not None:
        updates["topic_title_override"] = topic_title_override.strip() or None
    if video_duration_minutes is not None:
        updates["video_duration_minutes"] = max(video_duration_minutes, 0)
    if video_resolution is not None:
        updates["video_resolution"] = video_resolution.strip() or None

    if not updates:
        return {}

    set_clause = ", ".join([f"{k} = %({k})s" for k in updates.keys()])
    query = f"UPDATE chapter_materials SET {set_clause} WHERE id = %(id)s RETURNING *"
    updates["id"] = material_id

    with get_pg_cursor() as cur:
        cur.execute(query, updates)
        result = cur.fetchone()
    
    return result if result else {}


def get_chapter_material(material_id: int) -> Optional[Dict[str, Any]]:
    query = "SELECT * FROM chapter_materials WHERE id = %(id)s"
    with get_pg_cursor() as cur:
        cur.execute(query, {"id": material_id})
        result = cur.fetchone()
    return result


def list_chapter_materials(
    admin_id: int,
    std: Optional[str] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Check if is_global column exists
    has_is_global = _check_is_global_column_exists()
    
    # Build WHERE clause based on whether is_global column exists
    if has_is_global:
        where_clauses = ["(admin_id = %(admin_id)s OR is_global = TRUE)"]
    else:
        where_clauses = ["admin_id = %(admin_id)s"]
    params = {"admin_id": admin_id}
    
    # Require both std and subject for filtering
    if std and subject:
        std_clean = std.strip().lower()
        subject_clean = subject.strip().lower()
        where_clauses.append("LOWER(std) = %(std)s")
        params["std"] = std_clean
        
        # Try exact match first
        where_clause_str = " AND ".join(where_clauses + ["LOWER(subject) = %(subject)s"])
        query = f"SELECT * FROM chapter_materials WHERE {where_clause_str} ORDER BY created_at DESC"
        params["subject"] = subject_clean
        
        with get_pg_cursor() as cur:
            cur.execute(query, params)
            exact_results = cur.fetchall()
        
        if exact_results:
            return exact_results
        
        # Try partial/contains match
        where_clause_str = " AND ".join(where_clauses + ["LOWER(subject) LIKE %(subject_like)s"])
        query = f"SELECT * FROM chapter_materials WHERE {where_clause_str} ORDER BY created_at DESC"
        params["subject_like"] = f"%{subject_clean}%"
        
        with get_pg_cursor() as cur:
            cur.execute(query, params)
            contains_results = cur.fetchall()
        
        if contains_results:
            return contains_results
        
        # Try variations
        variations = get_subject_variations(subject_clean)
        if variations:
            placeholders = ",".join([f"%(var_{i})s" for i in range(len(variations))])
            where_clause_str = " AND ".join(where_clauses + [f"LOWER(subject) IN ({placeholders})"])
            query = f"SELECT * FROM chapter_materials WHERE {where_clause_str} ORDER BY created_at DESC"
            for i, var in enumerate(variations):
                params[f"var_{i}"] = var
            
            with get_pg_cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
        
        return []
    
    elif std:
        std_clean = std.strip().lower()
        where_clauses.append("LOWER(std) = %(std)s")
        params["std"] = std_clean
    
    elif subject:
        subject_clean = subject.strip().lower()
        where_clauses.append("LOWER(subject) = %(subject)s")
        params["subject"] = subject_clean
    
    where_clause_str = " AND ".join(where_clauses)
    query = f"SELECT * FROM chapter_materials WHERE {where_clause_str} ORDER BY created_at DESC"
    
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def list_materials(
    admin_id: int,
    *,
    std: Optional[str] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compatibility helper returning serialized chapter materials."""
    return list_chapter_materials(admin_id, std=std, subject=subject)


def get_chapter_filter_options(
    admin_id: int,
) -> Dict[str, Any]:
    query = "SELECT std, subject, chapter_number as chapter FROM chapter_materials WHERE admin_id = %(admin_id)s"
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        rows = cur.fetchall()

    classes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        std_value = (row.std or "").strip()
        subject_value = (row.subject or "").strip()
        chapter_value = (row.chapter or "").strip()
        if not std_value or not subject_value:
            continue

        std_slug = std_value.lower().replace(" ", "_")
        std_entry = classes.setdefault(
            std_slug,
            {
                "label": std_value,
                "value": std_value,
                "slug": std_slug,
                "subjects": {},
            },
        )

        subject_slug = subject_value.lower().replace(" ", "_")
        subject_entry = std_entry["subjects"].setdefault(
            subject_slug,
            {
                "label": subject_value,
                "value": subject_value,
                "slug": subject_slug,
                "chapters": set(),
            },
        )

        if chapter_value:
            subject_entry["chapters"].add(chapter_value)

    response: List[Dict[str, Any]] = []
    for std_entry in classes.values():
        subjects: List[Dict[str, Any]] = []
        for subject_entry in std_entry["subjects"].values():
            subjects.append(
                {
                    "label": subject_entry["label"],
                    "value": subject_entry["value"],
                    "slug": subject_entry["slug"],
                    "chapters": sorted(subject_entry["chapters"]),
                }
            )
        response.append(
            {
                "label": std_entry["label"],
                "value": std_entry["value"],
                "slug": std_entry["slug"],
                "subjects": sorted(subjects, key=lambda s: s["label"].lower()),
            }
        )

    response.sort(key=lambda entry: entry["label"].lower())
    return {"classes": response}


def list_chapters_for_selection(
    admin_id: int,
    std: str,
    subject: str,
) -> List[str]:
    query = """
        SELECT chapter_title_override, chapter_title, chapter_number
        FROM chapter_materials
        WHERE admin_id = %(admin_id)s
        AND LOWER(std) = %(std)s
        AND LOWER(subject) = %(subject)s
    """
    with get_pg_cursor() as cur:
        cur.execute(query, {
            "admin_id": admin_id,
            "std": std.strip().lower(),
            "subject": subject.strip().lower(),
        })
        rows = cur.fetchall()

    title_candidates = {
        (row.get("chapter_title_override") or row.get("chapter_title") or "").strip()
        for row in rows
        if (row.get("chapter_title_override") or row.get("chapter_title"))
    }
    if title_candidates:
        return sorted(title_candidates, key=lambda value: value.lower())

    number_candidates = {
        (row.get("chapter_number") or "").strip()
        for row in rows
        if row.get("chapter_number")
    }
    return sorted(number_candidates)


def list_subjects_for_std(
    admin_id: int,
    std: str,
) -> List[str]:
    query = """
        SELECT DISTINCT subject FROM chapter_materials
        WHERE admin_id = %(admin_id)s
        AND LOWER(std) = %(std)s
        AND subject IS NOT NULL
    """
    with get_pg_cursor() as cur:
        cur.execute(query, {
            "admin_id": admin_id,
            "std": std.strip().lower(),
        })
        rows = cur.fetchall()
    
    seen: Dict[str, str] = {}
    for row in rows:
        subject_value = (row.get("subject") or "").strip()
        if not subject_value:
            continue
        normalized = subject_value.lower()
        seen.setdefault(normalized, subject_value)
    return sorted(seen.values(), key=lambda value: value.lower())


def list_standards_for_admin(
    admin_id: int,
) -> List[str]:
    query = """
        SELECT DISTINCT std FROM chapter_materials
        WHERE admin_id = %(admin_id)s
        AND std IS NOT NULL
    """
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        rows = cur.fetchall()
    
    seen: Dict[str, str] = {}
    for row in rows:
        std_value = (row.get("std") or "").strip()
        if not std_value:
            continue
        normalized = std_value.lower()
        seen.setdefault(normalized, std_value)
    return sorted(seen.values(), key=lambda value: value.lower())


def get_subject_variations(subject: str) -> List[str]:
    """Generate common subject variations and spelling corrections"""
    variations = [subject]
    
    # Common subject mappings
    subject_mappings = {
        'science': ['science', 'sci', 'scince', 'sceince', 'scienc'],
        'math': ['math', 'maths', 'mathematics', 'mathmatic'],
        'english': ['english', 'eng', 'engish', 'englis'],
        'hindi': ['hindi', 'hindi', 'hindi'],
        'social science': ['social science', 'socialscience', 'social', 'sst', 'social studies'],
        'physics': ['physics', 'physic', 'phy'],
        'chemistry': ['chemistry', 'chem', 'chemical'],
        'biology': ['biology', 'bio', 'bilogy'],
        'computer': ['computer', 'comp', 'computer science', 'cs', 'it'],
        'history': ['history', 'hist', 'histry'],
        'geography': ['geography', 'geo', 'geography'],
        'economics': ['economics', 'eco', 'economics', 'econ'],
    }
    
    # Find matching variations
    for key, values in subject_mappings.items():
        if subject in values or key in subject or any(v in subject for v in values):
            variations.extend(values)
            variations.append(key)
            break
    
    # Remove duplicates
    return list(set(variations))


def list_recent_chapter_materials(admin_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    query = """
        SELECT * FROM chapter_materials
        WHERE admin_id = %(admin_id)s
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id, "limit": limit})
        return cur.fetchall()


def update_chapter_material_db(
    material_id: int,
    *,
    std: str,
    subject: str,
    sem: str,
    board: str,
    chapter_number: str,
    chapter_title: Optional[str] = None,
    new_file_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Get current material first
    current = get_chapter_material(material_id)
    if not current:
        return {}
    
    normalized_title = chapter_title.strip() if chapter_title else (current.get("chapter_title") or chapter_number)
    
    query = """
        UPDATE chapter_materials
        SET std = %(std)s,
            subject = %(subject)s,
            sem = %(sem)s,
            board = %(board)s,
            chapter_number = %(chapter_number)s,
            chapter_title = %(chapter_title)s
        WHERE id = %(id)s
        RETURNING *
    """
    
    params = {
        "id": material_id,
        "std": std.strip(),
        "subject": subject.strip(),
        "sem": sem.strip() if sem else "",
        "board": board.strip() if board else "",
        "chapter_number": chapter_number.strip(),
        "chapter_title": normalized_title,
    }
    
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        result = cur.fetchone()
    
    # Delete old file if new file info provided
    if new_file_info is not None and current.get("file_path"):
        try:
            delete_file(current["file_path"])
        except Exception:
            logger.warning("Failed to delete old file: %s", current.get("file_path"))
    
    return result if result else {}


def delete_chapter_material_db(material_id: int) -> bool:
    # Get file path before deleting
    current = get_chapter_material(material_id)
    if not current:
        return False
    
    file_path = current.get("file_path")
    
    query = "DELETE FROM chapter_materials WHERE id = %(id)s"
    with get_pg_cursor() as cur:
        cur.execute(query, {"id": material_id})
    
    try:
        if file_path:
            delete_file(file_path)
    except Exception:
        logger.warning("Failed to delete material file: %s", file_path)
    
    return True


def get_dashboard_stats(admin_id: int) -> Dict[str, int]:
    query = """
        SELECT
            COUNT(*) as total_materials,
            COUNT(DISTINCT subject) as unique_subjects,
            COUNT(DISTINCT std) as unique_classes
        FROM chapter_materials
        WHERE admin_id = %(admin_id)s
    """
    
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        result = cur.fetchone()
    
    return {
        "total_materials": result.get("total_materials", 0) if result else 0,
        "unique_subjects": result.get("unique_subjects", 0) if result else 0,
        "unique_classes": result.get("unique_classes", 0) if result else 0,
    }


def formatFileSize(bytes):
    """Format file size in human readable format"""
    if not bytes:
        return '—'
    
    mb = bytes / (1024 * 1024)
    if mb < 0.5:
        kb = bytes / 1024
        return f"{kb:.1f} KB"
    
    return f"{mb:.2f} MB"


def get_chapter_overview_data(admin_id: int) -> List[Dict[str, Any]]:
    """
    Get chapter overview data with lecture information including:
    - subject, chapter title, topics, lecture size, video info
    - list of generated lectures pulled from lecture_gen table
    """
    settings = get_settings()

    query = """
        SELECT 
            cm.id,
            cm.subject,
            cm.std,
            cm.sem,
            cm.board,
            cm.chapter_title,
            cm.chapter_number,
            cm.chapter_title_override,
            cm.topic_title_override,
            cm.video_duration_minutes,
            cm.video_resolution,
            cm.file_name,
            cm.file_path,
            cm.file_size,
            cm.admin_id,
            cm.created_at AS material_created_at,
            cm.updated_at AS material_updated_at,
            lg.id AS lecture_id,
            lg.lecture_link,
            lg.lecture_uid,
            lg.chapter_title AS lecture_chapter_title,
            lg.std AS lecture_std,
            lg.subject AS lecture_subject,
            lg.sem AS lecture_sem,
            lg.board AS lecture_board,
            lg.created_at AS lecture_created_at,
            lg.updated_at AS lecture_updated_at,
            lg.lecture_data
        FROM chapter_materials cm
        LEFT JOIN lecture_gen lg 
            ON lg.material_id = cm.id
            AND lg.admin_id = cm.admin_id
        WHERE cm.admin_id = %(admin_id)s
        ORDER BY cm.created_at DESC, lg.created_at DESC NULLS LAST
    """
    
    storage_base = Path("./storage/lectures")
    lecture_asset_cache: Dict[str, Dict[str, Any]] = {}

    def _get_lecture_assets(lecture_uid: Optional[str]) -> Dict[str, Any]:
        if not lecture_uid:
            return {"json_size": 0, "video": None}
        if lecture_uid in lecture_asset_cache:
            return lecture_asset_cache[lecture_uid]
        json_size = 0
        video_info = None
        try:
            lecture_folder = storage_base / lecture_uid
            lecture_json_path = lecture_folder / "lecture.json"
            if lecture_json_path.exists():
                json_size = lecture_json_path.stat().st_size
            video_path = lecture_folder / "lecture.mp4"
            if video_path.exists():
                video_info = {"size": video_path.stat().st_size, "path": str(video_path)}
        except Exception:
            pass
        lecture_asset_cache[lecture_uid] = {"json_size": json_size, "video": video_info}
        return lecture_asset_cache[lecture_uid]

    def _extract_video_info(raw: Any) -> Dict[str, Any] | None:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, dict):
            return None

        # Direct keys commonly used by generators
        for key in ("video", "video_info", "video_meta"):
            value = raw.get(key)
            if value:
                return value if isinstance(value, dict) else {"value": value}

        assets = raw.get("assets")
        if isinstance(assets, dict):
            for key in ("video", "video_file", "videoAsset"):
                value = assets.get(key)
                if value:
                    return value if isinstance(value, dict) else {"value": value}

        link = raw.get("video_link") or raw.get("video_url")
        if link:
            return {"link": link}

        return None

    def _topics_from_payload(raw: Any) -> List[str]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, dict):
            return []

        slides = raw.get("slides")
        if not isinstance(slides, list):
            return []

        topics: List[str] = []
        for slide in slides:
            if not isinstance(slide, dict):
                continue
            candidate = (slide.get("title") or "").strip()
            if not candidate or candidate.lower().startswith("i'm sorry"):
                continue
            topics.append(candidate)
            if len(topics) == 5:
                break
        return topics

    def _topic_strings_from_material(topics_data: List[Dict[str, Any]]) -> List[str]:
        results: List[str] = []
        for topic in topics_data:
            if not isinstance(topic, dict):
                continue
            candidate = (topic.get("title") or topic.get("name") or "").strip()
            if not candidate:
                continue
            results.append(candidate)
            if len(results) == 5:
                break
        return results

    def _extract_duration(raw: Any) -> Optional[int]:
        payload: Optional[Dict[str, Any]] = None
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None

        if not isinstance(payload, dict):
            return None

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        candidates = [
            payload.get("estimated_duration"),
            payload.get("requested_duration"),
            metadata.get("duration") if isinstance(metadata, dict) else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        return None

    default_thumbnail_url = (
        getattr(settings, "default_lecture_thumbnail", None)
        or settings.dict().get("default_lecture_thumbnail")
        or "/static/images/lecture-placeholder.png"
    )
    default_duration = (
        getattr(settings, "default_lecture_duration", None)
        or settings.dict().get("default_lecture_duration")
        or 45
    )

    grouped: Dict[int, Dict[str, Any]] = {}
    lectures: List[Dict[str, Any]] = []

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        rows = cur.fetchall()
    
    for row in rows:
        material_id = row["id"]
        material_size_bytes = row.get("file_size") or 0

        if material_id not in grouped:
            topics_data: List[Dict[str, Any]] = []
            extracted_chapter_title = None
            try:
                payload, topics_list = read_topics_file_if_exists(row["admin_id"], material_id)
                if topics_list:
                    topics_data = topics_list[:5]
                if payload and payload.get("chapter_title"):
                    extracted_chapter_title = payload.get("chapter_title")
            except Exception:
                topics_data = []

            chapter_title = (
                row["chapter_title_override"]
                or extracted_chapter_title
                or row["chapter_title"]
                or row["chapter_number"]
            )

            inferred_topic_title = ""
            if row.get("topic_title_override"):
                inferred_topic_title = row["topic_title_override"]
            elif topics_data:
                first_topic = topics_data[0]
                inferred_topic_title = first_topic.get("title") or first_topic.get("name") or ""

            fallback_topic_titles = _topic_strings_from_material(topics_data)

            grouped[material_id] = {
                "material_id": material_id,
                "subject": row["subject"],
                "std": row["std"],
                "chapter": chapter_title,
                "file_size_bytes": material_size_bytes,
                "fallback_topics": fallback_topic_titles,
                "admin_id": row.get("admin_id"),
                "generated_lectures": [],

                "_latest_lecture_ts": None,

                "_latest_lecture_size": 0,

                "_latest_video_info": None,
            }

        lecture_id = row.get("lecture_id")
        if lecture_id:
            lecture_assets = _get_lecture_assets(row.get("lecture_uid"))
            lecture_size_bytes = lecture_assets.get("json_size", 0) or 0
            video_info = lecture_assets.get("video")

            lecture_data_details = row.get("lecture_data")
            if not video_info:
                video_info = _extract_video_info(lecture_data_details) or {}
            if not video_info and row.get("lecture_link"):
                video_info = {"link": row.get("lecture_link")}

            chapter_entry = grouped[material_id]
            lecture_topics = _topics_from_payload(lecture_data_details)
            if not lecture_topics:
                lecture_topics = chapter_entry.get("fallback_topics", [])

            lecture_duration = _extract_duration(lecture_data_details)
            if lecture_duration is None:
                lecture_duration = row.get("video_duration_minutes") or default_duration

            thumbnail_url: Optional[str] = None
            if isinstance(video_info, dict):
                thumb_candidate = video_info.get("thumbnail") or video_info.get("thumbnail_url") or video_info.get("thumb")
                if isinstance(thumb_candidate, str) and thumb_candidate.strip():
                    thumbnail_url = thumb_candidate.strip()
            if not thumbnail_url:
                thumbnail_url = default_thumbnail_url

            lecture_uid = row.get("lecture_uid")
            display_size_bytes = lecture_size_bytes or chapter_entry["file_size_bytes"]
            chapter_value = (
                row.get("lecture_chapter_title")
                or row.get("chapter_title")
                or chapter_entry["chapter"]
            )

            lecture_payload = {
                "id": str(lecture_uid or lecture_id),
                "lecture_uid": lecture_uid,
                "lecture_uuid": lecture_uid,
                "admin_id": chapter_entry.get("admin_id"),
                "material_id": material_id,
                "lecture_link": row.get("lecture_link"),
                "std": row.get("lecture_std") or chapter_entry["std"],
                "subject": row.get("lecture_subject") or chapter_entry["subject"],
                "chapter": chapter_value,
                "topics": lecture_topics,
                "size": formatFileSize(display_size_bytes),
                "video_duration_minutes": lecture_duration,
                "thumbnail_url": thumbnail_url,
                "_created_at": row.get("lecture_created_at"),
            }

            lectures.append(lecture_payload)

    lectures.sort(key=lambda item: item.get("_created_at") or datetime.min, reverse=True)
    for lecture in lectures:
        lecture.pop("_created_at", None)

    return lectures


# -------------------------
# Topic file helpers & persistence
# -------------------------

def _load_topics_path(admin_id: int, material_id: int) -> str:
    topics_folder = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}")
    return os.path.join(topics_folder, f"extracted_topics_{material_id}.json")


def load_material_topics(admin_id: int, material_id: int) -> Dict[str, Any]:
    topics_path = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}", f"extracted_topics_{material_id}.json")
    if not os.path.exists(topics_path):
        raise FileNotFoundError("Topics not found")
    with open(topics_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    _ensure_topic_ids_for_payload(admin_id, material_id, payload)
    return payload


def persist_material_topics(admin_id: int, material_id: int, payload: Dict[str, Any]) -> None:
    admin_folder = f"chapter_materials/admin_{admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)

    topics_json_path = os.path.join(topics_dir, f"extracted_topics_{material_id}.json")
    try:
        with open(topics_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Failed to persist topics for material %s: %s", material_id, exc)


def save_extracted_topics_files(admin_id: int, material_id: int, extraction: Dict[str, Any]) -> Tuple[str, str]:
    admin_dir = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}")
    os.makedirs(admin_dir, exist_ok=True)

    topics = extraction.get("topics") or []
    if isinstance(topics, list):
        _assign_topic_ids(topics)

    txt_path = os.path.join(admin_dir, f"extracted_topics_{material_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(extraction.get("topics_text", ""))

    json_path = os.path.join(admin_dir, f"extracted_topics_{material_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(extraction, jf, indent=2, ensure_ascii=False)

    return txt_path, json_path


def read_topics_file_if_exists(admin_id: int, material_id: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    topics_file = Path(UPLOAD_DIR) / f"chapter_materials/admin_{admin_id}" / f"extracted_topics_{material_id}.json"
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
                topics_list = payload.get("topics", [])
                _ensure_topic_ids_for_payload(admin_id, material_id, payload)
                topics_list = payload.get("topics", [])
                return payload, topics_list
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading topics file for material {material_id}: {e}")
            return None, []
    return None, []


def _assign_topic_ids(topics: List[Dict[str, Any]]) -> bool:
    updated = False
    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            continue
        expected = str(idx)
        if topic.get("topic_id") != expected:
            topic["topic_id"] = expected
            updated = True
    return updated


def _ensure_topic_ids_for_payload(admin_id: int, material_id: int, payload: Dict[str, Any]) -> None:
    topics = payload.get("topics")
    if isinstance(topics, list) and _assign_topic_ids(topics):
        persist_material_topics(admin_id, material_id, payload)


def _assistant_suggestions_path(admin_id: int, material_id: int) -> str:
    return os.path.join(
        UPLOAD_DIR,
        f"chapter_materials/admin_{admin_id}",
        f"assistant_suggestions_{material_id}.json",
    )


def persist_assistant_suggestions(admin_id: int, material_id: int, suggestions: List[Dict[str, Any]]) -> None:
    path = _assistant_suggestions_path(admin_id, material_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "admin_id": admin_id,
        "material_id": material_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "suggestions": suggestions,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_assistant_suggestions(admin_id: int, material_id: int) -> Dict[str, Any]:
    path = _assistant_suggestions_path(admin_id, material_id)
    if not os.path.exists(path):
        return {"suggestions": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read assistant suggestions cache for material %s: %s", material_id, exc)
        return {"suggestions": []}


def get_cached_suggestions_by_ids(
    admin_id: int,
    material_id: int,
    suggestion_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    cache = load_assistant_suggestions(admin_id, material_id)
    suggestions = cache.get("suggestions", []) or []
    index = {str(item.get("suggestion_id")): item for item in suggestions if item.get("suggestion_id")}
    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []
    for sid in suggestion_ids:
        entry = index.get(str(sid))
        if entry:
            resolved.append(entry)
        else:
            missing.append(str(sid))
    return resolved, missing


def get_topics_by_ids(
    admin_id: int,
    material_id: int,
    topic_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    payload, topics_list = read_topics_file_if_exists(admin_id, material_id)
    topics_list = topics_list or []
    index = {
        str(topic.get("topic_id")): topic
        for topic in topics_list
        if isinstance(topic, dict) and topic.get("topic_id")
    }
    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []
    for tid in topic_ids:
        entry = index.get(str(tid))
        if entry:
            resolved.append(entry)
        else:
            missing.append(str(tid))
    return resolved, missing


# -------------------------
# Utilities for text conversion (topic -> text)
# -------------------------

def _safe_join(parts: List[str]) -> str:
    return "\n".join(filter(None, parts))


def topic_to_text(topic: Dict[str, Any]) -> str:
    lines: List[str] = []

    title = topic.get("title")
    if title:
        lines.append(f"Topic: {title}")

    summary = topic.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    if topic.get("content"):
        lines.append(f"Content: {topic['content']}")

    subtopics = topic.get("subtopics", [])
    if isinstance(subtopics, list) and subtopics:
        sub_lines: List[str] = []
        for subtopic in subtopics:
            if isinstance(subtopic, dict):
                stitle = subtopic.get("title")
                snarration = subtopic.get("narration")
                part = _safe_join([
                    f"Subtopic: {stitle}" if stitle else None,
                    snarration,
                ])
                if part:
                    sub_lines.append(part)
            else:
                sub_lines.append(f"Subtopic: {subtopic}")
        if sub_lines:
            lines.append("\n".join(sub_lines))

    return "\n".join(lines)


# -------------------------
# Manual topic append & assistant topics add
# -------------------------

def append_manual_topic_to_file(admin_id: int, material_id: int, chapter_number: str, topic_payload: Dict[str, Any]) -> Dict[str, Any]:
    admin_folder = f"chapter_materials/admin_{admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)
    topics_path = os.path.join(topics_dir, f"extracted_topics_{material_id}.json")

    topics_payload: dict = {
        "material_id": material_id,
        "language_code": None,
        "language_label": None,
        "topics": [],
        "headings": [],
        "excerpt": "",
        "topics_text": "",
        "chapter_title": chapter_number,
    }

    if os.path.exists(topics_path):
        try:
            with open(topics_path, "r", encoding="utf-8") as tf:
                topics_payload = json.load(tf)
        except json.JSONDecodeError:
            pass

    topics_list = topics_payload.setdefault("topics", [])
    _assign_topic_ids(topics_list)
    if not topic_payload.get("topic_id"):
        topic_payload["topic_id"] = str(len([t for t in topics_list if isinstance(t, dict)]) + 1)
    topics_list.append(topic_payload)

    with open(topics_path, "w", encoding="utf-8") as tf:
        json.dump(topics_payload, tf, ensure_ascii=False, indent=2)

    topics_payload.setdefault("chapter_title", chapter_number)
    return {"topic": topic_payload, "topics": topics_list, "chapter_title": topics_payload.get("chapter_title")}


def add_assistant_topics_to_file(admin_id: int, material_id: int, chapter_number: str, selected_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    admin_folder = f"chapter_materials/admin_{admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)
    topics_path = os.path.join(topics_dir, f"extracted_topics_{material_id}.json")

    topics_payload: dict = {
        "material_id": material_id,
        "language_code": None,
        "language_label": None,
        "topics": [],
        "headings": [],
        "excerpt": "",
        "topics_text": "",
        "chapter_title": chapter_number,
    }

    if os.path.exists(topics_path):
        try:
            with open(topics_path, "r", encoding="utf-8") as tf:
                topics_payload = json.load(tf)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse topics file for material {material_id}")

    existing_topics = topics_payload.get("topics", [])
    _assign_topic_ids(existing_topics)
    existing_titles = {
        str(topic.get("title", "")).strip().lower()
        for topic in existing_topics
        if isinstance(topic, dict)
    }

    added_topics = []
    skipped_duplicates = []

    for suggestion in selected_suggestions:
        if not isinstance(suggestion, dict):
            continue
        title = str(suggestion.get("title", "")).strip()
        if not title:
            continue
        normalized_title = title.lower()
        if normalized_title in existing_titles:
            skipped_duplicates.append(title)
            continue
        new_topic = {
            "title": title,
            "summary": str(suggestion.get("summary", "")).strip(),
            "supporting_quote": str(suggestion.get("supporting_quote", "")).strip(),
            "is_assistant_generated": True,
            "subtopics": [],
        }
        new_topic["topic_id"] = str(len([t for t in existing_topics if isinstance(t, dict)]) + 1)
        existing_topics.append(new_topic)
        added_topics.append(new_topic)
        existing_titles.add(normalized_title)

    topics_payload["topics"] = existing_topics

    try:
        with open(topics_path, "w", encoding="utf-8") as tf:
            json.dump(topics_payload, tf, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save topics for material {material_id}: {e}")
        raise

    return {
        "added_topics": added_topics,
        "skipped_duplicates": skipped_duplicates,
        "total_topics": len(existing_topics),
        "chapter_title": topics_payload.get("chapter_title"),
    }


# -------------------------
# Read a PDF snippet for assistant context (uses topic_extractor.read_pdf if available)
# -------------------------

def read_pdf_context_for_material(material_file_path: str, max_chars: int = 12_000) -> str:
    try:
        from app.utils.topic_extractor import read_pdf
        return read_pdf(Path(material_file_path))[:max_chars]
    except Exception:
        logger.debug("read_pdf not available or failed; returning empty string")
        return ""