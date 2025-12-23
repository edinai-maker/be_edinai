"""
Lecture Repository - Data Storage and Retrieval Layer
Handles all database operations for lecture storage
"""

import asyncio
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.postgres import get_pg_cursor
from app.models.chapter_material import LectureGen

def _slugify(value: Any) -> str:
    """Convert metadata values to slug format for comparisons."""
    return str(value or "").strip().lower().replace(" ", "_")


def _text_or(value: Any, *, default: Optional[str] = None) -> Optional[str]:
    """Return a clean string representation or the provided default."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    try:
        text = str(value).strip()
    except Exception:
        return default
    return text if text else default


def _ensure_metadata_dict(value: Any) -> Dict[str, Any]:
    """Ensure metadata-like payloads are dictionaries."""
    if isinstance(value, dict):
        return value
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    """Convert various numeric representations into integers, otherwise None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = text.replace(",", "")
        try:
            numeric = float(text)
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return int(numeric)
        except ValueError:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    return None
    return None


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Best-effort conversion to datetime objects for API responses."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        candidates = [text]
        # Support space-delimited timestamps and Z suffix
        if "T" not in text and " " in text:
            candidates.append(text.replace(" ", "T"))
        if text.endswith("Z"):
            candidates.append(f"{text[:-1]}+00:00")
        for candidate in candidates:
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None

# Basic Gujarati/Hindi -> Roman transliteration map for search normalization.
_GUJARATI_ROMAN_MAP = {
    "અ": "a",
    "આ": "aa",
    "ઇ": "i",
    "ઈ": "ii",
    "ઉ": "u",
    "ઊ": "uu",
    "ઋ": "ri",
    "એ": "e",
    "ઐ": "ai",
    "ઓ": "o",
    "ઔ": "au",
    "ઍ": "e",
    "ઑ": "o",
    # Consonants
    "ક": "k",
    "ખ": "kh",
    "ગ": "g",
    "ઘ": "gh",
    "ઙ": "n",
    "ચ": "ch",
    "છ": "chh",
    "જ": "j",
    "ઝ": "jh",
    "ઞ": "n",
    "ટ": "t",
    "ઠ": "th",
    "ડ": "d",
    "ઢ": "dh",
    "ણ": "n",
    "ત": "t",
    "થ": "th",
    "દ": "d",
    "ધ": "dh",
    "ન": "n",
    "પ": "p",
    "ફ": "ph",
    "બ": "b",
    "ભ": "bh",
    "મ": "m",
    "ય": "y",
    "ર": "r",
    "લ": "l",
    "વ": "v",
    "શ": "sh",
    "ષ": "sh",
    "સ": "s",
    "હ": "h",
    "ળ": "l",
    # Anusvara / chandrabindu / visarga
    "ઁ": "n",
    "ં": "n",
    "ઃ": "h",
    # Matras
    "ા": "a",
    "િ": "i",
    "ી": "ii",
    "ુ": "u",
    "ૂ": "uu",
    "ૃ": "ri",
    "ે": "e",
    "ૈ": "ai",
    "ો": "o",
    "ૌ": "au",
    # Digits
    "૦": "0",
    "૧": "1",
    "૨": "2",
    "૩": "3",
    "૪": "4",
    "૫": "5",
    "૬": "6",
    "૭": "7",
    "૮": "8",
    "૯": "9",
    # Hindi Devanagari vowels
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    # Hindi Devanagari consonants
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "n",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "n",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    # Hindi nukta variants (approximate)
    "क़": "k",    "ख़": "kh",
    "ग़": "g",    "ज़": "z",
    "ड़": "d",    "ढ़": "dh",
    "फ़": "f",    # Hindi diacritics
    "ँ": "n",
    "ं": "n",
    "ः": "h",
    # Hindi matras
    "ा": "a",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    # Hindi digits
    "०": "0",
    "१": "1",
    "२": "2",
    "३": "3",
    "४": "4",
    "५": "5",
    "६": "6",
    "७": "7",
    "८": "8",
    "९": "9",
}


def _normalize_title_for_search(value: Any) -> str:
    """Normalize lecture titles and queries for flexible search.

    - Converts Gujarati characters to a simple Roman approximation so that
      WhatsApp-style Roman queries can match Gujarati titles.
    - Lowercases and removes extra whitespace and punctuation.
    """

    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    parts: List[str] = []
    for ch in text:
        if ch in _GUJARATI_ROMAN_MAP:
            parts.append(_GUJARATI_ROMAN_MAP[ch])
        elif ch.isalnum():
            # Keep ASCII letters and digits as-is for Roman text
            parts.append(ch)
        elif ch in {" ", "\t", "\n", "_", "-", ".", ",", "/", "|", "(", ")", "[", "]", "{", "}", ":"}:
            # Normalize various separators to a single space
            parts.append(" ")
        # Other characters are ignored for search purposes

    normalized = "".join(parts)
    # Collapse multiple spaces
    tokens = [token for token in normalized.split(" ") if token]
    return " ".join(tokens)


def _normalize_title_for_fuzzy_match(value: str) -> str:
    """Create a looser representation for matching titles.

    This helper removes vowels from tokens so that small differences in
    transliteration (e.g. Gujarati -> Roman vs WhatsApp spelling) still
    match on the consonant pattern.
    """

    if not value:
        return ""

    vowels = {"a", "e", "i", "o", "u"}
    tokens: List[str] = []
    for raw_token in value.split(" "):
        token = raw_token.strip()
        if not token:
            continue
        stripped = "".join(ch for ch in token if ch not in vowels)
        if stripped:
            tokens.append(stripped)
    return " ".join(tokens)

async def search_lectures_by_title(
    *,
    query: str,
    language: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    division: Optional[str] = None,
    admin_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search lectures by title using Gujarati-aware normalization.

    The search is performed in Python to allow matching Gujarati titles against
    WhatsApp-style Roman queries using [_normalize_title_for_search](cci:1://file:///d:/final-code-main2/backend/app/repository/lecture_repository.py:178:0-208:27).
    """

    normalized_query = _normalize_title_for_search(query)
    if not normalized_query:
        return []

    # Fuzzy representation (vowels stripped) to allow loose transliteration matches
    fuzzy_query = _normalize_title_for_fuzzy_match(normalized_query)

    with get_pg_cursor() as cur:
        sql = "SELECT * FROM lecture_gen"
        params: Dict[str, Any] = {}

        if admin_id is not None:
            sql += " WHERE admin_id = %(admin_id)s"
            params["admin_id"] = admin_id

        sql += " ORDER BY created_at DESC"
        cur.execute(sql, params)
        rows = cur.fetchall()

    std_filter = _slugify(std) if std else None
    subject_filter = _slugify(subject) if subject else None
    division_filter = _slugify(division) if division else None
    lang_filter = (language or "").lower() if language else None

    results: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("lecture_data"):
            continue
        record = _clone_record(row.get("lecture_data"))
        metadata = _ensure_metadata_dict(record.get("metadata"))

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

        title_value = record.get("title") or row.get("lecture_title") or ""
        normalized_title = _normalize_title_for_search(title_value)
        if not normalized_title:
            continue

        # First, try a direct raw substring match to support exact Hindi/Gujarati
        # script queries matching the stored title.
        raw_title = str(title_value or "").strip().lower()
        raw_query = str(query or "").strip().lower()

        if raw_query and raw_query not in raw_title:
            # If raw match fails, fall back to strict normalized and then fuzzy match.
            # First try strict substring match on normalized forms (supports partial matches).
            if normalized_query not in normalized_title:
                # Fall back to fuzzy consonant-pattern matching for loose transliteration
                fuzzy_title = _normalize_title_for_fuzzy_match(normalized_title)
                if fuzzy_query:
                    any_token_match = False
                    for token in fuzzy_query.split(" "):
                        token = token.strip()
                        if not token:
                            continue
                        if token in fuzzy_title:
                            any_token_match = True
                            break
                    if not any_token_match:
                        continue
                else:
                    # No fuzzy query constructed and strict match failed
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
            "cover_photo_url": record.get("cover_photo_url") or row.get("cover_photo_url"),
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": _slugify(std_value),
            "subject_slug": _slugify(subject_value),
            "division_slug": _slugify(division_value) if division_value else None,
        }

        slides = record.get("slides") or []
        bullets: List[str] = []
        for slide in slides:
            if not isinstance(slide, dict):
                continue
            for bullet in slide.get("bullets") or []:
                text = (bullet or "").strip()
                if text:
                    bullets.append(text)

        summary["bullets"] = bullets
        results.append(summary)

    return results[offset : offset + limit]




def _sort_key(value: str) -> Tuple[int, str]:
    """Sort numerically when possible, otherwise lexicographically."""
    try:
        return (0, f"{int(value):02d}")
    except (ValueError, TypeError):
        return (1, (value or "").lower())


def _clone_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of the payload to prevent accidental mutations."""
    return json.loads(json.dumps(payload or {}))

def _maybe_reuse_existing_lecture_id(
    *,
    admin_id: Optional[int],
    material_id: Optional[int],
) -> Optional[str]:
    """
    Return the most recent lecture UID generated for the same admin/material combo.
    This lets us overwrite/regenerate lectures without producing duplicate IDs.
    """
    if not admin_id or not material_id:
        return None
    query = """
        SELECT lecture_uid
        FROM lecture_gen
        WHERE admin_id = %(admin_id)s
          AND material_id = %(material_id)s
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
    """
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id, "material_id": material_id})
        row = cur.fetchone()
    return row.get("lecture_uid") if row and row.get("lecture_uid") else None

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
    reuse_existing: bool = False,
    media_triggers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = _default_metadata(metadata)
    admin_value = admin_id or metadata.get("admin_id") 
    material_value = material_id or metadata.get("material_id")
    reuse_lecture_uid: Optional[str] = None
    if reuse_existing and not lecture_uid and material_value:
        reuse_lecture_uid = _maybe_reuse_existing_lecture_id(
            admin_id=admin_value,
            material_id=material_value,
        )
    lecture_id = lecture_uid or reuse_lecture_uid or await _generate_lecture_id()
    created_at = datetime.utcnow()

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
    if media_triggers:
        record["media_triggers"] = media_triggers

    record["lecture_url"] = lecture_url or _build_default_url(record)
    subject_value = subject or metadata.get("subject")
    std_value = std or metadata.get("std") or metadata.get("class")
    sem_value = sem or metadata.get("sem")
    board_value = board or metadata.get("board")

    record["created_at"] = record["created_at"].isoformat()
    record["updated_at"] = record["updated_at"].isoformat()

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
    record.setdefault("cover_photo_url", row.get("cover_photo_url"))
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

    with get_pg_cursor() as cur:
        cur.execute("""
            SELECT * FROM lecture_gen
        """)
        all_rows = cur.fetchall()

    deleted: List[Dict[str, Any]] = []
    std_filter = _slugify(std)
    subject_filter = _slugify(subject)
    division_filter = _slugify(division) if division else None

    for row in all_rows:
        record = row.get("lecture_data") or {}
        metadata = record.get("metadata") or {}
        
        # Get values from metadata first, then fall back to database columns
        std_value = metadata.get("std") or metadata.get("class") or row.get("std")
        subject_value = metadata.get("subject") or row.get("subject")
        division_value = metadata.get("division") or metadata.get("section")
        
        # Compare using slugified versions
        if _slugify(std_value) != std_filter:
            continue
        if _slugify(subject_value) != subject_filter:
            continue
        
        division_slug = _slugify(division_value) if division_value else None
        if division_filter and division_slug != division_filter:
            continue
        
        # If lecture_id is specified, only delete that specific lecture
        if lecture_id and row.get("lecture_uid") != lecture_id:
            continue

        lecture_entry = {
            "lecture_id": row.get("lecture_uid"),
            "title": record.get("title") or row.get("chapter_title"),
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": _slugify(std_value),
            "subject_slug": _slugify(subject_value),
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
        metadata = _ensure_metadata_dict(record.get("metadata"))

        if lang_filter and (record.get("language") or "").lower() != lang_filter:
            continue

        std_value = (
            _text_or(metadata.get("std"))
            or _text_or(metadata.get("class"))
            or _text_or(row.get("std"))
            or "general"
        )
        subject_value = (
            _text_or(metadata.get("subject"))
            or _text_or(row.get("subject"))
            or "lecture"
        )
        division_value = _text_or(metadata.get("division")) or _text_or(metadata.get("section"))

        if std_filter and _slugify(std_value) != std_filter:
            continue
        if subject_filter and _slugify(subject_value) != subject_filter:
            continue
        if division_filter and _slugify(division_value) != division_filter:
            continue

        lecture_uid = _text_or(row.get("lecture_uid")) or _text_or(record.get("lecture_id"))
        if not lecture_uid:
            continue

        summary = {
            "lecture_id": lecture_uid,
            "title": (
                _text_or(record.get("title"))
                or _text_or(row.get("lecture_title"))
                or "Untitled lecture"
            ),
            "language": _text_or(record.get("language")),
            "total_slides": _coerce_int(record.get("total_slides")),
            "estimated_duration": _coerce_int(record.get("estimated_duration")),
            "created_at": _coerce_datetime(record.get("created_at") or row.get("created_at")),
            "fallback_used": record.get("fallback_used", False),
            "lecture_url": _text_or(record.get("lecture_url")) or _text_or(row.get("lecture_link")),
            "cover_photo_url": _text_or(record.get("cover_photo_url")) or _text_or(row.get("cover_photo_url")),
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": _slugify(std_value),
            "subject_slug": _slugify(subject_value),
            "division_slug": _slugify(division_value) if division_value else None,
        }
        slides = record.get("slides") or []
        bullets: List[str] = []
        for slide in slides:
            if not isinstance(slide, dict):
                continue
            for bullet in slide.get("bullets") or []:
                text = (bullet or "").strip()
                if text:
                    bullets.append(text)

        summary["bullets"] = bullets
        summaries.append(summary)

    return summaries[offset : offset + limit]

async def search_lectures_by_title(
    *,
    query: str,
    language: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    division: Optional[str] = None,
    admin_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search lectures by title using Gujarati-aware normalization.

    The search is performed in Python to allow matching Gujarati titles against
    WhatsApp-style Roman queries using `_normalize_title_for_search`.
    """

    normalized_query = _normalize_title_for_search(query)
    if not normalized_query:
        return []

    # Fuzzy representation (vowels stripped) to allow loose transliteration matches
    fuzzy_query = _normalize_title_for_fuzzy_match(normalized_query)

    with get_pg_cursor() as cur:
        sql = "SELECT * FROM lecture_gen"
        params: Dict[str, Any] = {}

        if admin_id is not None:
            sql += " WHERE admin_id = %(admin_id)s"
            params["admin_id"] = admin_id

        sql += " ORDER BY created_at DESC"
        cur.execute(sql, params)
        rows = cur.fetchall()

    std_filter = _slugify(std) if std else None
    subject_filter = _slugify(subject) if subject else None
    division_filter = _slugify(division) if division else None
    lang_filter = (language or "").lower() if language else None

    results: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("lecture_data"):
            continue
        record = _clone_record(row.get("lecture_data"))
        metadata = record.get("metadata") or {}

        if lang_filter and (record.get("language") or "").lower() != lang_filter:
            continue

        std_value = (
            _text_or(metadata.get("std"))
            or _text_or(metadata.get("class"))
            or _text_or(row.get("std"))
            or "general"
        )
        subject_value = (
            _text_or(metadata.get("subject"))
            or _text_or(row.get("subject"))
            or "lecture"
        )
        division_value = _text_or(metadata.get("division")) or _text_or(metadata.get("section"))

        if std_filter and _slugify(std_value) != std_filter:
            continue
        if subject_filter and _slugify(subject_value) != subject_filter:
            continue
        if division_filter and _slugify(division_value) != division_filter:
            continue

        title_value = record.get("title") or row.get("lecture_title") or ""
        normalized_title = _normalize_title_for_search(title_value)
        if not normalized_title:
            continue

        # First, try a direct raw substring match to support exact Hindi/Gujarati
        # script queries matching the stored title.
        raw_title = str(title_value or "").strip().lower()
        raw_query = str(query or "").strip().lower()

        if raw_query and raw_query not in raw_title:
            # If raw match fails, fall back to strict normalized and then fuzzy match.
            # First try strict substring match on normalized forms (supports partial matches).
            if normalized_query not in normalized_title:
                # Fall back to fuzzy consonant-pattern matching for loose transliteration
                fuzzy_title = _normalize_title_for_fuzzy_match(normalized_title)
                if fuzzy_query:
                    any_token_match = False
                    for token in fuzzy_query.split(" "):
                        token = token.strip()
                        if not token:
                            continue
                        if token in fuzzy_title:
                            any_token_match = True
                            break
                    if not any_token_match:
                        continue
                else:
                    # No fuzzy query constructed and strict match failed
                    continue

        lecture_uid = _text_or(row.get("lecture_uid")) or _text_or(record.get("lecture_id"))
        if not lecture_uid:
            continue

        summary = {
            "lecture_id": lecture_uid,
            "title": (
                _text_or(record.get("title"))
                or _text_or(row.get("lecture_title"))
                or "Untitled lecture"
            ),
            "language": _text_or(record.get("language")),
            "total_slides": _coerce_int(record.get("total_slides")),
            "estimated_duration": _coerce_int(record.get("estimated_duration")),
            "created_at": _coerce_datetime(record.get("created_at") or row.get("created_at")),
            "fallback_used": record.get("fallback_used", False),
            "lecture_url": _text_or(record.get("lecture_url")) or _text_or(row.get("lecture_link")),
            "cover_photo_url": _text_or(record.get("cover_photo_url")) or _text_or(row.get("cover_photo_url")),
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": _slugify(std_value),
            "subject_slug": _slugify(subject_value),
            "division_slug": _slugify(division_value) if division_value else None,
        }

        slides = record.get("slides") or []
        bullets: List[str] = []
        for slide in slides:
            if not isinstance(slide, dict):
                continue
            for bullet in slide.get("bullets") or []:
                text = (bullet or "").strip()
                if text:
                    bullets.append(text)

        summary["bullets"] = bullets
        results.append(summary)

    return results[offset : offset + limit]


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

        # Derive duration in minutes using existing lecture metadata
        duration_minutes: Optional[int] = None
        if isinstance(record, dict):
            duration_candidates = [
                record.get("estimated_duration"),
                record.get("requested_duration"),
                (record.get("metadata") or {}).get("duration"),
            ]
            for candidate in duration_candidates:
                if candidate is not None:
                    try:
                        duration_minutes = int(candidate)
                        break
                    except (TypeError, ValueError):
                        continue

        played.append(
            {
                "lecture_id": row.get("lecture_uid"),
                "title": record.get("title") or row.get("lecture_title"),
                "language": record.get("language"),
                "play_count": play_count,
                "last_played_at": record.get("last_played_at"),
                # "lecture_url": record.get("lecture_url") or row.get("lecture_link"),
                # "lecture_url": record.get("lecture_url") or row.get("lecture_link"),"cover_photo_url": record.get("cover_photo_url") or row.get("cover_photo_url"),
                "lecture_url": record.get("lecture_url") or row.get("lecture_link"),
                "cover_photo_url": record.get("cover_photo_url") or row.get("cover_photo_url"),
                "duration": duration_minutes,
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

async def create_chatbot_entry(
    *,
    lecture_id: str,
    question: Optional[str],
    response_text: Optional[str],
    audio_url: Optional[str],
    language: Optional[str],
    extra_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a lecture chatbot interaction."""
    payload = {
        "lecture_id": lecture_id,
        "question": question,
        "response_text": response_text,
        "audio_url": audio_url,
        "language": language,
        "extra_data": json.dumps(extra_data or {}),
    }
    with get_pg_cursor() as cur:
        cur.execute(
            """
            INSERT INTO lecture_chatbot
                (lecture_id, question, response_text, audio_url, language, extra_data)
            VALUES
                (%(lecture_id)s, %(question)s, %(response_text)s, %(audio_url)s, %(language)s, %(extra_data)s)
            RETURNING *;
            """,
            payload,
        )
        row = cur.fetchone()
    return row or payload

async def list_chatbot_entries_for_lecture(lecture_id: str) -> List[Dict[str, Any]]:
    """Return all chatbot Q&A entries for a given lecture_id."""
    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                lecture_id,
                question,
                response_text,
                audio_url,
                language,
                extra_data,
                created_at,
                updated_at
            FROM lecture_chatbot
            WHERE lecture_id = %(lecture_id)s
            ORDER BY created_at ASC, id ASC
            """,
            {"lecture_id": lecture_id},
        )
        rows = cur.fetchall()

    entries: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        raw_extra = entry.get("extra_data")
        if isinstance(raw_extra, str):
            try:
                entry["extra_data"] = json.loads(raw_extra)
            except Exception:
                entry["extra_data"] = {}
        entries.append(entry)

    return entries









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

    async def search_lectures_by_title(
        self,
        *,
        query: str,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        std: Optional[str] = None,
        subject: Optional[str] = None,
        division: Optional[str] = None,
        admin_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await search_lectures_by_title(
            query=query,
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

    async def create_chatbot_entry(
        self,
        *,
        lecture_id: str,
        question: Optional[str],
        response_text: Optional[str],
        audio_url: Optional[str],
        language: Optional[str],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await create_chatbot_entry(
            lecture_id=lecture_id,
            question=question,
            response_text=response_text,
            audio_url=audio_url,
            language=language,
            extra_data=extra_data,
        )

    async def list_chatbot_entries_for_lecture(self, lecture_id: str) -> List[Dict[str, Any]]:
        return await list_chatbot_entries_for_lecture(lecture_id)