# app/routes/chapter_material_routes.py

import asyncio
import json
import logging
import os
from pydoc import resolve
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Query, Request, Body
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session, aliased
from sqlalchemy import desc, func, or_

from app.config import get_settings
from app.database import get_db
from app.models import ChapterMaterial, LectureGen
from app.schemas.admin_schema import WorkType
from app.schemas.chapter_material_schema import (
    TopicExtractRequest,
    LectureGenerationRequest,
    LectureConfigRequest,
    LectureChatRequest,
    LectureLookupRequest,
    ManualTopicCreate,
    AssistantSuggestRequest,
    AssistantAddTopicsRequest,
    MultipleChapterSelectionRequest,
    CreateMergedLectureRequest,
    TopicSelection,
    ResponseBase,
)
from app.repository import auth_repository, registration_repository,lecture_credit_repository
from app.repository.chapter_material_repository import (
    create_chapter_material,
    get_chapter_material,
    list_chapter_materials,
    list_recent_chapter_materials,
    delete_chapter_material_db,
    get_dashboard_stats,
    get_chapter_overview_data,
    _load_topics_path,
    load_material_topics,
    persist_material_topics,
    save_extracted_topics_files,
    read_topics_file_if_exists,
    append_manual_topic_to_file,
    add_assistant_topics_to_file,
    topic_to_text,
    read_pdf_context_for_material,
    LANGUAGE_OUTPUT_RULES,
    SUPPORTED_LANGUAGES,
    DURATION_OPTIONS,
    persist_assistant_suggestions,
    get_cached_suggestions_by_ids,
    get_topics_by_ids,
)
from app.plan_limits import PLAN_CREDIT_LIMITS, PLAN_DURATION_LIMITS, PLAN_SUGGESTION_LIMITS
from app.utils.file_handler import (
    save_uploaded_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
    get_file_url,
)
from app.utils.s3_file_handler import (
    upload_pdf_to_s3,
    upload_image_to_s3,
    upload_audio_to_s3,
    get_s3_service,
)
from app.services.lecture_service import LectureService
from app.services.topic_extract_queue import (
    topic_extraction_queue,
    TopicExtractionQueueFullError,
    TopicExtractionQueueTimeoutError,
)
from app.utils.dependencies import admin_required, get_current_user
from groq import Groq

import os
from pathlib import Path
from datetime import datetime
router = APIRouter(prefix="/chapter-materials", tags=["Chapter Materials"])


def get_lecture_service(db: Session = Depends(get_db)) -> LectureService:
    return LectureService(db=db)

logger = logging.getLogger(__name__)

def _extract_error_message(detail: Any) -> str:
    """Normalize HTTPException.detail into a single human-readable string."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        for key in ("message", "detail", "error"):
            value = detail.get(key)
            if value:
                return str(value)
        return json.dumps(detail)
    return str(detail)


async def chapter_material_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure chapter management errors always return ErrorResponse with only a details.message."""
    message = _extract_error_message(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": False,
            "details": {"message": message},
        },
    )

def _normalize_string(value: Optional[str]) -> str:
    return (value or "").strip().lower()

def _unique_sorted_strings(values: List[Optional[str]]) -> List[str]:
    seen: Dict[str, str] = {}
    for raw in values:
        candidate = (raw or "").strip()
        if not candidate:
            continue
        normalized = candidate.lower()
        seen.setdefault(normalized, candidate)
    return sorted(seen.values(), key=lambda item: item.lower())

# Local constants (kept same)
PDF_MAX_SIZE = 50 * 1024 * 1024  # 15MB
DEFAULT_MIN_DURATION = 5
DEFAULT_MAX_DURATION = 180
MAX_ASSISTANT_SUGGESTIONS = 10
DEFAULT_LANGUAGE_CODE = "eng"
SUPPORTED_MODELS = ["inai", "vinai", "aira"]

MERGED_LECTURES_DIR = Path("./storage/merged_lectures")


# -------------------------
# Chapter filters
# -------------------------


@router.get("/chapters/filters", response_model=ResponseBase)
async def get_chapter_filters(
    std: Optional[str] = Query(None, description="Class/standard to fetch subjects for"),
    subject: Optional[str] = Query(None, description="Subject to fetch chapters for"),
    chapter: Optional[str] = Query(None, description="Chapter title to fetch lecture data for"),
    chapter_alias: Optional[str] = Query(
        None,
        alias="chapters",
        description="Legacy chapter filter alias (supports existing frontend requests)",
    ),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    overview_entries = get_chapter_overview_data(admin_id) or []
    
    generated_entries = [
        entry
        for entry in overview_entries
        if entry.get("std") and entry.get("subject") and entry.get("chapter")
    ]

    standards = _unique_sorted_strings([entry.get("std") for entry in generated_entries])
    subjects: List[str] = []
    chapter_options: List[str] = []
    selected_chapter = chapter or chapter_alias

    # lectures: List[Dict[str, Any]] = []

    # if std:
    #     subjects = list_subjects_for_std(admin_id, std=std)
    #     if subject:
    #         chapter_options = list_chapters_for_selection(
    #             admin_id,
    #             std=std,
    #             subject=subject,
    #         )
    #         if selected_chapter:
    #             chapter_clean = selected_chapter.strip().lower()
    #             if chapter_clean not in {c.strip().lower() for c in chapter_options}:
    #                 chapter_options = [selected_chapter]
    #             else:
    #                 chapter_options = [c for c in chapter_options if c.strip().lower() == chapter_clean]
    #             material_ids = find_material_ids_for_chapter(
    #                 admin_id=admin_id,
    #                 std=std,
    #                 subject=subject,
    #                 chapter_identifier=selected_chapter,
    #             )
    std_filter = _normalize_string(std) if std else None
    subject_filter = _normalize_string(subject) if subject else None
    chapter_filter = _normalize_string(selected_chapter) if selected_chapter else None

    std_filtered_entries = (
        [entry for entry in generated_entries if _normalize_string(entry.get("std")) == std_filter]
        if std_filter
        else []
    )

    if std_filter:
        subjects = _unique_sorted_strings([entry.get("subject") for entry in std_filtered_entries])

    material_ids: List[int] = []
    subject_filtered_entries: List[Dict[str, Any]] = []
    if std_filter and subject_filter:
        subject_filtered_entries = [
            entry for entry in std_filtered_entries if _normalize_string(entry.get("subject")) == subject_filter
        ]
        chapter_options = _unique_sorted_strings([entry.get("chapter") for entry in subject_filtered_entries])

        if chapter_filter:
            matching_entries = [
                entry for entry in subject_filtered_entries if _normalize_string(entry.get("chapter")) == chapter_filter
            ]
            material_ids = [
                entry.get("material_id")
                for entry in matching_entries
                if isinstance(entry.get("material_id"), int)
            ]

    lectures: List[Dict[str, Any]] = []
    if std and subject:
        lectures = _fetch_filtered_lectures(
            db,
            admin_id=admin_id,
            std=std,
            subject=subject,
            chapter=None if material_ids else selected_chapter,
            material_ids=material_ids or None,
        )

    response_data = {
        "standards": standards,
        "subjects": subjects,
        "chapter": chapter_options,
        "selected_std": std,
        "selected_subject": subject,
        "selected_chapter": selected_chapter,
        "lectures": lectures,
        "lectures_count": len(lectures),
    }
    
    return ResponseBase(status=True, message="Chapter filters fetched", data=response_data)

@router.get("/chapters", response_model=ResponseBase)
async def list_chapters_endpoint(
    std: str = Query(..., description="Class/standard"),
    subject: Optional[str] = Query(None, description="Subject"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    chapters: List[str] = []
    if subject:
        chapters = list_chapters_for_selection(
            admin_id,
            std=std,
            subject=subject,
        )

    subject_payload: Any = subject if subject is not None else []

    return ResponseBase(
        status=True,
        message="Chapters fetched successfully",
        data={
            "std": std,
            "subject": subject_payload,
            "chapters": chapters,
        },
    )


# -------------------------
# Helper to resolve admin_id from current_user
# -------------------------
def _resolve_member_admin_id(current_user: dict) -> Optional[int]:
    user_obj = current_user.get("user_obj")
    if user_obj is None:
        return current_user.get("admin_id")
    if isinstance(user_obj, dict):
        return user_obj.get("admin_id") or current_user.get("admin_id")
    return getattr(user_obj, "admin_id", None) or current_user.get("admin_id")


def _resolve_admin_id(current_user: dict) -> int:
    if current_user["role"] == "admin":
        return current_user["id"]
    if current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access restricted to chapter management members",
            )
        resolved = _resolve_member_admin_id(current_user)
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member admin not found")
        return resolved
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


def _normalize_plan_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    value = value.strip().lower()
    if value in PLAN_SUGGESTION_LIMITS:
        return value
    return None


def _resolve_plan_label_for_admin(
    admin_id: int,
    current_user: dict,
    requested_plan_label: Optional[str],
) -> Optional[str]:
    """Derive a normalized plan label for the admin, honoring any explicit override."""

    normalized_override = _normalize_plan_label(requested_plan_label)
    if normalized_override:
        return normalized_override

    def _extract_plan(source: Optional[Any]) -> Optional[str]:
        if not source:
            return None
        candidate_keys = ("plan_label", "plan", "package", "package_plan", "packagePlan")
        for key in candidate_keys:
            if isinstance(source, dict):
                value = source.get(key)
            else:
                value = getattr(source, key, None)
            normalized_value = _normalize_plan_label(value)
            if normalized_value:
                return normalized_value
        return None

    # Try to resolve from the current request context first.
    contextual_plan = _extract_plan(current_user)
    if contextual_plan:
        return contextual_plan

    user_obj = current_user.get("user_obj")
    user_obj_plan = _extract_plan(user_obj)
    if user_obj_plan:
        return user_obj_plan

    # Fall back to persisted admin records.
    admin_record = auth_repository.get_admin_by_id(admin_id) or registration_repository.get_admin_by_id(admin_id)
    return _extract_plan(admin_record)

def _get_admin_credit_summary(admin_id: int, current_user: dict) -> Dict[str, Any]:
    """Return lecture credit usage summary for the given admin."""

    plan_label = _resolve_plan_label_for_admin(admin_id, current_user, requested_plan_label=None)
    plan_credit_total = PLAN_CREDIT_LIMITS.get(plan_label) if plan_label else None
    credit_usage = lecture_credit_repository.get_usage(admin_id)

    credit_remaining: Optional[int]
    if plan_credit_total is not None:
        credit_remaining = max(plan_credit_total - credit_usage["credits_used"], 0)
        post_limit_generated = (
            credit_usage["credits_used"] - plan_credit_total
            if credit_usage["credits_used"] > plan_credit_total
            else 0
        )
    else:
        credit_remaining = None
        post_limit_generated = 0

    return {
        "plan_label": plan_label,
        "total": plan_credit_total,
        "used": credit_usage["credits_used"],
        "remaining": credit_remaining,
        "post_limit_generated": post_limit_generated,
        "overflow_attempts": credit_usage["overflow_attempts"],
    }

def _get_allowed_duration_options(admin_id: int, current_user: dict) -> Tuple[Optional[str], List[int]]:
    plan_label = _resolve_plan_label_for_admin(admin_id, current_user, requested_plan_label=None)
    max_duration = PLAN_DURATION_LIMITS.get(plan_label) if plan_label else None

    if max_duration is None:
        allowed = list(DURATION_OPTIONS)
    else:
        allowed = [duration for duration in DURATION_OPTIONS if int(duration) <= int(max_duration)]

    if not allowed:
        allowed = [DURATION_OPTIONS[0]]

    return plan_label, allowed


def _enforce_plan_duration(duration: Optional[int], allowed: List[int], plan_label: Optional[str]) -> Optional[int]:
    if duration is None:
        return None
    if duration not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Duration {duration} is not allowed under the {plan_label or 'current'} plan.",
        )
    return duration

def _ensure_lecture_config_access(current_user: dict) -> None:
    if current_user["role"] == "admin":
        return
    if current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can access lecture configuration",
            )
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")


def _match_supported_language_option(value: Optional[str]) -> Optional[Dict[str, str]]:
    """Return the supported language option that matches the provided value or label."""
    if not value:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None

    for option in SUPPORTED_LANGUAGES:
        candidates = {option["value"].strip().lower()}
        label = option.get("label")
        if label:
            label_clean = label.strip().lower()
            if label_clean:
                candidates.add(label_clean)
            # Split labels like "हिंदी / Hindi" so that either portion matches.
            for part in re.split(r"[\/\-\(\)\|]", label):
                part_clean = part.strip().lower()
                if part_clean:
                    candidates.add(part_clean)

        if normalized in candidates:
            return option

    return None


def _build_lecture_config_response(
    *,
    requested_language: Optional[str],
    requested_duration: Optional[int],
    duration_options: Optional[List[int]] = None,
    requested_model: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    default_language = settings.dict().get("default_language") or getattr(settings, "default_language", None)

    language_option = _match_supported_language_option(requested_language)
    if not language_option and default_language:
        language_option = _match_supported_language_option(default_language)

    if language_option:
        language_value = language_option["value"]
        language_label = language_option.get("label") or language_value
    else:
        fallback_option = SUPPORTED_LANGUAGES[0]
        language_value = fallback_option["value"]
        language_label = fallback_option.get("label") or fallback_option["value"]

    resolved_duration_options = list(duration_options) if duration_options else list(DURATION_OPTIONS)

    configured_default_duration = (
        getattr(settings, "default_lecture_duration", None)
        or settings.dict().get("default_lecture_duration")
        or resolved_duration_options[0]
    )
    selected_duration = requested_duration if requested_duration is not None else configured_default_duration

    # Model selection: normalize and constrain to supported models
    normalized_model = _normalize_requested_model(requested_model)
    selected_model = normalized_model

    if selected_duration not in resolved_duration_options:
        selected_duration = resolved_duration_options[-1]

    configured_default_model = (
        getattr(settings, "default_lecture_model", None)
        or settings.dict().get("default_lecture_model")
    )
    default_model = _normalize_requested_model(configured_default_model) or SUPPORTED_MODELS[0]

    if selected_model is None:
        selected_model = default_model

    return {
        "selected_duration": selected_duration,
        "duration_options": resolved_duration_options,
        "selected_language": language_value,
        "selected_language_label": language_label,
        "video_duration_minutes": selected_duration,
        "selected_model": selected_model,
        "default_model": default_model,
    }


def _normalize_requested_language(value: Optional[str]) -> Optional[str]:
    language_option = _match_supported_language_option(value)
    return language_option["value"] if language_option else None


def _normalize_requested_duration(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized < DEFAULT_MIN_DURATION or normalized > DEFAULT_MAX_DURATION:
        return None
    return normalized


def _normalize_requested_model(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return normalized if normalized in SUPPORTED_MODELS else None


def _resolve_model_selection(value: Optional[str], settings: Any) -> str:
    normalized = _normalize_requested_model(value)
    if normalized:
        return normalized

    configured_default_model = (
        getattr(settings, "default_lecture_model", None)
        or settings.dict().get("default_lecture_model")
    )
    return _normalize_requested_model(configured_default_model) or SUPPORTED_MODELS[0]

def _save_merged_lecture_payload(lecture_id: str, payload: Dict[str, Any]) -> None:
    MERGED_LECTURES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = MERGED_LECTURES_DIR / f"{lecture_id}.json"
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_merged_lecture_payload(lecture_id: str) -> Optional[Dict[str, Any]]:
    file_path = MERGED_LECTURES_DIR / f"{lecture_id}.json"
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load merged lecture %s: %s", lecture_id, exc)
        return None


def _prepare_generation_from_material(
    *,
    request: LectureGenerationRequest,
    current_user: dict,
    db: Session,
    settings: Any,
) -> Dict[str, Any]:
    material = get_chapter_material(request.material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    material_admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
    
    if current_user["role"] == "admin":
        if material_admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can generate lectures",
            )
        member_admin_id = _resolve_member_admin_id(current_user)
        if material_admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")
    material_id = material.get("id") if isinstance(material, dict) else material.id
    topics_path = _load_topics_path(material_admin_id, material_id)
    if not os.path.exists(topics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topics not found. Please extract topics before generating a lecture.",
        )

    with open(topics_path, "r", encoding="utf-8") as fh:
        topics_payload = json.load(fh)
    topics = topics_payload.get("topics", [])
    if not topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No topics available for this material",
        )

    index_by_id: Dict[str, Dict[str, Any]] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        suggestion_id = topic.get("suggestion_topic_id")
        if topic_id is not None:
            index_by_id[str(topic_id)] = topic
        if suggestion_id is not None:
            index_by_id[str(suggestion_id)] = topic

    selected_topics: List[Dict[str, Any]] = []
    missing_topic_ids: List[str] = []
    for tid in request.selected_topic_ids or []:
        entry = index_by_id.get(str(tid))
        if entry:
            selected_topics.append(entry)
        else:
            missing_topic_ids.append(str(tid))

    if missing_topic_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topics not found for IDs: {', '.join(missing_topic_ids)}",
        )

    if not selected_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected topics are invalid",
        )

    aggregate_text_parts = [topic_to_text(topic) for topic in selected_topics]
    aggregate_text = "\n\n".join(part for part in aggregate_text_parts if part)
    if not aggregate_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to extract content from selected topics",
        )

    topics_language_label = topics_payload.get("language_label") if topics_payload else None
    language = topics_language_label or "English"
    override_language = _normalize_requested_language(request.language)
    if override_language:
        language = override_language

    duration = settings.default_lecture_duration
    override_duration = _normalize_requested_duration(request.duration)
    if override_duration is not None:
        duration = override_duration

    plan_label, allowed_duration = _get_allowed_duration_options(material_admin_id,current_user)
    _enforce_plan_duration(duration, allowed_duration, plan_label)
    extracted_chapter_title = (topics_payload or {}).get("chapter_title") if topics_payload else None
    resolved_chapter_title = extracted_chapter_title or ""
    material_subject = material.get("subject") if isinstance(material, dict) else material.subject
    title = resolved_chapter_title or (f"{material_subject} Lecture" if material_subject else "Generated Lecture")
    # Ensure title is never None or empty
    if not title or title.isspace():
        title = "Generated Lecture"

    material_id = material.get("id") if isinstance(material, dict) else material.id
    material_std = material.get("std") if isinstance(material, dict) else material.std
    material_admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
    
    selected_model = _resolve_model_selection(request.model, settings)

    metadata = {
        "material_id": material_id,
        "material_subject": material_subject,
        "selected_topic_ids": request.selected_topic_ids,
        "topics_source_file": topics_path,
        "language_label": topics_payload.get("language_label") if topics_payload else None,
        "language_code": topics_payload.get("language_code") if topics_payload else None,
        "std": material_std,
        "subject": material_subject,
        "chapter_title": resolved_chapter_title,
        "requested_language": override_language,
        "requested_duration": override_duration,
        "admin_id": material_admin_id,
        "model": selected_model,
    }

    std_value = material_std or "general"
    subject_value = material_subject or "subject"
    std_slug = std_value.replace(" ", "_").lower()
    subject_slug = subject_value.replace(" ", "_").lower()

    material_board = material.get("board") if isinstance(material, dict) else material.board
    material_sem = material.get("sem") if isinstance(material, dict) else material.sem
    
    log_context = {
        "material_id": material_id,
        "std": material_std or "N/A",
        "subject": material_subject or "N/A",
        "board": material_board or "N/A",
        "sem": material_sem or "N/A",
        "selected_topics_count": len(selected_topics),
        "materials_count": 1,
        "merged_lecture_id": None,
    }

    return {
        "aggregate_text": aggregate_text,
        "language": language,
        "duration": duration,
        "title": title,
        "metadata": metadata,
        "std_slug": std_slug,
        "subject_slug": subject_slug,
        "log_context": log_context,
        "material_snapshot": {
            "id": material_id,
            "admin_id": material_admin_id,
            "std": material_std,
            "subject": material_subject,
            "board": material_board,
            "sem": material_sem,
            "chapter_number": material.get("chapter_number") if isinstance(material, dict) else material.chapter_number,
            "file_name": material.get("file_name") if isinstance(material, dict) else material.file_name,
        },
        "requested_language": override_language,
        "requested_duration": override_duration,
        "model": selected_model,
    }


def _prepare_generation_from_merged(
    *,
    request: LectureGenerationRequest,
    current_user: dict,
    db: Session,
    settings: Any,
) -> Dict[str, Any]:
    lecture_id = request.merged_lecture_id or ""
    merged_payload = _load_merged_lecture_payload(lecture_id)
    if not merged_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Merged lecture not found")

    creator_admin_id = merged_payload.get("admin_id")

    response_payload = (merged_payload.get("response") or {}).get("data") or {}
    merged_topics = response_payload.get("merged_topics") or []

    aggregated_topics: List[Dict[str, Any]] = []
    candidate_material_ids: List[Any] = []

    def _material_attr(material_obj: Any, attribute: str, default: Any = None) -> Any:
        if material_obj is None:
            return default
        if isinstance(material_obj, dict):
            return material_obj.get(attribute, default)
        return getattr(material_obj, attribute, default)
    for block in merged_topics:
        material_id_value = block.get("material_id")
        if material_id_value is not None:
            candidate_material_ids.append(material_id_value)
        for topic in block.get("topics") or []:
            if isinstance(topic, dict):
                aggregated_topics.append(topic)

    primary_material = None
    inferred_admin_id = None
    for candidate in candidate_material_ids:
        try:
            candidate_id = int(candidate)
        except (TypeError, ValueError):
            continue
        material = get_chapter_material(candidate_id)
        if material and (material.get("admin_id") if isinstance(material, dict) else material.admin_id) == creator_admin_id:
            primary_material = material
            break
        if material and not inferred_admin_id:
            inferred_admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
            if primary_material is None:
                primary_material = material

    if creator_admin_id is None:
        creator_admin_id = inferred_admin_id

    if creator_admin_id is None:
        try:
            creator_admin_id = _resolve_admin_id(current_user)
        except HTTPException:
            creator_admin_id = _resolve_member_admin_id(current_user)

    if creator_admin_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Merged lecture metadata missing admin reference",
        )

    if current_user["role"] == "admin":
        if current_user["id"] != creator_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can generate lectures",
            )
        member_admin_id = _resolve_member_admin_id(current_user)
        if member_admin_id != creator_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    if not aggregated_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Merged lecture does not contain topics to generate from",
        )

    aggregate_text_parts = [topic_to_text(topic) for topic in aggregated_topics]
    aggregate_text = "\n\n".join(part for part in aggregate_text_parts if part).strip()
    if not aggregate_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to extract content from merged lecture topics",
        )

    language = response_payload.get("language") or "English"
    override_language = _normalize_requested_language(request.language)
    if override_language:
        language = override_language

    configured_duration = response_payload.get("duration") or settings.default_lecture_duration
    duration = configured_duration
    override_duration = _normalize_requested_duration(request.duration)
    if override_duration is not None:
        duration = override_duration

    plan_label, allowed_durations = _get_allowed_duration_options(int(creator_admin_id),current_user)
    _enforce_plan_duration(duration, allowed_durations, plan_label)
    title = response_payload.get("chapter_title") 
    # Ensure title is never None or empty
    if not title or title.isspace():
        title = f"Merged Lecture {lecture_id}"

    resolved_material_id = _material_attr(primary_material, "id")
    if resolved_material_id is None:
        for candidate in candidate_material_ids:
            try:
                resolved_material_id = int(candidate)
                break
            except (TypeError, ValueError):
                continue

    resolved_std = _material_attr(primary_material, "std")
    resolved_subject = _material_attr(primary_material, "subject")
    resolved_board = _material_attr(primary_material, "board")
    resolved_sem = _material_attr(primary_material, "sem")

    std_slug_source = resolved_std or "merged"
    subject_slug_source = resolved_subject or "combined"

    std_slug = std_slug_source.replace(" ", "_").lower()
    subject_slug = subject_slug_source.replace(" ", "_").lower()
    selected_model = _resolve_model_selection(request.model or response_payload.get("model"), settings)

    metadata: Dict[str, Any] = {
        "source": "merged_topics",
        "merged_lecture_id": lecture_id,
        "materials_count": response_payload.get("materials_count"),
        "topics_count": response_payload.get("topics_count"),
        "selected_material_ids": response_payload.get("selected_materials"),
        "admin_id": creator_admin_id,
        "material_id": resolved_material_id,
        "std": resolved_std,
        "subject": resolved_subject,
        "board": resolved_board,
        "sem": resolved_sem,
        "model": selected_model,
    }

    if primary_material:
        metadata.update(
            {
                "primary_material_id": resolved_material_id,
                "primary_material_subject": resolved_subject,
                "primary_material_std": resolved_std,
            }
        )

    log_context = {
        "material_id": resolved_material_id,
        "std": resolved_std or "Merged",
        "subject": resolved_subject or "Merged Topics",
        "board": resolved_board or "Mixed",
        "sem": resolved_sem or "Mixed",
        "selected_topics_count": response_payload.get("topics_count") or len(aggregated_topics),
        "materials_count": response_payload.get("materials_count") or len(candidate_material_ids),
        "merged_lecture_id": lecture_id,
    }

    material_snapshot = {
        "id": resolved_material_id,
        "admin_id": _material_attr(primary_material, "admin_id", creator_admin_id),
        "std": resolved_std or log_context["std"],
        "subject": resolved_subject or log_context["subject"],
        "board": resolved_board or log_context["board"],
        "sem": resolved_sem or log_context["sem"],
        "chapter_number": _material_attr(primary_material, "chapter_number"),
        "file_name": _material_attr(primary_material, "file_name"),
    }

    return {
        "aggregate_text": aggregate_text,
        "language": language,
        "duration": duration,
        "title": title,
        "metadata": metadata,
        "std_slug": std_slug,
        "subject_slug": subject_slug,
        "log_context": log_context,
        "material_snapshot": material_snapshot,
        "model": selected_model,
    }


def _fetch_filtered_lectures(
    db: Session,
    *,
    admin_id: int,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    material_ids: Optional[str] = None,
) -> List[Dict[str, Any]]:
    settings = get_settings()
    configured_default_duration = (
        getattr(settings, "default_lecture_duration", None)
        or settings.dict().get("default_lecture_duration")
        or DURATION_OPTIONS[0]
    )
    default_thumbnail_url = (
        getattr(settings, "default_lecture_thumbnail", None)
        or settings.dict().get("default_lecture_thumbnail")
        or "/static/images/lecture-placeholder.png"
    )

    # latest_lecture_subquery = (
    #     db.query(
    #         LectureGen.material_id.label("material_id"),
    #         func.max(LectureGen.created_at).label("latest_created_at"),
    #     )
    #     .filter(LectureGen.admin_id == admin_id)
    #     .group_by(LectureGen.material_id)
    #     .subquery()
    # )

    # query = (
    #     db.query(
    #         ChapterMaterial,
    #         lecture_alias.lecture_uid,
    #         lecture_alias.chapter_title,
    #         lecture_alias.lecture_link,
    #         lecture_alias.created_at.label("lecture_created_at"),
    #         lecture_alias.lecture_data,
    #     )
    #     .filter(ChapterMaterial.admin_id == admin_id)
    #     .outerjoin(
    #         latest_lecture_subquery,
    #         latest_lecture_subquery.c.material_id == ChapterMaterial.id,
    #     )
    #     .outerjoin(
    #         lecture_alias,
    #         (lecture_alias.material_id == ChapterMaterial.id)
    #         & (
    #             lecture_alias.created_at == latest_lecture_subquery.c.latest_created_at
    #         ),
    #     )
    # )

    # if std:
    #     query = query.filter(func.lower(func.trim(ChapterMaterial.std)) == std.strip().lower())
    # if subject:
    #     query = query.filter(func.lower(func.trim(ChapterMaterial.subject)) == subject.strip().lower())
    # if chapter:
    #     chapter_clean = chapter.strip().lower()
    #     title_expr = func.lower(
    #         func.trim(
    #             func.coalesce(ChapterMaterial.chapter_title, "")
    #         )
    #     )
    #     number_expr = func.lower(func.trim(ChapterMaterial.chapter_number))
    #     query = query.filter(
    #         or_(
    #             title_expr == chapter_clean,
    #             number_expr == chapter_clean,
    #         )
    #     )

    # records = query.order_by(desc(ChapterMaterial.created_at)).all()

    storage_base = Path("./storage/chapter_lectures")
    items: List[Dict[str, Any]] = []

    def format_file_size(bytes_size: Optional[int]) -> str:
        if not bytes_size:
            return "41.1 KB"
        mb = bytes_size / (1024 * 1024)
        if mb < 0.5:
            kb = bytes_size / 1024
            return f"{kb:.1f} KB"
        return f"{mb:.2f} MB"

    def _filter_topic_titles(candidates: List[Optional[str]]) -> List[str]:
        filtered: List[str] = []
        for raw in candidates:
            if not raw:
                continue
            normalized = str(raw).strip()
            if not normalized or normalized.lower().startswith("i'm sorry"):
                continue
            filtered.append(normalized)
            if len(filtered) == 5:
                break
        return filtered
    base_query = db.query(LectureGen).filter(LectureGen.admin_id == admin_id)
    
    if material_ids:
        base_query = base_query.filter(LectureGen.material_id.in_(material_ids))
    if std:
        std_clean = std.strip().lower()
        base_query = base_query.filter(func.lower(func.trim(LectureGen.std)) == std_clean)
    if subject:
        subject_clean = subject.strip().lower()
        base_query = base_query.filter(func.lower(func.trim(LectureGen.subject)) == subject_clean)
    if chapter:
        chapter_clean = chapter.strip().lower()
        base_query = base_query.filter(func.lower(func.trim(LectureGen.chapter_title)) == chapter_clean)

    records = base_query.order_by(desc(LectureGen.created_at)).all()

    for record in records:
        lecture_data_raw = record.lecture_data
        lecture_data: Optional[Dict[str, Any]] = None
        if isinstance(lecture_data_raw, dict):
            lecture_data = lecture_data_raw
        elif isinstance(lecture_data_raw, str):
            try:
                lecture_data = json.loads(lecture_data_raw)
            except (json.JSONDecodeError, TypeError):
                lecture_data = None


        if not lecture_data:
            continue

        slides = lecture_data.get("slides") or []
        if not isinstance(slides, list) or not slides:
            continue

        topic_titles = _filter_topic_titles(
            [slide.get("title") if isinstance(slide, dict) else None for slide in slides]
        )
        if not topic_titles:
            continue

        chapter_title = (
            (record.chapter_title or "").strip()
            or lecture_data.get("chapter_title")
            or lecture_data.get("metadata", {}).get("chapter_title")
        )

        lecture_size = 0
        thumbnail_url: Optional[str] = record.cover_photo_url
        lecture_uid = record.lecture_uid
        if record.lecture_link and lecture_uid:
            try:
                lecture_json_path = storage_base / lecture_uid / "lecture.json"
                if lecture_json_path.exists():
                    lecture_size = lecture_json_path.stat().st_size
                    thumbnail_candidates = [
                        storage_base / lecture_uid / name
                        for name in ("thumbnail.jpg", "thumbnail.png", "cover.jpg", "cover.png")
                    ]
                    for candidate in thumbnail_candidates:
                        if candidate.exists():
                            thumbnail_url = get_file_url(str(candidate))
                            break
            except Exception:
                pass

        lecture_duration_minutes: Optional[int] = None
        duration_candidates = [
            lecture_data.get("estimated_duration"),
            lecture_data.get("requested_duration"),
            lecture_data.get("metadata", {}).get("duration"),
        ]
        for candidate in duration_candidates:
            if candidate is not None:
                try:
                    lecture_duration_minutes = int(candidate)
                    break
                except (TypeError, ValueError):
                    continue

        effective_duration = lecture_duration_minutes or configured_default_duration
        resolved_lecture_id = str(lecture_uid)

        items.append(
            {
                "id": resolved_lecture_id,
                "lecture_uid": resolved_lecture_id,
                "admin_id": record.admin_id,
                "material_id": record.material_id,
                "lecture_link": record.lecture_link,
                "std": record.std,
                "subject": record.subject,
                "chapter": chapter_title,
                "topics": topic_titles,
                "size": format_file_size(lecture_size),
                "video_duration_minutes": effective_duration,
                "thumbnail_url": thumbnail_url or default_thumbnail_url,
            }
        )

    return items


# -------------------------
# Generated lectures listing
# -------------------------
@router.get("/chapter_lectures", response_model=ResponseBase)
async def list_generated_lectures(
    std: Optional[str] = Query(default=None, description="Filter by class/standard"),
    subject: Optional[str] = Query(default=None, description="Filter by subject"),
    chapter: Optional[str] = Query(default=None, description="Filter by chapter title (or number)"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id_for_lecture_access(current_user)
    items = _fetch_filtered_lectures(
        db,
        admin_id=admin_id,
        std=std,
        subject=subject,
        chapter=chapter,
    )

    return ResponseBase(
        status=True,
        message="Lectures fetched successfully",
        data={"items": items, "total": len(items)},
    )


@router.post("/public_lecture/start_new_lecture", response_model=ResponseBase)
async def lookup_chapter_lecture(
    request: LectureLookupRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)

    std_value = request.std.strip().lower()
    subject_value = request.subject.strip().lower()
    chapter_value = request.chapter_title.strip().lower()

    material = (
        db.query(ChapterMaterial)
        .filter(ChapterMaterial.admin_id == admin_id)
        .filter(func.lower(ChapterMaterial.std) == std_value)
        .filter(func.lower(ChapterMaterial.subject) == subject_value)
        .filter(
            or_(
                func.lower(ChapterMaterial.chapter_number) == chapter_value,
                func.lower(func.coalesce(ChapterMaterial.chapter_title, "")) == chapter_value,
            )
        )
        .order_by(desc(ChapterMaterial.updated_at))
        .first()
    )

    if not material:
        return ResponseBase(
            status=True,
            message="No lecture found for the provided information",
            data={"has_lecture": False},
        )

    lecture_record = (
        db.query(LectureGen)
        .filter(LectureGen.admin_id == admin_id, LectureGen.material_id == material.id)
        .order_by(desc(LectureGen.created_at))
        .first()
    )

    material_summary = {
        "id": material.id,
        "std": material.std,
        "subject": material.subject,
        "chapter_number": material.chapter_number,
        "chapter_title": material.chapter_title,
        "sem": material.sem,
        "board": material.board,
    }

    if not lecture_record:
        return ResponseBase(
            status=True,
            message="No lecture exists for this chapter",
            data={"has_lecture": False, "material": material_summary},
        )

    topics_preview: List[Dict[str, Any]] = []
    topics_count = 0
    chapters_count = 1
    try:
        _, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
        if topics_list:
            topics_count = len(topics_list)
            topics_preview = [
                {
                    "title": topic.get("title"),
                    "summary": topic.get("summary"),
                    "subtopics": topic.get("subtopics", []),
                }
                for topic in topics_list[:5]
                if isinstance(topic, dict)
            ]
    except Exception as exc:
        logger.warning("Failed to load topics for material %s: %s", material.id, exc)

    lecture_info = {
        "has_lecture": True,
        "lecture_uid": lecture_record.lecture_uid,
        "chapter_title": lecture_record.chapter_title,
        "lecture_link": lecture_record.lecture_link,
        "std": lecture_record.std or material.std,
        "subject": lecture_record.subject or material.subject,
        "sem": lecture_record.sem or material.sem,
        "board": lecture_record.board or material.board,
        "created_at": lecture_record.created_at.isoformat() if lecture_record.created_at else None,
        "chapters_count": chapters_count,
        "topics_count": topics_count,
        "topics_preview": topics_preview,
        "material": material_summary,
    }

    return ResponseBase(
        status=True,
        message="Lecture found",
        data=lecture_info,
    )


def _resolve_admin_id_for_lecture_access(current_user: dict) -> int:
    if current_user["role"] == "admin":
        return current_user["id"]
    if current_user["role"] == "member":
        allowed = {WorkType.CHAPTER.value, WorkType.LECTURE.value}
        if current_user.get("work_type") not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access restricted to chapter or lecture members",
            )
        resolved = _resolve_member_admin_id(current_user)
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member admin not found")
        return resolved
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


# -------------------------
# Dashboard endpoint
# -------------------------
@router.get("/dashboard")
async def get_chapter_dashboard(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        stats = get_dashboard_stats(admin_id)
        chapter_overview = get_chapter_overview_data(admin_id)
        lecture_credits = _get_admin_credit_summary(admin_id, current_user)
        return {
            "status": True,
            "message": "Dashboard data retrieved successfully",
            "data": {
                "chapter_metrics": stats,
                "chapter_overview": chapter_overview,
                                 "lecture_credits": lecture_credits,
            },
        }
    except Exception as e:
        logger.exception("Error fetching dashboard data")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -------------------------
# Upload / CRUD endpoints
# -------------------------
@router.post("/upload")
async def upload_chapter_material(
    std: str = Form(...),
    subject: str = Form(...),
    sem: str = Form(default=""),
    board: str = Form(default=""),
    chapter_number: str = Form(...),
    pdf_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if pdf_file.content_type not in ALLOWED_PDF_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    admin_id = _resolve_admin_id(current_user)

    # Upload PDF to S3
    try:
        settings = get_settings()
        s3_service = get_s3_service(settings)
    except (ValueError, Exception) as e:
        logger.error(f"S3 service initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 configuration error: {str(e)}. Please ensure AWS credentials are configured in .env file."
        )
    
    try:
        file_info = await upload_pdf_to_s3(
            pdf_file,
            s3_service,
            subfolder=f"admin_{admin_id}",
        )
    except HTTPException as e:
        logger.error(f"PDF upload to S3 failed: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload PDF to S3: {str(e)}"
        )

    chapter_material = create_chapter_material(
        admin_id=admin_id,
        std=std,
        subject=subject,
        sem=sem,
        board=board,
        chapter_number=chapter_number,
        file_info=file_info,
    )

    return {
        "status": True,
        "message": "Chapter material uploaded successfully",
        "data": {"material": chapter_material if isinstance(chapter_material, dict) else (chapter_material.to_dict() if hasattr(chapter_material, "to_dict") else chapter_material.__dict__)},
    }


@router.post("/chapter-suggestion")
async def list_chapter_materials_post(
    request_data: dict = Body(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        
        # Extract parameters from body
        std = request_data.get("std")
        subject = request_data.get("subject")
        # Validate parameters
        if std is not None and not str(std).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Standard cannot be empty")
        if subject is not None and not str(subject).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Subject cannot be empty")
        
        # Get materials with filtering
        materials = list_chapter_materials(admin_id, str(std).strip() if std else None, str(subject).strip() if subject else None)
        
        # Serialize materials with clean structure
        serialized = []
        for material in materials:
            try:
                # Clean data structure - only include necessary fields
                clean_material = {
                    "id": material["id"],
                    "std": material["std"],
                    "subject": material["subject"],
                    "sem": material["sem"],
                    "board": material["board"],
                    "chapter_number": material["chapter_number"],
                    "file_name": material["file_name"],
                    "file_size": material["file_size"],
                    "file_path": material["file_path"],
                    "created_at": material["created_at"].isoformat() if hasattr(material["created_at"], 'isoformat') else str(material["created_at"]),
                    "updated_at": material["updated_at"].isoformat() if hasattr(material["updated_at"], 'isoformat') else str(material["updated_at"])
                }
                serialized.append(clean_material)
            except Exception as e:
                logger.warning(f"Error serializing material {material.get('id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Returning {len(serialized)} materials for std='{std}', subject='{subject}'")
        
        return {
            "status": True, 
            "message": "Chapter materials retrieved successfully", 
            "data": {
                "materials": serialized,
                "total": len(serialized),
                "filters_applied": {
                    "std": std,
                    "subject": subject
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_chapter_materials_post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chapter materials")


@router.get("/recent")
async def list_recent_chapter_materials_route(
    limit: int = 5,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be a positive integer")
    admin_id = _resolve_admin_id(current_user)
    materials = list_recent_chapter_materials(admin_id, limit)
    serialized = [m if isinstance(m, dict) else m for m in materials]
    return {"status": True, "message": "Recent chapter materials retrieved successfully", "data": {"materials": serialized}}



@router.post("/lectures/{lecture_uid}/cover-photo", response_model=ResponseBase)
async def upload_lecture_cover_photo(
    lecture_uid: str,
    cover_photo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)
    lecture: Optional[LectureGen] = (
        db.query(LectureGen).filter(LectureGen.lecture_uid == lecture_uid).first()
    )
    if not lecture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found",
        )
    if lecture.admin_id != admin_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    settings = get_settings()
    s3_service = get_s3_service(settings)
    upload_result = await upload_image_to_s3(
        cover_photo,
        s3_service=s3_service,
        subfolder=f"lectures/{admin_id}/{lecture_uid}/cover-photos",
    )
    lecture.cover_photo_url = upload_result.get("s3_url")
    lecture.updated_at = datetime.utcnow()
    db.add(lecture)
    db.commit()
    db.refresh(lecture)
    return ResponseBase(
        status=True,
        message="Lecture cover photo uploaded successfully",
        data={
            "lecture_uid": lecture_uid,
            "cover_photo_url": lecture.cover_photo_url,
            "material_id": lecture.material_id,
        },
    )
@router.delete("/{lecture_id}")
async def delete_chapter_material_route(
    lecture_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)
    lecture_identifier = str(lecture_id).strip()

    lecture_record = (
        db.query(LectureGen)
        .filter(LectureGen.lecture_uid == lecture_identifier)
        .first()
    )

    material_id: Optional[int] = None
    if lecture_record and lecture_record.admin_id == admin_id:
        material_id = lecture_record.material_id
    elif lecture_identifier.isdigit():
        # Backward compatibility: allow direct material_id usage
        material_id = int(lecture_identifier)

    if material_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found")

    chapter_material = get_chapter_material(material_id)
    if not chapter_material or (chapter_material.get("admin_id") if isinstance(chapter_material, dict) else chapter_material.admin_id) != admin_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    delete_chapter_material_db(material_id)

    if lecture_record:
        try:
            db.delete(lecture_record)
            db.commit()
        except Exception:
            db.rollback()

    return {"status": True, "message": "Chapter material deleted successfully"}


# -------------------------
# Topic extraction endpoint (uses the user's topic_extractor)
# -------------------------
@router.post("/extract-topics")
async def extract_topics_from_materials(
    request_data: TopicExtractRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material_ids = request_data.material_ids
    
    # Validate input
    if not material_ids or len(material_ids) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_ids is required and cannot be empty")
    
    if len(material_ids) > 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot process more than 5 materials at once")

    try:
        from app.utils.topic_extractor import extract_topics_from_pdf
        logger.info(f"Extracting topics for {len(material_ids)} materials by user {current_user.get('email')}")
        topics_by_material: List[Dict[str, Any]] = []
        for material_id in material_ids:
            entry: Dict[str, Any] = {
                "material_id": material_id,
                "chapter_title": "",
                "topics": [],
            }
            try:
                # Validate material exists and user has access
                material = get_chapter_material(material_id)
                if not material:
                    logger.warning(f"Material {material_id} not found")
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Material {material_id} not found")

                material_admin_id = material.get("admin_id") if isinstance(material, dict) else getattr(material, "admin_id", None)
                material_is_global = bool(material.get("is_global") if isinstance(material, dict) else getattr(material, "is_global", False))

                # Permission check
                if current_user["role"] == "admin":
                    if not current_user.get("is_super_admin") and not material_is_global:
                        if material_admin_id != current_user["id"]:
                            logger.warning(f"Access denied for material {material_id}")
                            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied for material {material_id}")

                elif current_user["role"] == "member":
                    member_admin_id = _resolve_member_admin_id(current_user)
                    if not material_is_global and material_admin_id != member_admin_id:
                        logger.warning(f"Access denied for material {material_id}")
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied for material {material_id}")
                else:
                    logger.warning(f"Access denied for material {material_id}")
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied for material {material_id}")

                # Get file path (S3 URL or local path)
                material_file_path = material.get("file_path") if isinstance(material, dict) else material.file_path
                
                # Handle S3 files
                temp_file_path = None
                try:
                    if material_file_path.startswith("https://") and ".s3." in material_file_path:
                        # This is an S3 URL, download it temporarily
                        from app.utils.s3_file_handler import get_s3_service
                        settings = get_settings()
                        s3_service = get_s3_service(settings)
                        
                        # Extract S3 key from URL (e.g., "https://bucket.s3.region.amazonaws.com/key" -> "key")
                        s3_key = material_file_path.split(f"{settings.aws_s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/", 1)[-1]
                        
                        # Download file from S3
                        file_content = s3_service.get_file(s3_key)
                        if not file_content:
                            logger.warning(f"File not found in S3 for material {material_id}: {material_file_path}")
                            entry["error"] = f"PDF file not found in storage: {material_file_path.split('/')[-1]}"
                            entry["error_type"] = "FILE_NOT_FOUND"
                            topics_by_material.append(entry)
                            continue
                        
                        # Save to temporary file
                        import tempfile
                        temp_fd, temp_file_path = tempfile.mkstemp(suffix=".pdf")
                        os.write(temp_fd, file_content)
                        os.close(temp_fd)
                        material_file_path = temp_file_path
                    else:
                        # Local file path
                        if not os.path.exists(material_file_path):
                            logger.warning(f"File not found for material {material_id}: {material_file_path}")
                            entry["error"] = f"PDF file not found on server: {os.path.basename(material_file_path)}"
                            entry["error_type"] = "FILE_NOT_FOUND"
                            topics_by_material.append(entry)
                            continue

                    # Extract topics
                    # Extract topics with queue management to prevent server overload
                    try:
                        async with topic_extraction_queue.acquire():
                            extraction = await asyncio.to_thread(
                                extract_topics_from_pdf,
                                Path(material_file_path),
                            )
                    except TopicExtractionQueueFullError:
                        logger.warning("Topic extraction queue is full for material %s", material_id)
                        entry["error"] = "Server is busy processing other topic extractions. Please try again shortly."
                        entry["error_type"] = "QUEUE_FULL"
                        topics_by_material.append(entry)
                        continue
                    except TopicExtractionQueueTimeoutError:
                        logger.warning("Topic extraction queue timed out for material %s", material_id)
                        entry["error"] = "Timed out while waiting for topic extraction. Please retry later."
                        entry["error_type"] = "QUEUE_TIMEOUT"
                        topics_by_material.append(entry)
                        continue
                    if not extraction.get("success", True):
                        entry["error"] = extraction.get("error") or "No text could be extracted from the PDF."
                        entry["error_type"] = extraction.get("error_type") or "NO_TEXT_EXTRACTED"
                        topics_by_material.append(entry)
                        logger.warning(
                            "Topic extraction returned no content for material %s: %s",
                            material_id,
                            entry["error"],
                        )
                        continue
                    # Save to files
                    txt_path, json_path = save_extracted_topics_files(material_admin_id, material_id, extraction)

                    # Return only topics with a single chapter title value
                    topics = extraction.get("topics", [])
                    material_chapter_number = material.get("chapter_number") if isinstance(material, dict) else material.chapter_number
                    chapter_title = (
                        extraction.get("chapter_title")
                        or (extraction.get("chapter_titles") or [None])[0]
                        or material_chapter_number
                        or ""
                    )

                    entry.update({
                        "chapter_title": chapter_title,
                        "topics": topics,
                    })
                    topics_by_material.append(entry)
                    
                    logger.info(f"Successfully extracted {len(topics)} topics for material {material_id}")
                finally:
                    # Clean up temporary file if it was created
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temp file {temp_file_path}: {cleanup_error}")

            except HTTPException:
                # Re-raise HTTP exceptions (permission errors, not found, etc.)
                raise
            except Exception as e:
                logger.error(f"Error extracting topics for material {material_id}: {str(e)}")
                entry["error"] = f"Failed to extract topics: {str(e)}"
                entry["error_type"] = "EXTRACTION_ERROR"
                topics_by_material.append(entry)
                continue

        # Calculate statistics
        successful_count = sum(1 for item in topics_by_material if item.get("topics"))
        failed_count = sum(1 for item in topics_by_material if item.get("error"))
        errors_payload = [
            {
                "material_id": item["material_id"],
                "error": item.get("error"),
                "error_type": item.get("error_type"),
            }
            for item in topics_by_material
            if item.get("error")
        ]
        if successful_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Topic extraction failed for all provided materials.",
                    "errors": errors_payload,
                },
            )
        message = (
            f"Topics extraction completed with {failed_count} error(s)"
            if failed_count > 0
            else "Topics extracted successfully"
        )
        # Build appropriate message
        message = (
            f"Topics extraction completed with {failed_count} error(s)"
            if failed_count > 0
            else "Topics extracted successfully"
        )
        return {
            "status": True, 
            "message": message, 
            "data": {
                "topics": topics_by_material,
                "total_materials": len(material_ids),
                "successful_extractions": successful_count,
                "failed_extractions": failed_count,
                "errors": errors_payload,
            }
        }
        
    except HTTPException as http_exc:
        # Re-raise FastAPI HTTP exceptions (e.g., permission errors) without wrapping them as 500s
        raise http_exc
    except Exception as exc:
        logger.exception("Unexpected error in extract_topics_from_materials")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract topics")




@router.get("/recent-topics")
async def list_recent_material_topics(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be a positive integer")
    admin_id = _resolve_admin_id(current_user)
    materials = list_recent_chapter_materials(admin_id, limit)

    results = []
    for material in materials:
        payload, topics_list = read_topics_file_if_exists(material.get("admin_id") if isinstance(material, dict) else material.admin_id, material.get("id") if isinstance(material, dict) else material.id)
        results.append({"material": material if isinstance(material, dict) else (material.to_dict() if hasattr(material, "to_dict") else material.__dict__), "topics": topics_list, "topics_metadata": payload})

    return {"status": True, "message": "Recent topics retrieved successfully", "data": {"items": results}}


@router.get("/{material_id}/topics")
async def get_material_topics_route(
    material_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    material_admin_id = material.get("admin_id") if isinstance(material, dict) else getattr(material, "admin_id", None)
    material_is_global = bool(material.get("is_global") if isinstance(material, dict) else getattr(material, "is_global", False))

    if current_user["role"] == "admin":
        if not current_user.get("is_super_admin") and not material_is_global:
            if material_admin_id != current_user["id"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if not material_is_global and material_admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    payload, topics_list = read_topics_file_if_exists(
        material.get("admin_id") if isinstance(material, dict) else material.admin_id,
        material.get("id") if isinstance(material, dict) else material.id
    )
    if not topics_list:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topics not found. Please extract topics first.")

    sanitized_topics: List[Dict[str, Any]] = []
    for topic in topics_list:
        if not isinstance(topic, dict):
            continue
        topic_copy = dict(topic)
        if topic_copy.get("is_assistant_generated"):
            suggestion_value = topic_copy.get("suggestion_topic_id") or topic_copy.get("topic_id")
            if suggestion_value is not None:
                topic_copy["suggestion_topic_id"] = str(suggestion_value)
            topic_copy.pop("topic_id", None)
        else:
            topic_id_value = topic_copy.get("topic_id")
            if topic_id_value is not None:
                topic_copy["topic_id"] = str(topic_id_value)
            topic_copy.pop("suggestion_topic_id", None)
        sanitized_topics.append(topic_copy)

    chapter_title = (payload or {}).get("chapter_title") or ""

    return {
        "status": True,
        "message": "Topics fetched successfully",
        "data": {
            "material_id": material.get("id") if isinstance(material, dict) else material.id,
            "topics_count": len(sanitized_topics),
            "chapter_title": chapter_title,
            "topic_id": sanitized_topics,
        },
    }


@router.post("/{material_id}/topics")
async def add_manual_topic_route(
    material_id: int,
    topic: ManualTopicCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not topic.title.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Topic title is required")

    material = get_chapter_material(material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    material_admin_id = material.get("admin_id") if isinstance(material, dict) else getattr(material, "admin_id", None)
    material_is_global = bool(material.get("is_global") if isinstance(material, dict) else getattr(material, "is_global", False))

    # Authorization checks
    if current_user["role"] == "admin":
        if not current_user.get("is_super_admin") and not material_is_global:
            if material_admin_id != current_user["id"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if not material_is_global and material_admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    normalized_subtopics = []
    if topic.subtopics:
        for subtopic in topic.subtopics:
            if not subtopic.get("title") and not subtopic.get("narration"):
                continue
            normalized_subtopics.append({"title": (subtopic.get("title") or "").strip(), "narration": (subtopic.get("narration") or "").strip()})

    new_topic = {"title": topic.title.strip(), "summary": (topic.summary or "").strip(), "subtopics": normalized_subtopics, "is_manual": True}

    material_admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
    material_id = material.get("id") if isinstance(material, dict) else material.id
    material_chapter_number = material.get("chapter_number") if isinstance(material, dict) else material.chapter_number
    added = append_manual_topic_to_file(material_admin_id, material_id, material_chapter_number, new_topic)

    return {"status": True, "message": "Topic added successfully", "data": added}


# -------------------------
# Assistant suggest topics (calls Groq)
# -------------------------
@router.post("/{material_id}/assistant-suggest-topics")
async def assistant_suggest_topics(
    material_id: int,
    request: AssistantSuggestRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    material_admin_id = material.get("admin_id") if isinstance(material, dict) else getattr(material, "admin_id", None)
    material_is_global = bool(material.get("is_global") if isinstance(material, dict) else getattr(material, "is_global", False))

    # Authorization
    if current_user["role"] == "admin":
        admin_id = current_user["id"]
        if not current_user.get("is_super_admin") and not material_is_global:
            if material_admin_id != admin_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        admin_id = current_user["admin_id"]
        if not material_is_global and material_admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    topics_path = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}", f"extracted_topics_{material.get('id') if isinstance(material, dict) else material.id}.json")
    if not os.path.exists(topics_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topics not found. Please extract topics first.")

    try:
        with open(topics_path, "r", encoding="utf-8") as fh:
            topics_blob = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read topics file") from exc

    existing_topics = topics_blob.get("topics", [])
    topics_text = topics_blob.get("topics_text", "")
    excerpt = topics_blob.get("excerpt", "")

    pdf_context_text = ""
    try:
        pdf_context_text = read_pdf_context_for_material(material.get("file_path") if isinstance(material, dict) else material.file_path) or excerpt or topics_text
    except Exception:
        pdf_context_text = excerpt or topics_text

    if not pdf_context_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF content unavailable")

    condensed_topics = []
    for index, topic in enumerate(existing_topics, start=1):
        if isinstance(topic, dict):
            title = str(topic.get("title", f"Topic {index}"))
            summary = str(topic.get("summary", "")).strip()
            condensed_topics.append(f"{index}. {title}{' — ' + summary if summary else ''}")
        else:
            condensed_topics.append(f"{index}. {topic}")

    topics_summary = "\n".join(condensed_topics)[:4_000]

    context_payload = (
        "# Existing Topics\n"
        f"{topics_summary if topics_summary else 'No topics extracted yet.'}\n\n"
        "# PDF Content\n"
        f"{pdf_context_text}"
    )

    user_query = request.user_query
    temperature = 0.2  # Fixed temperature
    plan_label = _resolve_plan_label_for_admin(admin_id, current_user, request.plan_label)
    plan_limit = PLAN_SUGGESTION_LIMITS.get(plan_label) if plan_label else None

    addition_text = "Only suggest genuinely grounded subtopics."
    limit_text = (
        f"You must return no more than {MAX_ASSISTANT_SUGGESTIONS} suggested subtopics per response. {addition_text}"
    )

    material_language_code = topics_blob.get("language_code") or DEFAULT_LANGUAGE_CODE
    language_rule = LANGUAGE_OUTPUT_RULES.get(material_language_code, LANGUAGE_OUTPUT_RULES[DEFAULT_LANGUAGE_CODE])
    language_instruction = language_rule["instruction"]
    language_label = language_rule["label"]

    system_prompt = (
        "You are an AI assistant for educational content. "
        "Analyze the provided PDF content and existing topics. "
        "Suggest NEW, relevant subtopics that are missing but present in the PDF. "
        f"{language_instruction}. "
        f"Return ONLY valid JSON with this structure: "
        '{"suggestions": [{"title": "...", "summary": "...", "supporting_quote": "..."}]}'
        f"\nMaximum {MAX_ASSISTANT_SUGGESTIONS} suggestions. "
        f"{limit_text} "
        "Only respond when the user query clearly references the provided chapter topics or PDF content. "
        'If the query is unrelated, respond with EXACTLY this JSON (no additional keys or text): '
        '{"suggestions": [], "message": "I am currently answering questions related to this chapter only. Please ask a question based on the ongoing lecture topic."}'
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GROQ_API_KEY environment variable is not set")

    client = Groq(api_key=api_key)
    model = "openai/gpt-oss-120b"

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_payload},
        {"role": "user", "content": user_query},
    ]

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_completion_tokens=4000,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Assistant API failed: {exc}") from exc

    if not completion.choices:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No response from assistant")

    raw_reply = (completion.choices[0].message.content or "").strip()
    suggestions = []
    reply_text = raw_reply
    rejection_notice: Optional[str] = None

    try:
        parsed = json.loads(raw_reply)
        if isinstance(parsed, dict):
            parsed_suggestions = parsed.get("suggestions", [])
            rejection_message = str(parsed.get("message", "")).strip() if parsed.get("message") is not None else ""

            if (not parsed_suggestions or not isinstance(parsed_suggestions, list)) and rejection_message:
                rejection_notice = rejection_message or "I am currently answering questions related to this chapter only. Please ask a question based on the ongoing lecture topic."
                suggestions = []
                reply_text = rejection_notice
            else:
                for idx, item in enumerate(parsed_suggestions, start=1):
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", "")).strip()
                    summary = str(item.get("summary", "")).strip()
                    quote = str(item.get("supporting_quote", "")).strip()
                    if title and quote:
                        suggestions.append({
                            "suggestion_id": str(item.get("suggestion_id") or item.get("id") or idx),
                            "title": title,
                            "summary": summary,
                            "supporting_quote": quote[:240],
                        })
                if len(suggestions) > MAX_ASSISTANT_SUGGESTIONS:
                    suggestions = suggestions[:MAX_ASSISTANT_SUGGESTIONS]
                if suggestions:
                    reply_lines = ["Here are topic suggestions based on your PDF:"]
                    for idx, suggestion in enumerate(suggestions, start=1):
                        summary_part = f" — {suggestion['summary']}" if suggestion['summary'] else ""
                        reply_lines.append(f"{idx}. {suggestion['title']}{summary_part}")
                    reply_text = "\n".join(reply_lines)
                else:
                    reply_text = "No additional grounded subtopics were found in the supplied PDF excerpt."
    
    except json.JSONDecodeError:
        logger.warning("Assistant did not return valid JSON")
        suggestions = []
        reply_text = "I couldn't generate structured suggestions. Please try rephrasing your query."

    if rejection_notice:
        return {
            "status": False,
            "message": rejection_notice,
            "data": {
                "suggestions": [],
                "reply": rejection_notice,
                "plan_label": plan_label,
                "plan_limit": plan_limit,
                "max_suggestions": MAX_ASSISTANT_SUGGESTIONS,
                "language_code": material_language_code,
                "language_label": language_label,
                "existing_topics_count": len(existing_topics),
                "reason": "UNRELATED_QUERY",
            },
        }

    if suggestions:
        persist_assistant_suggestions(admin_id, material.get("id") if isinstance(material, dict) else material.id, suggestions)

    return {
        "status": True,
        "message": "Suggestions generated",
        "data": {
            "suggestions": suggestions,
            "reply": reply_text,
            "plan_label": plan_label,
            "plan_limit": plan_limit,
            "max_suggestions": MAX_ASSISTANT_SUGGESTIONS,
            "language_code": material_language_code,
            "language_label": language_label,
            "existing_topics_count": len(existing_topics),
        },
    }


@router.post("/{material_id}/assistant-add-topics")
async def assistant_add_topics_route(
    material_id: int,
    request: AssistantAddTopicsRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    if current_user["role"] == "admin":
        admin_id = current_user["id"]
        if (material.get("admin_id") if isinstance(material, dict) else material.admin_id) != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        admin_id = member_admin_id or current_user.get("admin_id")
        if (material.get("admin_id") if isinstance(material, dict) else material.admin_id) != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    selected_suggestions = list(request.selected_suggestions or [])
    suggestion_ids = request.suggestion_ids or []

    if suggestion_ids:
        resolved, missing = get_cached_suggestions_by_ids(
            material.get("admin_id") if isinstance(material, dict) else material.admin_id,
            material.get("id") if isinstance(material, dict) else material.id,
            suggestion_ids
        )
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suggestions not found for IDs: {', '.join(missing)}",
            )
        selected_suggestions.extend(resolved)

    if not selected_suggestions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No suggestions selected to add")

    admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
    material_id = material.get("id") if isinstance(material, dict) else material.id
    chapter_number = material.get("chapter_number") if isinstance(material, dict) else material.chapter_number
    
    
    plan_label = _resolve_plan_label_for_admin(admin_id, current_user, requested_plan_label=None)
    plan_limit = PLAN_SUGGESTION_LIMITS.get(plan_label) if plan_label else None

    if plan_limit is not None:
        _, existing_topics = read_topics_file_if_exists(admin_id, material_id)
        existing_topics = existing_topics or []
        existing_titles = {
            str(topic.get("title", "")).strip().lower()
            for topic in existing_topics
            if isinstance(topic, dict)
        }
        assistant_generated_count = sum(
            1
            for topic in existing_topics
            if isinstance(topic, dict) and topic.get("is_assistant_generated")
        )
        remaining_capacity = max(plan_limit - assistant_generated_count, 0)

        unique_selected_titles: Set[str] = set()
        for suggestion in selected_suggestions:
            if not isinstance(suggestion, dict):
                continue
            title = str(suggestion.get("title", "")).strip()
            if not title:
                continue
            unique_selected_titles.add(title.lower())

        addable_titles = [title for title in unique_selected_titles if title not in existing_titles]
        requested_additions = len(addable_titles)

        if remaining_capacity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Assistant topic limit reached. Under the {plan_label} plan you can add a maximum of {plan_limit} assistant topics. "
                    f"You have already added {assistant_generated_count}."
                ),
            )

        if requested_additions > remaining_capacity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Under the {plan_label} plan you can add a maximum of {plan_limit} assistant topics. "
                    f"You have already added {assistant_generated_count}. You can add only {remaining_capacity} more."
                ),
            )
    
    result = add_assistant_topics_to_file(admin_id, material_id, chapter_number, selected_suggestions)
    sanitized_added: List[Dict[str, Any]] = []
    for topic in result.get("added_topics", []) or []:
        if not isinstance(topic, dict):
            continue
        topic_copy = dict(topic)
        topic_copy["suggestion_topic_id"] = str(topic_copy.get("topic_id"))
        topic_copy.pop("topic_id", None)
        sanitized_added.append(topic_copy)
    result["added_topics"] = sanitized_added

    return {
        "status": True,
        "message": f"Added {len(result.get('added_topics', []))} topics from assistant suggestions",
        "data": result,
    }


# -------------------------
# Lecture generation endpoints (calls lecture_service)
# -------------------------
@router.post("/chapter_lecture/config")
async def post_lecture_generation_config(
    payload: LectureConfigRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    _ensure_lecture_config_access(current_user)
    admin_id = _resolve_admin_id(current_user)
    plan_label, allowed_durations = _get_allowed_duration_options(admin_id,current_user)

    requested_language = _normalize_requested_language(payload.language) or payload.language
    requested_duration = _normalize_requested_duration(payload.duration) if payload.duration is not None else payload.duration
    incoming_model = _normalize_requested_model(payload.model)
    requested_model = incoming_model
    
    if payload.merged_id:
        merged_payload = _load_merged_lecture_payload(payload.merged_id)
        if merged_payload and "response" in merged_payload:
            response_data = merged_payload["response"]
            data_section = response_data.setdefault("data", {})
            save_required = False

            normalized_language = _normalize_requested_language(payload.language)
            if normalized_language:
                data_section["language"] = normalized_language
                requested_language = normalized_language
                save_required = True
            elif not requested_language:
                requested_language = data_section.get("language")

            normalized_duration = (
                _normalize_requested_duration(payload.duration)
                if payload.duration is not None
                else None
            )
            if normalized_duration is not None:
                data_section["duration"] = normalized_duration
                requested_duration = normalized_duration
                save_required = True
            elif requested_duration is None:
                requested_duration = data_section.get("duration")

            normalized_existing_model = _normalize_requested_model(data_section.get("model"))



            if incoming_model:
                data_section["model"] = incoming_model
                requested_model = incoming_model
                save_required = True
            elif requested_model is None:
                requested_model = normalized_existing_model

            if save_required:
                try:
                    _save_merged_lecture_payload(payload.merged_id, merged_payload)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist lecture config overrides for %s: %s",
                        payload.merged_id,
                        exc,
                    )
    _enforce_plan_duration(requested_duration, allowed_durations, plan_label)
    config_response = _build_lecture_config_response(
        requested_language=requested_language,
        requested_duration=requested_duration,
        duration_options=allowed_durations,
        requested_model=requested_model,
    )
    config_response["merged_id"] = payload.merged_id

    return {
        "status": True,
        "message": "Lecture configuration fetched successfully",
        "data": config_response,
    }
async def _ensure_audio_file(lecture_id: str, filename: str) -> Path:
    """Return an existing audio file path or generate it on demand."""
    storage_base = Path(__file__).parent.parent.parent / "storage" / "chapter_lectures"
    audio_path = storage_base / lecture_id / "audio" / filename

    if audio_path.exists():
        return audio_path

    try:
        from app.services.tts_service import GoogleTTSService
        from app.postgres import get_pg_cursor
        from app.config import get_settings

        # Fetch lecture data from database
        with get_pg_cursor() as cur:
            cur.execute(
                "SELECT lecture_data FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s",
                {"lecture_uid": lecture_id},
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Lecture not found",
                    "lecture_id": lecture_id,
                },
            )

        lecture_data = row.get("lecture_data") or {}
        slides = lecture_data.get("slides") or []
        language = lecture_data.get("language", "English")

        try:
            slide_num = int(filename.replace("slide-", "").replace(".mp3", ""))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid filename format"},
            )

        target_slide = None
        for slide in slides:
            if slide.get("number") == slide_num or (slides.index(slide) + 1) == slide_num:
                target_slide = slide
                break

        if not target_slide or not target_slide.get("narration"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Slide narration not found",
                    "lecture_id": lecture_id,
                    "slide_number": slide_num,
                },
            )

        settings = get_settings()
        tts_service = GoogleTTSService(
            storage_root=str(storage_base.parent),
            credentials_path=getattr(settings, "gcp_tts_credentials_path", None),
        )
        audio_path_result = await tts_service.synthesize_text(
            lecture_id=lecture_id,
            text=target_slide["narration"],
            language=language,
            filename=filename,
            subfolder="audio",
        )

        if not audio_path_result or not audio_path_result.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to generate audio"},
            )

        return audio_path_result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error generating audio for {lecture_id}/{filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to generate or retrieve audio",
                "lecture_id": lecture_id,
                "filename": filename,
            },
        )


@router.get("/chapter_lecture/audio/{lecture_id}/{filename}")
async def get_lecture_audio(
    lecture_id: str,
    filename: str,
):
    audio_path = await _ensure_audio_file(lecture_id, filename)
    return FileResponse(audio_path, media_type="audio/mpeg")


@router.get("/chapter_lecture/audio/{lecture_id}/{filename}/download")
async def download_lecture_audio(
    lecture_id: str,
    filename: str,
):
    audio_path = await _ensure_audio_file(lecture_id, filename)
    response = FileResponse(audio_path, media_type="audio/mpeg")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response

@router.get("/chapter_lectures/{std}/{subject}/{lecture_id}")
async def get_lecture_json(
    std: str,
    subject: str,
    lecture_id: str,
):
    """
    PUBLIC endpoint to serve lecture JSON file.
    URL: /chapter-materials/chapter_lectures/{std}/{subject}/{lecture_id}.json
    Example: /chapter-materials/chapter_lectures/9/science/4172d9c9c0e6.json
    """
    # Remove .json extension if present
    lecture_id_clean = lecture_id.replace(".json", "")
    
    # Build the file path
    storage_base = Path("./storage/chapter_lectures")
    lecture_path = storage_base / lecture_id_clean / "lecture.json"
    
    # Log for debugging
    logger.info(f"🔍 Searching for lecture: {lecture_id_clean}")
    logger.info(f"📂 Path: {lecture_path}")
    logger.info(f"✅ Exists: {lecture_path.exists()}")
    
    if not lecture_path.exists():
        logger.error(f"❌ Lecture file not found at: {lecture_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Lecture file not found",
                "lecture_id": lecture_id_clean,
                "expected_path": str(lecture_path),
                "message": "The lecture JSON file does not exist. Please generate the lecture first."
            }
        )
    
    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            lecture_data = json.load(f)
        
        # Add metadata
        lecture_data["std"] = std.replace("_", " ").title()
        lecture_data["subject"] = subject.replace("_", " ").title()
        lecture_data["accessed_at"] = datetime.now().isoformat()
        lecture_data["file_path"] = str(lecture_path)
        
        logger.info(f"✅ Successfully serving lecture: {lecture_id_clean}")
        
        return JSONResponse(
            content=lecture_data,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in file {lecture_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON format in lecture file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error reading lecture file {lecture_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading lecture file: {str(e)}"
        )


# Modified generate_lecture_from_topics endpoint
@router.post("/chapter_lecture/generate")
async def generate_lecture_from_topics(
    request: LectureGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
    lecture_service: LectureService = Depends(get_lecture_service),
):
    """Generate lecture content from either material topics or merged lecture payloads."""

    settings = get_settings()
    style = request.style or "storytelling"

    if request.merged_lecture_id:
        context_payload = _prepare_generation_from_merged(
            request=request,
            current_user=current_user,
            db=db,
            settings=settings,
        )
    else:
        context_payload = _prepare_generation_from_material(
            request=request,
            current_user=current_user,
            db=db,
            settings=settings,
        )

    lecture_record = await lecture_service.create_lecture_from_text(
        text=context_payload["aggregate_text"],
        language=context_payload["language"],
        duration=context_payload["duration"],
        style=style,
        title=context_payload["title"],
        metadata=context_payload["metadata"],
        model=context_payload.get("model"),
    )
# Record lecture credit usage for this admin (1 credit per generated lecture)
    try:
        admin_id = _resolve_admin_id(current_user)
        lecture_credit_repository.upsert_usage(admin_id, credits_delta=1)
    except Exception as exc:
        logger.warning("Failed to record lecture credit usage for admin %s: %s", admin_id, exc)
    # ============================================================================
    # URL GENERATION AND TERMINAL PRINTING
    # ============================================================================
    
    # Get lecture ID from the generated lecture
    lecture_id = lecture_record.get("lecture_id", "")

    # Generate JSON URL with class and subject
    std_slug = context_payload["std_slug"]
    subject_slug = context_payload["subject_slug"]
    
    # Use chapter-materials prefix for the URL
    lecture_json_url = f"/chapter-materials/chapter_lectures/{std_slug}/{subject_slug}/{lecture_id}.json"

    # Add URL to lecture_record
    lecture_record["lecture_json_url"] = lecture_json_url

    # Print detailed information to terminal
    print(f"\n{'='*60}")
    print(f"📚 LECTURE GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Lecture ID: {lecture_id}")
    log_context = context_payload["log_context"]
    print(f"Material ID: {log_context['material_id'] or 'N/A'}")
    print(f"Class (STD): {log_context['std']}")
    print(f"Subject: {log_context['subject']}")
    print(f"Board: {log_context['board']}")
    print(f"Semester: {log_context['sem']}")
    print(f"Title: {context_payload['title']}")
    print(f"Language: {context_payload['language']}")
    print(f"Duration: {context_payload['duration']} minutes")
    print(f"Style: {style}")
    print(f"Selected Topics: {log_context['selected_topics_count']}")
    print(f"Total Slides: {lecture_record.get('total_slides', 'N/A')}")
    print(f"Fallback Used: {lecture_record.get('fallback_used', False)}")
    print(f"")
    print(f"📄 JSON URL: {lecture_json_url}")
    print(f"🌐 Full URL: http://localhost:3020{lecture_json_url}")
    if lecture_record.get("lecture_path"):
        print(f"📂 File Path: {lecture_record.get('lecture_path')}")
    print(f"{'='*60}\n")

    material_snapshot = context_payload.get("material_snapshot") or {}
    lecture_record["db_saved"] = bool(lecture_record.get("db_record_id"))

    return {
        "status": True,
        "message": "Lecture generated successfully",
        "data": {
            "lecture": {
                **lecture_record,
                "lecture_json_url": lecture_json_url,
                "db_record_id": lecture_record.get("db_record_id"),
                "db_saved": lecture_record.get("db_saved", False),
                "material_info": material_snapshot,
                "selected_topic_ids": request.selected_topic_ids,
            }
        }
    }


@router.post("/chapter_lecture/generate/{lecture_id}")
async def generate_lecture_from_path(
    lecture_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
    lecture_service: LectureService = Depends(get_lecture_service),
):
    request_payload = LectureGenerationRequest.model_validate({"lecture_id": lecture_id})
    return await generate_lecture_from_topics(
        request=request_payload,
        current_user=current_user,
        db=db,
        lecture_service=lecture_service,
    )

@router.post("/chapter_lecture/{lecture_id}/chat")
async def chat_about_lecture(
    lecture_id: str,
    request: LectureChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    lecture_service: LectureService = Depends(get_lecture_service),
):
    try:
        answer = await lecture_service.answer_question(
            lecture_id=lecture_id,
            question=request.question,
            answer_type=request.answer_type,
            is_edit_command=request.is_edit_command,
            context_override=request.context_override,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to process chat request") from exc

    if isinstance(answer, dict):
        payload = answer
    else:
        payload = {"answer": answer}

    return {"status": True, "message": "Response generated", "data": payload}
# -------------------------
# -------------------------

@router.post("/create_merged_chapter_lecture")
async def create_merged_lecture(
    request: CreateMergedLectureRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        # Validate input
        if not request:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Lecture data is required")

        title = (request.title or "").strip()

        materials = request.materials or []
        topic_selections = request.topic_selections or []
        selected_topics_payload = request.selected_topics or {}

        if not materials and not topic_selections and not selected_topics_payload:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either materials or topics must be provided")

        # Validate materials if provided
        if materials:
            for material in materials:
                if not isinstance(material, dict) or "id" not in material:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid material format")

        # Resolve topics coming in via explicit payload (legacy support)
        combined_topics: Dict[str, List[Dict[str, Any]]] = {}
        selection_summary: Dict[str, Dict[str, Any]] = {}
        extracted_chapter_title = ""
        
        for material_key, topics in selected_topics_payload.items():
            if not isinstance(topics, list):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Topics for material {material_key} must be an array")
            filtered = []
            for topic in topics:
                if not isinstance(topic, dict) or "title" not in topic:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid topic format")
                filtered.append(topic)
            if filtered:
                combined_topics[str(material_key)] = filtered
            
            # Extract chapter_title from payload if available
            if not extracted_chapter_title:
                try:
                    material_id_int = int(material_key)
                    material = get_chapter_material(material_id_int)
                    if material:
                        material_admin_id = material.get("admin_id") if isinstance(material, dict) else material.admin_id
                        payload, _ = read_topics_file_if_exists(material_admin_id, material_id_int)
                        if payload:
                            extracted_chapter_title = payload.get("chapter_title", "")
                except (ValueError, TypeError):
                    pass

        # Resolve topics via topic selections (material + indices)
        if topic_selections:
            for selection in topic_selections:
                if not isinstance(selection, TopicSelection):
                    continue

                material = get_chapter_material(selection.material_id)
                if not material:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Material {selection.material_id} not found")

                material_admin_id = material.get("admin_id") if isinstance(material, dict) else getattr(material, "admin_id", None)
                material_is_global = bool(material.get("is_global") if isinstance(material, dict) else getattr(material, "is_global", False))

                # Authorization checks
                if current_user["role"] == "admin":
                    if not current_user.get("is_super_admin") and not material_is_global:
                        if material_admin_id != current_user["id"]:
                            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
                elif current_user["role"] == "member":
                    member_admin_id = _resolve_member_admin_id(current_user)
                    if not material_is_global and material_admin_id != member_admin_id:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
                else:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

                payload, topics_list = read_topics_file_if_exists(material["admin_id"], material["id"])
                if not topics_list:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No topics found for material {material['id']}")

                # Extract chapter_title from payload if available
                if payload and not extracted_chapter_title:
                    extracted_chapter_title = payload.get("chapter_title", "")

                key = str(material["id"])
                resolved_topics: List[Dict[str, Any]] = []
                selection_summary.setdefault(key, {"selected_indices": [], "selected_topic_ids": []})

                if selection.topic_indices:
                    for index in selection.topic_indices:
                        if index < 0 or index >= len(topics_list):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Topic index {index} out of range for material {material['id']}"
                            )
                        resolved_topics.append(topics_list[index])
                    selection_summary[key]["selected_indices"].extend(selection.topic_indices)

                selection_topic_ids: List[str] = []
                if selection.topic_ids:
                    selection_topic_ids.extend(selection.topic_ids)
                if selection.suggestion_topic_ids:
                    selection_topic_ids.extend(selection.suggestion_topic_ids)

                if selection_topic_ids:
                    resolved_by_ids, missing_ids = get_topics_by_ids(material["admin_id"], material["id"], selection_topic_ids)
                    if missing_ids:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Topics not found for IDs: {', '.join(missing_ids)}",
                        )
                    resolved_topics.extend(resolved_by_ids)
                    selection_summary[key]["selected_topic_ids"].extend(selection_topic_ids)

                combined_topics.setdefault(key, []).extend(resolved_topics)

        total_topics = sum(len(topics) for topics in combined_topics.values())
        if total_topics == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one topic must be selected")

        # Use extracted chapter_title if no title was provided
        if not title:
            title = extracted_chapter_title or f"Merged Lecture {datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Generate unique lecture ID
        import uuid
        lecture_id = f"merged_lecture_{uuid.uuid4().hex[:8]}"
        
        # Log the creation
        logger.info(
            f"Creating merged lecture '{title}' with {len(materials) or len(combined_topics)} materials "
            f"Creating merged lecture '{title}' with {len(materials) or len(combined_topics.keys())} materials "
            f"and {total_topics} topics by user {current_user.get('email')}"
        )
        
        # TODO: Implement actual lecture creation logic
        # For now, return success response
        
        # Count materials and topics properly
        materials_count = len(materials) if materials else len(combined_topics)
        topics_count = total_topics

        merged_topics_response: List[Dict[str, Any]] = []
        for material_key, topics in combined_topics.items():
            try:
                material_id_value: Any = int(material_key)
            except ValueError:
                material_id_value = material_key
            sanitized_topics: List[Dict[str, Any]] = []
            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                topic_copy = dict(topic)
                if topic_copy.get("is_assistant_generated"):
                    suggestion_value = topic_copy.get("suggestion_topic_id") or topic_copy.get("topic_id")
                    if suggestion_value is not None:
                        topic_copy["suggestion_topic_id"] = str(suggestion_value)
                    topic_copy.pop("topic_id", None)
                else:
                    topic_id_value = topic_copy.get("topic_id")
                    if topic_id_value is not None:
                        topic_copy["topic_id"] = str(topic_id_value)
                    topic_copy.pop("suggestion_topic_id", None)
                sanitized_topics.append(topic_copy)
            merged_topics_response.append(
                {
                    "material_id": material_id_value,
                    "topics_count": len(topics),
                    "topics": sanitized_topics,
                    "selection": selection_summary.get(material_key),
                }
            )

        response_payload = {
            "status": True,
            "message": "Merged lecture created successfully",
            "data": {
                "merged_id": lecture_id,
                "chapter_title": title,
                "materials_count": materials_count,
                "topics_count": topics_count,
                "created_at": datetime.utcnow().isoformat(),
                "selected_materials": list(combined_topics.keys()) if combined_topics else [m.get("id") for m in materials],
                "merged_topics": merged_topics_response,
                "language": request.language,
                "duration": request.duration,
            },
        }

        creator_admin_id = None
        try:
            creator_admin_id = _resolve_admin_id(current_user)
        except HTTPException:
            creator_admin_id = _resolve_member_admin_id(current_user)

        stored_payload = {
            "admin_id": creator_admin_id,
            "created_by": current_user.get("id"),
            "created_by_role": current_user.get("role"),
            "response": response_payload,
        }
        try:
            _save_merged_lecture_payload(lecture_id, stored_payload)
        except Exception as exc:
            logger.warning("Failed to cache merged lecture %s: %s", lecture_id, exc)

        return response_payload
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating merged lecture: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create merged lecture")


@router.get("/merged-topics/{lecture_id}")
async def get_merged_lecture(
    lecture_id: str,
    current_user: dict = Depends(get_current_user),
):
    payload = _load_merged_lecture_payload(lecture_id)
    if not payload or "response" not in payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Merged lecture not found")

    try:
        viewer_admin_id = _resolve_admin_id(current_user)
    except HTTPException:
        # For lecture viewers (non chapter roles) fall back to member admin resolution
        viewer_admin_id = _resolve_member_admin_id(current_user)

    stored_admin_id = payload.get("admin_id")
    if stored_admin_id and viewer_admin_id != stored_admin_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for this lecture")

    response = payload["response"]
    if not isinstance(response, dict):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stored lecture payload is invalid")

    # Transform response to use chapter_title instead of title
    if "data" in response and isinstance(response["data"], dict):
        data = response["data"]
        # If title exists but chapter_title doesn't, rename title to chapter_title
        if "title" in data and "chapter_title" not in data:
            data["chapter_title"] = data.pop("title")
        # If both exist, remove title and keep chapter_title
        elif "chapter_title" in data and "title" in data:
            data.pop("title", None)

    return response


# -------------------------
# Multiple Chapter Selection endpoints (POST only)
# -------------------------
@router.post("/select-multiple-chapters")
async def select_multiple_chapters(
    request: MultipleChapterSelectionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        
        # Extract selected IDs from Pydantic model
        selected_ids = request.selected_ids
        
        logger.info(f"Select multiple chapters - selected_ids: {selected_ids}")
        
        # Validate parameters
        if not selected_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chapters selected")
        
        # Get all materials for admin
        materials = list_chapter_materials(admin_id, None, None)
        
        # Filter by selected IDs
        selected_materials = [m for m in materials if str(m.get("id") if isinstance(m, dict) else m.id) in [str(id) for id in selected_ids]]
        
        # Debug logging
        logger.info(f"Requested IDs: {selected_ids}")
        logger.info(f"Found IDs: {[m.get('id') if isinstance(m, dict) else m.id for m in selected_materials]}")
        logger.info(f"Total materials in DB: {len(materials)}")
        
        return {
            "status": True, 
            "message": f"Successfully selected {len(selected_materials)} chapters", 
            "data": {
                "total_chapters": len(selected_materials)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in select_multiple_chapters: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve selected chapters")
