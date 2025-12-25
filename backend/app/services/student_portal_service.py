"""Business logic for the student portal module."""
from __future__ import annotations

from ast import Import
from datetime import datetime
from pathlib import Path

import re
from typing import Any, Dict, List, Tuple, Optional, Union

from fastapi import HTTPException, UploadFile, status

from ..repository import (
    chapter_material_repository,
    student_portal_repository,
    student_portal_video_repository,
)
from ..schemas import (
    SendChatMessageRequest,
    StudentLoginRequest,
    StudentProfileResponse,
    StudentSignupRequest,
)
from ..utils.file_handler import delete_file, get_file_url, save_uploaded_file
from ..utils.student_portal_security import hash_password, verify_password


CHAT_ATTACHMENT_SUBDIR = "chat_attachments"
VIDEOS_SUBDIR = "videos"


DEFAULT_VIDEO_SAMPLES: List[Dict[str, Any]] = []

LEGACY_SAMPLE_VIDEO_URLS: Tuple[str, ...] = (
    # "D:/finalcode/EDinai-Backend/backend/uploads/videos/2d0dbcda-d44d-4d29-ac06-f39bb4f06e12.webm",
    # "D:/finalcode/EDinai-Backend/backend/uploads/videos/488b3812-7b93-4dd5-92d6-292cb8d0f8f2.webm",
    # "D:/finalcode/EDinai-Backend/backend/uploads/videos/46205a31-8a72-44fb-9b4d-bd7d90f6740a.webm",
    # "D:/finalcode/EDinai-Backend/backend/uploads/videos/d85f4eae-2e09-4717-a6b3-3694650f41fe.webm",
    # "D:/finalcode/EDinai-Backend/backend/uploads/videos/dc31b9be-dbfa-486f-8e67-9048e310904c.mp4",
)


def _normalize_legacy_url(value: str) -> str:
    normalized = value.replace("\\", "/").lstrip("/")
    return normalized.lower()


LEGACY_SAMPLE_VIDEO_URL_SET = {_normalize_legacy_url(item) for item in LEGACY_SAMPLE_VIDEO_URLS}
LEGACY_SAMPLE_FILENAMES: Tuple[str, ...] = tuple(Path(item).name.lower() for item in LEGACY_SAMPLE_VIDEO_URLS)


def _resolve_media_url(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://") or value.startswith("//"):
        return value
    return get_file_url(value)


def _format_date_ddmmyyyy(value: Optional[Union[str, datetime]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_value = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            dt_value = datetime.fromisoformat(normalized)
        except ValueError:
            return text
    return dt_value.strftime("%d/%m/%Y")


def _seconds_to_minutes(value: Optional[Union[int, str]]) -> Optional[int]:
    if value is None:
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return 0
    return (seconds + 59) // 60


def _format_progress_text(
    watched_seconds: Optional[Union[int, str]],
    duration_seconds: Optional[Union[int, str]],
) -> str:
    watched_minutes = _seconds_to_minutes(watched_seconds) or 0
    duration_minutes = _seconds_to_minutes(duration_seconds)
    if duration_minutes and duration_minutes > 0:
        watched_minutes = min(watched_minutes, duration_minutes)
        return f"{watched_minutes}/{duration_minutes} Min"
    if watched_minutes > 0:
        return f"{watched_minutes} Min watched"
    return "0 Min watched"


def _prepare_video_payload(video: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(video)
    prepared.setdefault("total_watch_time_seconds", 0)
    prepared.setdefault("total_watch_count", 0)
    prepared.setdefault("total_likes", 0)
    prepared.setdefault("total_comments", 0)
    prepared.setdefault("total_subscribers", 0)
    prepared.setdefault("user_liked", False)
    prepared.setdefault("user_subscribed", False)
    prepared.setdefault("user_watch_duration_seconds", 0)
    prepared.setdefault("user_last_watched_at", None)
    prepared["thumbnail_url"] = _resolve_media_url(prepared.get("thumbnail_url"))
    prepared["video_url"] = _resolve_media_url(prepared.get("video_url"))
    return prepared

def _is_legacy_video_url(value: Optional[str]) -> bool:
    if not value:
        return False

    text = str(value).strip()
    if not text:
        return False

    normalized = text.replace("\\", "/")
    canonical = normalized.lstrip("/").lower()
    if canonical in LEGACY_SAMPLE_VIDEO_URL_SET:
        return True

    filename = Path(value).name.lower()
    if filename in LEGACY_SAMPLE_FILENAMES:
        return True

    return False

async def upload_static_video(
    *,
    file: UploadFile,
    title: str,
    subject: Optional[str],
    description: Optional[str],
    std: Optional[str],
    current_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    saved = await save_uploaded_file(
        file,
        VIDEOS_SUBDIR,
        allowed_extensions={".mp4", ".mov", ".mkv", ".webm"},
        allowed_types={"video/mp4", "video/x-m4v", "video/quicktime", "video/webm", "video/x-matroska"},
        max_size=500 * 1024 * 1024,
    )

    video_record = student_portal_video_repository.create_video(
        admin_id=current_context["admin_id"],
        std=std,
        subject=subject,
        title=title,
        description=description,
        chapter_name=None,
        duration_seconds=None,
        video_url=saved["file_path"],
        thumbnail_url=None,
    )

    return _prepare_video_payload(video_record)


def create_external_video(
    *,
    title: str,
    subject: Optional[str],
    description: Optional[str],
    std: Optional[str],
    video_url: str,
    thumbnail_url: Optional[str],
    current_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    video_record = student_portal_video_repository.create_video(
        admin_id=current_context["admin_id"],
        std=std,
        subject=subject,
        title=title,
        description=description,
        chapter_name=None,
        duration_seconds=None,
        video_url=video_url,
        thumbnail_url=thumbnail_url,
    )
    return _prepare_video_payload(video_record)


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _prepare_chat_message(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return record

    prepared = dict(record)
    attachment_path = prepared.pop("attachment_path", None)
    prepared["attachment_url"] = get_file_url(attachment_path) if attachment_path else None
    created_at = prepared.get("created_at")
    if isinstance(created_at, datetime):
        prepared["created_at"] = created_at.isoformat()
    return prepared


def _require_enrollment(enrollment_number: str) -> str:
    normalized = _sanitize(enrollment_number)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enrollment number is required",
        )
    return normalized


def signup_student(payload: StudentSignupRequest) -> None:
    enrollment_number = _require_enrollment(payload.enrollment_number)
    existing_account = student_portal_repository.get_student_account_by_enrollment(enrollment_number)
    if existing_account is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this enrollment number already exists",
        )

    student_portal_repository.create_student_account(
        enrollment_number=enrollment_number,
        password_hash=hash_password(payload.password),
    )


def _bootstrap_account_from_roster(enrollment_number: str) -> Optional[Dict[str, Any]]:
    roster_entry = student_portal_repository.get_roster_entry(enrollment_number)
    auto_password = roster_entry.get("auto_password") if roster_entry else None
    if not auto_password:
        return None

    return student_portal_repository.upsert_student_account(
        enrollment_number=enrollment_number,
        password_hash=hash_password(auto_password),
    )


def authenticate_student(payload: StudentLoginRequest) -> Dict[str, object]:
    enrollment_number = _require_enrollment(payload.enrollment_number)
    
    # Check if student exists in roster (not deleted)
    roster_entry = student_portal_repository.get_roster_entry(enrollment_number)
    if roster_entry is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid enrollment number or password",
        )
    
    account = student_portal_repository.get_student_account_by_enrollment(enrollment_number)
    if account is None:
        account = _bootstrap_account_from_roster(enrollment_number)
    if account is None or not verify_password(payload.password, account["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid enrollment number or password",
        )

    student_portal_repository.update_student_last_login(account["id"], datetime.utcnow())
    profile = student_portal_repository.get_student_profile_by_enrollment(enrollment_number)
    return {
        "account": account,
        "enrollment_number": enrollment_number,
        "profile_complete": profile is not None,
    }


def logout_student(enrollment_number: str) -> None:
    _require_enrollment(enrollment_number)


def change_student_password(*, enrollment_number: str, current_password: str, new_password: str) -> None:
    normalized = _require_enrollment(enrollment_number)
    account = student_portal_repository.get_student_account_by_enrollment(normalized)
    if account is None:
        account = _bootstrap_account_from_roster(normalized)

    if account is None or not verify_password(current_password, account["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    trimmed_new_password = new_password.strip()
    if not trimmed_new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New password cannot be empty")

    password_hash = hash_password(trimmed_new_password)
    updated = student_portal_repository.update_student_password(normalized, password_hash)
    if updated is None:
        updated = student_portal_repository.upsert_student_account(
            enrollment_number=normalized,
            password_hash=password_hash,
        )

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password",
        )

    student_portal_repository.update_roster_auto_password(normalized, trimmed_new_password)


def get_profile_status(enrollment_number: str) -> Dict[str, object]:
    normalized = _require_enrollment(enrollment_number)
    profile = student_portal_repository.get_student_profile_by_enrollment(normalized)
    roster_entry = student_portal_repository.get_roster_entry(normalized)

    prefill = None
    if roster_entry:
        middle_name= (roster_entry.get("middle_name") or "").strip()
        if not middle_name and profile is not None:
            middle_name = (profile.get("middle_name") or "").strip()
        prefill = {
            "first_name": roster_entry.get("first_name", ""),
            "middle_name": middle_name,
            "last_name": roster_entry.get("last_name") or "",
            "class_stream": roster_entry.get("std", ""),
            "division": roster_entry.get("division", ""),
            "enrollment_number": roster_entry.get("enrollment_number", normalized),
        }

    return {
        "profile_complete": profile is not None,
        "prefill": prefill,
    }


def upsert_student_profile(
    *,
    first_name: str,
    middle_name: Optional[str],
    last_name:Optional[str],
    class_stream: str,
    enrollment_number: str,
    division: Optional[str],
    class_head: Optional[str],
    mobile_number: Optional[str],
    parents_number: Optional[str],
    email: Optional[str],
    photo_path: Optional[str],
) -> StudentProfileResponse:
    normalized_enrollment = _require_enrollment(enrollment_number)
    profile = student_portal_repository.get_student_profile_by_enrollment(normalized_enrollment)

    payload = {
        "first_name": first_name.strip(),
        "middle_name": _sanitize(middle_name),
        "last_name": _sanitize(last_name),
        "class_stream": class_stream.strip(),
        "division": _sanitize(division),
        "class_head": _sanitize(class_head),
        "enrollment_number": normalized_enrollment,
        "mobile_number": _sanitize(mobile_number),
        "parents_number": _sanitize(parents_number),
        "email": _sanitize(email),
        "photo_path": photo_path,
    }

    if profile is None:
        created = student_portal_repository.create_student_profile(**payload)
        return StudentProfileResponse(**created)

    updates = payload.copy()
    updates.pop("enrollment_number")

    if photo_path is None:
        updates.pop("photo_path")
    elif profile.get("photo_path") and profile["photo_path"] != photo_path:
        delete_file(profile["photo_path"])

    updated = student_portal_repository.update_student_profile(profile["id"], **updates)
    data = updated or student_portal_repository.get_student_profile_by_id(profile["id"])
    if not data:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update profile")
    return StudentProfileResponse(**data)


def get_student_profile(enrollment_number: str) -> StudentProfileResponse:
    normalized = _require_enrollment(enrollment_number)
    profile = student_portal_repository.get_student_profile_by_enrollment(normalized)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student profile not found")
    return StudentProfileResponse(**profile)


def get_roster_context(
    enrollment_number: str,
    *,
    admin_id: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    context = student_portal_repository.get_student_roster_context(
        enrollment_number,
        admin_id=admin_id,
    )

    if context is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student is not in roster")

    profile_std = (context.get("profile_class_stream") or "").strip()
    roster_std = (context.get("std") or "").strip()
    normalized_std = roster_std or profile_std

    division = context.get("profile_division") or context.get("division")
    first_name = context.get("profile_first_name") or context.get("roster_first_name") or ""
    last_name = context.get("profile_last_name") or context.get("roster_last_name") or ""

    return {
        "admin_id": context["admin_id"],
        "enrollment_number": context["enrollment_number"],
        # "std": context["std"],
        "std": normalized_std or None,
        "division": division,
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "photo_path": context.get("photo_path"),
        "member_id": context.get("assigned_member_id"),
    }


def ensure_same_classmate(current: Dict[str, Optional[str]], peer_enrollment: str) -> Dict[str, Optional[str]]:
    peer_context = get_roster_context(peer_enrollment)
    if peer_context["admin_id"] != current["admin_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Students belong to different schools")
    if peer_context["std"] != current["std"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Students belong to different standards")
    if (peer_context.get("division") or "") != (current.get("division") or ""):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Students belong to different divisions")
    return peer_context


def list_chat_peers(current_context: Dict[str, Optional[str]]) -> List[Dict[str, Optional[str]]]:
    classmates = student_portal_repository.list_classmates(
        admin_id=current_context["admin_id"],
        std=current_context["std"],
        division=current_context.get("division"),
        exclude_enrollment=current_context["enrollment_number"],
    )

    enrollments = [peer["enrollment_number"] for peer in classmates]
    latest_messages = student_portal_repository.fetch_latest_peer_messages(
        admin_id=current_context["admin_id"],
        current_enrollment=current_context["enrollment_number"],
        peer_enrollments=enrollments,
    )

    def _display_name(peer: Dict[str, Optional[str]]) -> str:
        first = peer.get("profile_first_name") or peer.get("roster_first_name") or ""
        last = peer.get("profile_last_name") or peer.get("roster_last_name") or ""
        full = " ".join(part for part in (first.strip(), last.strip()) if part)
        return full or peer["enrollment_number"]

    peers_payload: List[Dict[str, Optional[str]]] = []
    for peer in classmates:
        latest_raw = latest_messages.get(peer["enrollment_number"]) or {}
        latest = _prepare_chat_message(latest_raw) or {}
        latest_message_preview = latest.get("message")
        if not latest_message_preview and latest.get("attachment_name"):
            latest_message_preview = latest["attachment_name"]
        peers_payload.append(
            {
                "enrollment_number": peer["enrollment_number"],
                "name": _display_name(peer),
                "std": peer["std"],
                "division": peer.get("division"),
                "photo_path": peer.get("photo_path"),
                "last_message": latest_message_preview,
                "last_message_at": latest.get("created_at"),
                "last_message_sender": latest.get("sender_enrollment"),
            }
        )

    return peers_payload


def list_chat_messages(
    *,
    current_context: Dict[str, Optional[str]],
    peer_context: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    records = student_portal_repository.fetch_chat_messages(
        admin_id=current_context["admin_id"],
        enrollment_a=current_context["enrollment_number"],
        enrollment_b=peer_context["enrollment_number"],
    )
    return [_prepare_chat_message(record) for record in records]


def send_chat_message(
    *,
    payload: SendChatMessageRequest,
    current_context: Dict[str, Optional[str]],
    peer_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    trimmed_message = _sanitize(payload.message)
    if not trimmed_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    record = student_portal_repository.insert_chat_message(
        admin_id=current_context["admin_id"],
        sender_enrollment=current_context["enrollment_number"],
        receiver_enrollment=peer_context["enrollment_number"],
        message=trimmed_message,
    )
    return _prepare_chat_message(record)

def share_video_to_chat(
    *,
    current_context: Dict[str, Optional[str]],
    peer_context: Dict[str, Optional[str]],
    video_id: int,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    video = student_portal_video_repository.get_video_with_engagement(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
        video_id=video_id,
    )
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    prepared = _prepare_video_payload(video)
    title = prepared.get("title") or f"Video {video_id}"
    video_url = prepared.get("video_url")
    parts = [f"Video: {title}"]
    if video_url:
        parts.append(str(video_url))
    trimmed = _sanitize(message)
    if trimmed:
        parts.append(trimmed)

    payload = SendChatMessageRequest(peer_enrollment=peer_context["enrollment_number"], message="\n".join(parts))
    return send_chat_message(payload=payload, current_context=current_context, peer_context=peer_context)

async def send_chat_attachment(
    *,
    file: UploadFile,
    message: Optional[str],
    current_context: Dict[str, Optional[str]],
    peer_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    trimmed_message = _sanitize(message)

    saved_file = await save_uploaded_file(
        file,
        CHAT_ATTACHMENT_SUBDIR,
    )

    attachment_name = file.filename or saved_file.get("saved_filename")

    record = student_portal_repository.insert_chat_message(
        admin_id=current_context["admin_id"],
        sender_enrollment=current_context["enrollment_number"],
        receiver_enrollment=peer_context["enrollment_number"],
        message=trimmed_message,
        attachment_path=saved_file.get("file_path"),
        attachment_name=attachment_name,
        attachment_mime_type=file.content_type,
        attachment_size=saved_file.get("file_size"),
    )

    return _prepare_chat_message(record)


def list_books_for_student(
    current_context: Dict[str, Optional[str]],
    *,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    admin_id = current_context["admin_id"]
    std = current_context.get("std")

    materials = chapter_material_repository.list_materials(
        admin_id,
        std=std,
        subject=subject,
    )

    books: List[Dict[str, Any]] = []
    for material in materials:
        title_parts = [material.get("subject") or "Chapter", material.get("chapter_number")]
        title = " - ".join(part for part in title_parts if part)
        created_at = material.get("created_at")

        books.append(
            {
                "id": material.get("id"),
                "title": title or material.get("file_name") or "Untitled Material",
                "subject": material.get("subject"),
                "board": material.get("board"),
                "std": material.get("std"),
                "chapter_number": material.get("chapter_number"),
                "file_name": material.get("file_name"),
                "file_url": get_file_url(material.get("file_path")) if material.get("file_path") else None,
                "file_size": material.get("file_size"),
                "uploaded_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
                "chapters": 1,
            }
        )

    return books


def ensure_dashboard_videos(current_context: Dict[str, Optional[str]]) -> None:
    admin_id = current_context["admin_id"]
    std = current_context.get("std")
    if LEGACY_SAMPLE_VIDEO_URLS:
        student_portal_video_repository.delete_videos_by_identifiers(
            admin_id=admin_id,
            canonical_urls=LEGACY_SAMPLE_VIDEO_URLS,
            filenames=LEGACY_SAMPLE_FILENAMES,
        )
    if not DEFAULT_VIDEO_SAMPLES:
        return
    student_portal_video_repository.ensure_sample_videos(admin_id, std, DEFAULT_VIDEO_SAMPLES)


def list_dashboard_videos(
    current_context: Dict[str, Optional[str]],
    *,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    ensure_dashboard_videos(current_context)
    videos = student_portal_video_repository.list_videos_for_student(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
        limit=limit,
    )
    # return [_prepare_video_payload(video) for video in videos]
    filtered_videos = [
        video
        for video in videos
        if not _is_legacy_video_url(video.get("video_url"))
    ]
    return [_prepare_video_payload(video) for video in filtered_videos]

def get_video_detail(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
) -> Dict[str, Any]:
    video = student_portal_video_repository.get_video_with_engagement(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
        video_id=video_id,
    )
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    related_videos = student_portal_video_repository.list_related_videos(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        subject=video.get("subject"),
        enrollment_number=current_context["enrollment_number"],
        exclude_video_id=video_id,
        limit=6,
    )
    payload = _prepare_video_payload(video)
    payload["related_videos"] = [_prepare_video_payload(item) for item in related_videos]
    return payload


def list_video_comments(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
) -> List[Dict[str, Any]]:
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != current_context["admin_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return student_portal_video_repository.list_comments(video_id)

def list_all_video_comments_for_admin(*, admin_id: int) -> List[Dict[str, Any]]:
    return student_portal_video_repository.list_comments_for_admin(admin_id)


def list_video_comments_for_admin(
    *,
    admin_id: int,
    video_id: int,
) -> List[Dict[str, Any]]:
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != admin_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return student_portal_video_repository.list_comments(video_id)

def delete_video_comment_for_admin(
    *,
    admin_id: int,
    comment_id: int,
) -> Dict[str, Any]:
    record = student_portal_video_repository.delete_comment(
        admin_id=admin_id,
        comment_id=comment_id,
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comment not found")
    return record


def delete_video_comments_for_admin_by_enrollment(
    *,
    admin_id: int,
    enrollment_number: str,
) -> List[Dict[str, Any]]:
    normalized = _sanitize(enrollment_number)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enrollment number is required",
        )

    records = student_portal_video_repository.delete_comments_by_enrollment(
        admin_id=admin_id,
        enrollment_number=normalized,
    )

    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No comments found for enrollment")

    return records
    
def add_video_comment(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
    enrollment_number: str,
    comment: str,
) -> Dict[str, Any]:
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != current_context["admin_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    roster_entry = student_portal_repository.get_student_roster_context(
        enrollment_number,
        admin_id=video.get("admin_id"),
    )
    if roster_entry is None or roster_entry.get("assigned_member_id") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student roster assignment not found for this admin",
        )
    resolved_member_id = roster_entry["assigned_member_id"]
    record = student_portal_video_repository.add_comment(
        video_id=video_id,
        admin_id=video["admin_id"],
        member_id=resolved_member_id,
        enrollment_number=enrollment_number,
        comment=comment,
    )
    return record


def set_video_like(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
    enrollment_number: str,
    liked: bool,
) -> Dict[str, bool]:
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != current_context["admin_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return student_portal_video_repository.set_like_status(
        video_id=video_id,
        enrollment_number=enrollment_number,
        liked=liked,
    )


def set_video_subscription(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
    enrollment_number: str,
    subscribed: bool,
) -> Dict[str, bool]:
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != current_context["admin_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return student_portal_video_repository.set_subscription_status(
        video_id=video_id,
        enrollment_number=enrollment_number,
        subscribed=subscribed,
    )


def record_video_watch(
    *,
    current_context: Dict[str, Optional[str]],
    video_id: int,
    enrollment_number: str,
    watch_seconds: int,
    duration_seconds: Optional[int] = None,
) -> None:
    if watch_seconds <= 0:
        return
    video = student_portal_video_repository.get_video(video_id)
    if video is None or video.get("admin_id") != current_context["admin_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    if duration_seconds and duration_seconds > 0:
        student_portal_video_repository.set_video_duration_seconds(
            video_id=video_id,
            duration_seconds=duration_seconds,
        )
    student_portal_video_repository.record_watch_event(
        video_id=video_id,
        enrollment_number=enrollment_number,
        watch_seconds=watch_seconds,
    )


def list_watched_videos(
    *,
    current_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    records = student_portal_video_repository.list_watched_videos(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
    )
    summary = student_portal_video_repository.get_watch_summary(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
    )
    prepared_records = [
        _prepare_video_payload(record)
        for record in records
        if (record.get("watch_duration_seconds") or 0) > 0
        or (record.get("user_watch_duration_seconds") or 0) > 0
    ]
    date_fields = ("created_at", "updated_at", "last_watched_at", "user_last_watched_at")
    for record in prepared_records:
        for field in date_fields:
            record[field] = _format_date_ddmmyyyy(record.get(field))
    return {
        "summary": summary,
        "videos": prepared_records,
    }

def get_lecture_watch_ratio(
    *,
    current_context: Dict[str, Optional[str]],
    std_override: Optional[str] = None,
) -> Dict[str, Any]:
    std_value = std_override or current_context.get("std")
    totals = student_portal_video_repository.get_video_totals_for_std(
        admin_id=current_context["admin_id"],
        std=std_value,
    )
    summary = student_portal_video_repository.get_watch_summary(
        admin_id=current_context["admin_id"],
        std=std_value,
        enrollment_number=current_context["enrollment_number"],
    )
    total_lectures = totals.get("total_videos", 0)
    watched_lectures = summary.get("watched_videos", 0)
    completion_fraction = 0.0
    if total_lectures > 0:
        completion_fraction = min(watched_lectures / total_lectures, 1.0)
    total_duration_seconds = totals.get("total_duration_seconds", 0) or 0
    watched_seconds = summary.get("total_watch_seconds", 0) or 0
    completed_lectures = summary.get("completed_videos", 0) or 0
    watched_progress_text = _format_progress_text(watched_seconds, total_duration_seconds)
    return {
        "summary": {
            "total_lectures": total_lectures,
            "completion": f"{watched_lectures}/{total_lectures}" if total_lectures else "0/0",
            "completion_ratio": round(completion_fraction * 100, 2),
            "watched_duration_text": watched_progress_text,
        }
    }
    
def list_watched_lecture_cards(
    *,
    current_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    records = student_portal_video_repository.list_watched_videos(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
    )
    lectures: List[Dict[str, Any]] = []
    for record in records:
        watched_seconds = record.get("user_watch_duration_seconds") or record.get("watch_duration_seconds") or 0
        if watched_seconds <= 0:
            continue
        lectures.append(
            {
                "lecture_title": record.get("title"),
                "subject": record.get("subject"),
                "chapter": record.get("chapter_name"),
                "progress": _format_progress_text(watched_seconds, record.get("duration_seconds")),
                "watched_date": _format_date_ddmmyyyy(
                    record.get("user_last_watched_at") or record.get("last_watched_at")
                ),
                "summary": record.get("description"),
            }
        )
    return {"lectures": lectures}


def list_all_lectures(
    *,
    current_context: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    ensure_dashboard_videos(current_context)
    records = student_portal_video_repository.list_all_videos(
        admin_id=current_context["admin_id"],
    )
    lectures: List[Dict[str, Any]] = []
    for record in records:
        lectures.append(
            {
                "lecture_title": record.get("title"),
                "subject": record.get("subject"),
                "chapter": record.get("chapter_name"),
                "date": _format_date_ddmmyyyy(record.get("created_at")),
                "class": record.get("std") or current_context.get("std"),
            }
        )
    return {"lectures": lectures}


def list_saved_videos(
    *,
    current_context: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    records = student_portal_video_repository.list_subscribed_videos(
        admin_id=current_context["admin_id"],
        std=current_context.get("std"),
        enrollment_number=current_context["enrollment_number"],
    )
    return [_prepare_video_payload(record) for record in records]