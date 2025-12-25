"""Routes for the student portal module."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ..config import settings
from ..schemas import WorkType
from ..schemas.response import ResponseBase
from ..schemas.student_portal_schema import (
    SendChatMessageRequest,
    StudentChangePasswordRequest,
    StudentLoginRequest,
    StudentLoginResponse,
    StudentProfileResponse,
    StudentSignupRequest,
    StudentVideoCommentRequest,
    StudentVideoLikeRequest,
    StudentVideoExternalRequest,
    StudentVideoUploadRequest,
    StudentVideoSubscribeRequest,
    StudentVideoShareRequest,
    StudentVideoWatchRequest,
)
from ..realtime.socket_server import broadcast_chat_message
from ..repository import student_portal_video_repository
from ..services import student_portal_service
from ..utils import s3_file_handler
from ..utils.dependencies import get_current_user
from ..utils.file_handler import save_uploaded_file

router = APIRouter(prefix="/school-portal", tags=["Student Portal"])
_student_security = HTTPBearer(auto_error=False)
_STUDENT_JWT_ALGORITHM = "HS256"


def _create_student_token(enrollment_number: str) -> str:
    expire = datetime.utcnow() + timedelta(days=settings.access_token_expire_days)
    payload = {
        "sub": "student",
        "enrollment_number": enrollment_number,
        "exp": expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_STUDENT_JWT_ALGORITHM)


def _get_current_student(
    credentials: HTTPAuthorizationCredentials = Depends(_student_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Student token required")

    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_STUDENT_JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid student token") from exc

    enrollment_number = payload.get("enrollment_number")
    if not enrollment_number:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid student token payload")
    return str(enrollment_number)


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "module": "student_portal"}


@router.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def student_signup(payload: StudentSignupRequest):
    student_portal_service.signup_student(payload)
    return {"status": True, "message": "Student account created successfully"}


@router.post("/auth/login", response_model=StudentLoginResponse)
async def student_login(payload: StudentLoginRequest) -> StudentLoginResponse:
    result = student_portal_service.authenticate_student(payload)
    token = _create_student_token(result["enrollment_number"])
    return StudentLoginResponse(
        status=True,
        message="Login successful",
        token=token,
        profile_complete=result["profile_complete"],
    )


@router.post("/auth/logout", response_model=ResponseBase)
async def student_logout(current_enrollment: str = Depends(_get_current_student)) -> ResponseBase:
    student_portal_service.logout_student(current_enrollment)
    return ResponseBase(status=True, message="Logout successful")


@router.post("/auth/change-password", response_model=ResponseBase)
async def student_change_password(
    payload: StudentChangePasswordRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    student_portal_service.change_student_password(
        enrollment_number=current_enrollment,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return ResponseBase(status=True, message="Password updated successfully", data={})


@router.get("/profile-status/{enrollment_number}")
async def student_profile_status(enrollment_number: str):
    status_payload = student_portal_service.get_profile_status(enrollment_number)
    return {
        "status": True,
        "profile_complete": status_payload["profile_complete"],
        "prefill": status_payload["prefill"],
    }


@router.post("/profile", response_model=StudentProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_or_update_student_profile(
    first_name: str = Form(...),
    middle_name: Optional[str] = Form(None),
    last_name: Optional[str] = Form(None),
    class_stream: str = Form(...),
    enrollment_number: str = Form(...),
    division: Optional[str] = Form(None),
    class_head: Optional[str] = Form(None),
    mobile_number: Optional[str] = Form(None),
    parents_number: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
):
    saved_photo_path: Optional[str] = None
    if photo is not None:
        try:
            if settings.s3_enabled:
                if not (settings.aws_access_key_id and settings.aws_secret_access_key and settings.aws_s3_bucket_name):
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="S3 is not configured",
                    )

                s3_service = s3_file_handler.get_s3_service(settings)
                upload_info = await s3_file_handler.upload_image_to_s3(
                    photo,
                    s3_service,
                    subfolder="student-profiles",
                )
                saved_photo_path = upload_info.get("s3_url")
            else:
                upload_info = await save_uploaded_file(photo, subfolder="student-profiles")
                saved_photo_path = upload_info["file_path"]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload profile photo: {exc}",
            ) from exc

    profile = student_portal_service.upsert_student_profile(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
        class_stream=class_stream,
        enrollment_number=enrollment_number,
        division=division,
        class_head=class_head,
        mobile_number=mobile_number,
        parents_number=parents_number,
        email=email,
        photo_path=saved_photo_path,
    )
    return profile


@router.get("/profile/{enrollment_number}", response_model=StudentProfileResponse)
async def get_student_profile(enrollment_number: str) -> StudentProfileResponse:
    return student_portal_service.get_student_profile(enrollment_number)


@router.get("/books", response_model=ResponseBase)
async def list_student_books(
    subject: Optional[str] = None,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    books = student_portal_service.list_books_for_student(context, subject=subject)
    return ResponseBase(status=True, message="Books fetched successfully", data={"books": books})


@router.get("/dashboard/videos", response_model=ResponseBase)
async def list_dashboard_videos(
    limit: int = 6,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    videos = student_portal_service.list_dashboard_videos(context, limit=max(1, min(limit, 12)))
    return ResponseBase(status=True, message="Dashboard videos fetched successfully", data={"videos": videos})


@router.get("/videos/saved", response_model=ResponseBase)
async def list_saved_videos(current_enrollment: str = Depends(_get_current_student)) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    videos = student_portal_service.list_saved_videos(current_context=context)
    return ResponseBase(status=True, message="Saved videos fetched successfully", data={"videos": videos})

@router.get("/lectures", response_model=ResponseBase)
async def list_all_lectures(
    std: Optional[str] = Query(None, min_length=1, max_length=16),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ResponseBase:
    if current_user["role"] == "admin":
        admin_id = int(current_user["id"])
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.STUDENT.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student management access required")
        admin_id = int(current_user["admin_id"])
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    videos = student_portal_video_repository.list_all_videos(
        admin_id=admin_id,
        std=std.strip() if std else None,
    )

    def _format_date(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                return value
        if isinstance(value, datetime):
            return value.date().isoformat()
        return str(value)

    lectures = [
        {
            "title": video.get("title"),
            "subject": video.get("subject"),
            "chapter": video.get("chapter_name"),
            "date": _format_date(video.get("created_at")),
            "class": video.get("std"),
        }
        for video in videos
    ]
    return ResponseBase(
        status=True,
        message="Lectures fetched successfully",
        data={"lectures": lectures},
    )

@router.get("/videos/{video_id}", response_model=ResponseBase)
async def get_video_detail(video_id: int, current_enrollment: str = Depends(_get_current_student)) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    video = student_portal_service.get_video_detail(current_context=context, video_id=video_id)
    comments = student_portal_service.list_video_comments(
        current_context=context,
        video_id=video_id,
    )
    return ResponseBase(
        status=True,
        message="Video fetched successfully",
        data={"video": video, "comments": comments},
    )


@router.get("/videos/{video_id}/comments", response_model=ResponseBase)
async def list_video_comments(
    video_id: int,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    comments = student_portal_service.list_video_comments(
        current_context=context,
        video_id=video_id,
    )
    return ResponseBase(status=True, message="Comments fetched successfully", data={"comments": comments})


@router.post("/videos/{video_id}/comments", response_model=ResponseBase)
async def add_video_comment(
    video_id: int,
    payload: StudentVideoCommentRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    record = student_portal_service.add_video_comment(
        current_context=context,
        video_id=video_id,
        enrollment_number=current_enrollment,
        comment=payload.comment,
    )
    return ResponseBase(status=True, message="Comment added successfully", data={"comment": record})


@router.post("/videos/{video_id}/like", response_model=ResponseBase)
async def set_video_like(
    video_id: int,
    payload: StudentVideoLikeRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    status_payload = student_portal_service.set_video_like(
        current_context=context,
        video_id=video_id,
        enrollment_number=current_enrollment,
        liked=payload.liked,
    )
    return ResponseBase(status=True, message="Video like status updated", data=status_payload)


@router.post("/videos/{video_id}/subscribe", response_model=ResponseBase)
async def set_video_subscription(
    video_id: int,
    payload: StudentVideoSubscribeRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    status_payload = student_portal_service.set_video_subscription(
        current_context=context,
        video_id=video_id,
        enrollment_number=current_enrollment,
        subscribed=payload.subscribed,
    )
    return ResponseBase(status=True, message="Video subscription status updated", data=status_payload)


@router.post("/videos/upload", response_model=ResponseBase)
async def upload_static_video(
    title: str = Form(...),
    subject: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    std: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    payload = StudentVideoUploadRequest(title=title, subject=subject, description=description, std=std)
    context = student_portal_service.get_roster_context(current_enrollment)
    video = await student_portal_service.upload_static_video(
        file=file,
        title=payload.title,
        subject=payload.subject,
        description=payload.description,
        std=payload.std,
        current_context=context,
    )
    return ResponseBase(status=True, message="Video uploaded successfully", data={"video": video})


@router.post("/videos/register", response_model=ResponseBase)
async def register_external_video(
    payload: StudentVideoExternalRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    video = student_portal_service.create_external_video(
        title=payload.title,
        subject=payload.subject,
        description=payload.description,
        std=payload.std,
        video_url=payload.video_url,
        thumbnail_url=payload.thumbnail_url,
        current_context=context,
    )
    return ResponseBase(status=True, message="Video registered successfully", data={"video": video})


@router.post("/videos/{video_id}/watch", response_model=ResponseBase)
async def record_video_watch(
    video_id: int,
    payload: StudentVideoWatchRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    student_portal_service.record_video_watch(
        current_context=context,
        video_id=video_id,
        enrollment_number=current_enrollment,
        watch_seconds=payload.watch_seconds,
        duration_seconds=payload.duration_seconds,
    )
    return ResponseBase(status=True, message="Watch progress saved", data={})

@router.post("/videos/{video_id}/share", response_model=ResponseBase)
async def share_video_to_chat(
    video_id: int,
    payload: StudentVideoShareRequest,
    current_enrollment: str = Depends(_get_current_student),
) -> ResponseBase:
    current_context = student_portal_service.get_roster_context(current_enrollment)
    peer_context = student_portal_service.ensure_same_classmate(
        current=current_context,
        peer_enrollment=payload.peer_enrollment,
    )

    record = student_portal_service.share_video_to_chat(
        current_context=current_context,
        peer_context=peer_context,
        video_id=video_id,
        message=payload.message,
    )

    messages = student_portal_service.list_chat_messages(
        current_context=current_context,
        peer_context=peer_context,
    )

    await broadcast_chat_message(
        admin_id=current_context["admin_id"],
        enrollments=[current_context["enrollment_number"], peer_context["enrollment_number"]],
        payload={
            "message": record,
            "participants": [current_context["enrollment_number"], peer_context["enrollment_number"]],
        },
    )

    return ResponseBase(
        status=True,
        message="Video shared successfully",
        data={"message": record, "messages": messages},
    )

@router.get("/watched-lectures", response_model=ResponseBase)
async def list_watched_lectures(current_enrollment: str = Depends(_get_current_student)) -> ResponseBase:
    context = student_portal_service.get_roster_context(current_enrollment)
    payload = student_portal_service.list_watched_videos(current_context=context)
    return ResponseBase(status=True, message="Watched lectures fetched successfully", data=payload)

@router.get("/watched-lectures/ratio", response_model=ResponseBase)
async def get_watched_lecture_ratio(
    enrollment_number: str = Query(..., min_length=3, max_length=32),
    std: Optional[str] = Query(None, min_length=1, max_length=32),
) -> ResponseBase:
    normalized = enrollment_number.strip()
    if not normalized:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Enrollment number required")
    context = student_portal_service.get_roster_context(normalized)
    payload = student_portal_service.get_lecture_watch_ratio(current_context=context, std_override=std)
    return ResponseBase(status=True, message="Lecture watch ratio fetched successfully", data=payload)

@router.get("/watched-lectures/cards", response_model=ResponseBase)
async def list_watched_lecture_cards(enrollment_number: str = Query(..., min_length=3, max_length=32)) -> ResponseBase:
    """Tokenless helper endpoint that returns watched-lecture summary for a given enrollment."""

    normalized = enrollment_number.strip()
    if not normalized:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Enrollment number required")

    context = student_portal_service.get_roster_context(normalized)
    payload = student_portal_service.list_watched_videos(current_context=context)
    return ResponseBase(status=True, message="Watched lectures fetched successfully", data=payload)

@router.get("/chat/peers", response_model=ResponseBase)
async def list_chat_peers(current_enrollment: str = Depends(_get_current_student)):
    context = student_portal_service.get_roster_context(current_enrollment)
    peers_payload = student_portal_service.list_chat_peers(context)
    return ResponseBase(
        status=True,
        message="Chat peers fetched successfully",
        data={"peers": peers_payload},
    )


@router.get("/chat/messages/{peer_enrollment}", response_model=ResponseBase)
async def list_chat_messages(
    peer_enrollment: str,
    current_enrollment: str = Depends(_get_current_student),
):
    current_context = student_portal_service.get_roster_context(current_enrollment)
    peer_context = student_portal_service.ensure_same_classmate(
        current=current_context, peer_enrollment=peer_enrollment
    )

    messages = student_portal_service.list_chat_messages(
        current_context=current_context,
        peer_context=peer_context,
    )

    return ResponseBase(
        status=True,
        message="Chat history fetched successfully",
        data={"messages": messages},
    )


@router.post("/chat/messages", response_model=ResponseBase)
async def send_chat_message(
    payload: SendChatMessageRequest = Body(...),
    current_enrollment: str = Depends(_get_current_student),
):
    current_context = student_portal_service.get_roster_context(current_enrollment)
    peer_context = student_portal_service.ensure_same_classmate(
        current=current_context, peer_enrollment=payload.peer_enrollment
    )

    record = student_portal_service.send_chat_message(
        payload=payload,
        current_context=current_context,
        peer_context=peer_context,
    )

    messages = student_portal_service.list_chat_messages(
        current_context=current_context,
        peer_context=peer_context,
    )

    await broadcast_chat_message(
        admin_id=current_context["admin_id"],
        enrollments=[current_context["enrollment_number"], peer_context["enrollment_number"]],
        payload={
            "message": record,
            "participants": [current_context["enrollment_number"], peer_context["enrollment_number"]],
        },
    )

    return ResponseBase(
        status=True,
        message="Message sent successfully",
        data={"message": record, "messages": messages},
    )


@router.post("/chat/messages/attachment", response_model=ResponseBase)
async def send_chat_attachment(
    peer_enrollment: str = Form(...),
    file: UploadFile = File(...),
    message: str | None = Form(None),
    current_enrollment: str = Depends(_get_current_student),
):
    current_context = student_portal_service.get_roster_context(current_enrollment)
    peer_context = student_portal_service.ensure_same_classmate(
        current=current_context, peer_enrollment=peer_enrollment
    )

    record = await student_portal_service.send_chat_attachment(
        file=file,
        message=message,
        current_context=current_context,
        peer_context=peer_context,
    )

    messages = student_portal_service.list_chat_messages(
        current_context=current_context,
        peer_context=peer_context,
    )

    await broadcast_chat_message(
        admin_id=current_context["admin_id"],
        enrollments=[current_context["enrollment_number"], peer_context["enrollment_number"]],
        payload={
            "message": record,
            "participants": [current_context["enrollment_number"], peer_context["enrollment_number"]],
        },
    )

    return ResponseBase(
        status=True,
        message="Attachment sent successfully",
        data={"message": record, "messages": messages},
    )


