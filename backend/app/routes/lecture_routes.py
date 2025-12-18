"""
Lecture Routes - FastAPI API Endpoints
Handles HTTP requests for lecture operations
"""
import os
from dotenv import load_dotenv
import json
import copy
import mimetypes
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Query, File, UploadFile , Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.schemas.lecture_schema import (
    CreateLectureRequest,
    AskQuestionRequest,
    LectureResponse,
    LectureSummaryResponse,
    LectureShareRequest,
    LectureShareResponse,
    LectureShareDeleteResponse,
    SharedLectureSummary,
    AnswerResponse,
    ErrorResponse,
    GenerationStatus,
    LectureListResponse,
)
from app.repository.lecture_repository import LectureRepository
from app.repository import student_portal_video_repository
from app.services.lecture_generation_service import GroqService
from app.services.lecture_share_service import LectureShareService
from app.schemas.admin_schema import WorkType
from app.utils.dependencies import admin_or_lecture_member, get_current_user, member_required
from app.utils.file_handler import save_uploaded_file
from app.postgres import get_pg_cursor
from app.database import get_db
from app.config import get_settings
from app.utils.s3_file_handler import get_s3_service
load_dotenv()

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(
    prefix="/lectures",
    tags=["lectures"],
    responses={
        404: {"model": ErrorResponse, "description": "Lecture not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)


# ============================================================================
# DEPENDENCIES
# ============================================================================

# These should be configured based on your app settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_repository(db: Session = Depends(get_db)) -> LectureRepository:
    """Dependency to get repository instance."""
    return LectureRepository(db)

def get_groq_service() -> GroqService:
    """Dependency to get Groq service instance."""
    return GroqService(api_key=GROQ_API_KEY)

def get_share_service(db: Session = Depends(get_db)) -> LectureShareService:
    return LectureShareService(db)


# ============================================================================
# LECTURE CRUD ENDPOINTS
# ============================================================================

@router.post(
    "/create",
    response_model=LectureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new lecture from text",
    description="Generate a complete lecture with slides from source text using AI"
)
async def create_lecture(
    request: CreateLectureRequest,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> LectureResponse:
    """
    Create a new lecture from text content.
    
    - **text**: Source text content (minimum 50 characters)
    - **language**: English, Hindi, or Gujarati
    - **duration**: Requested duration in minutes (10-120)
    - **style**: Teaching style (default: storytelling)
    - **title**: Lecture title
    - **metadata**: Optional additional metadata
    """
    try:
        # Generate lecture content using AI
        lecture_data = await groq_service.generate_lecture_content(
            text=request.text,
            language=request.language,
            duration=request.duration,
            style=request.style,
        )
        
        slides = lecture_data.get("slides", [])
        if not slides:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate lecture slides"
            )
        
        # Build context from narrations
        context = "\n\n".join(
            slide.get("narration", "")
            for slide in slides
            if slide.get("narration")
        )
        
        # Save to repository
        record = await repository.create_lecture(
            title=request.title,
            language=request.language,
            style=request.style,
            duration=request.duration,
            slides=slides,
            context=context,
            text=request.text,
            metadata=request.metadata,
            fallback_used=lecture_data.get("fallback_used", False),
        )
        
        # Generate JSON file URL with class and subject
        lecture_id = record.get("lecture_id", "")
        metadata = request.metadata or {}
        
        # Get class and subject from metadata
        std = metadata.get("std") or metadata.get("class") or "general"
        subject = metadata.get("subject") or "lecture"
        
        # Create URL slug
        std_slug = std.replace(" ", "_").lower()
        subject_slug = subject.replace(" ", "_").lower()
        
        # JSON file URL format: /lectures/{class}/{subject}/{lecture_id}.json
        lecture_json_url = f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"
        
        # Add URL to record
        record["lecture_url"] = lecture_json_url
        
        # Print to terminal
        print(f"\n{'='*60}")
        print(f" LECTURE GENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Lecture ID: {lecture_id}")
        print(f"Class: {std}")
        print(f"Subject: {subject}")
        print(f"Title: {request.title}")
        print(f"Language: {request.language}")
        print(f"JSON URL: {lecture_json_url}")
        print(f"Full Path: https://yourdomain.com{lecture_json_url}")
        print(f"Total Slides: {len(slides)}")
        print(f"{'='*60}\n")
        
        return LectureResponse(**record)
        
    except Exception as e:
        print(f" Error creating lecture: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create lecture: {str(e)}"
        )

@router.post(
    "/{lecture_id}/share",
    response_model=LectureShareResponse,
    status_code=status.HTTP_200_OK,
    summary="Share a lecture with students in a class",
)
async def share_lecture(
    lecture_id: str,
    payload: LectureShareRequest,
    current_user: Dict[str, Any] = Depends(member_required(WorkType.LECTURE)),
    share_service: LectureShareService = Depends(get_share_service),
) -> LectureShareResponse:
    result = await share_service.share_lecture(
        lecture_id=lecture_id,
        payload=payload,
        shared_by=str(current_user.get("id")),
        admin_id=current_user.get("admin_id"),
    )
    return LectureShareResponse(**result)

@router.post(
    "/{lecture_id}/share-recording",
    status_code=status.HTTP_200_OK,
    summary="Upload a recorded lecture video and share it with students via the student portal",
)
async def share_lecture_recording(
    lecture_id: str,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(member_required(WorkType.LECTURE)),
    share_service: LectureShareService = Depends(get_share_service),
) -> Dict[str, Any]:
    """Upload a screen recording for a lecture and register it as a student portal video.

    The recording is stored on disk and a corresponding entry is created in
    the ``student_portal_videos`` table so that students of the lecture's
    standard (derived from metadata when available) can see it on their
    portal dashboard.
    """

    # Resolve lecture metadata and admin context from DB
    with get_pg_cursor() as cur:
        cur.execute(
            """
            SELECT
                admin_id,
                lecture_title,
                subject,
                std,
                lecture_data,
                cover_photo_url
            FROM lecture_gen
            WHERE lecture_uid = %(lecture_uid)s
            """,
            {"lecture_uid": lecture_id},
        )
        lecture_row = cur.fetchone()

    if not lecture_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found",
        )

    admin_id = lecture_row.get("admin_id")
    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to determine school context for lecture",
        )

    title = (lecture_row.get("lecture_title") or f"Lecture {lecture_id}").strip() or f"Lecture {lecture_id}"
    chapter_name = (lecture_row.get("chapter_title") or title)

    lecture_payload_raw = lecture_row.get("lecture_data") or {}
    if isinstance(lecture_payload_raw, str):
        try:
            lecture_payload = json.loads(lecture_payload_raw)
        except Exception:
            lecture_payload = {}
    else:
        lecture_payload = lecture_payload_raw if isinstance(lecture_payload_raw, dict) else {}

    lecture_record = copy.deepcopy(lecture_payload) if isinstance(lecture_payload, dict) else {}
    if not isinstance(lecture_record, dict):
        lecture_record = {}

    lecture_record.setdefault("lecture_id", lecture_id)
    metadata = lecture_record.get("metadata")
    if metadata is None:
        metadata = {}
        lecture_record["metadata"] = metadata

    if lecture_row.get("subject"):
        lecture_record.setdefault("subject", lecture_row.get("subject"))
        metadata.setdefault("subject", lecture_row.get("subject"))
    if lecture_row.get("std"):
        metadata.setdefault("std", lecture_row.get("std"))
    if lecture_row.get("sem"):
        metadata.setdefault("sem", lecture_row.get("sem"))
    if lecture_row.get("board"):
        metadata.setdefault("board", lecture_row.get("board"))

    cover_photo_url_value = lecture_row.get("cover_photo_url")
    if isinstance(cover_photo_url_value, str):
        cover_photo_url_value = cover_photo_url_value.strip() or None
    else:
        cover_photo_url_value = None

    resolved_subject = share_service._resolve_subject(
        request_subject=None,
        row=lecture_row,
        record=lecture_record,
    )
    resolved_std = share_service._resolve_std(
        request_std="",
        row_std=lecture_row.get("std"),
        record=lecture_record,
    )

    subject_candidate = resolved_subject or lecture_row.get("subject") or metadata.get("subject")
    subject = subject_candidate.strip() if isinstance(subject_candidate, str) and subject_candidate.strip() else None

    def _extract_display_std(container: Dict[str, Any]) -> Optional[str]:
        candidate_keys = (
            "std_name",
            "standard_name",
            "standard",
            "class_name",
            "class_title",
            "class_label",
            "class",
            "grade_name",
            "grade",
            "grade_title",
            "std_title",
            "display_std",
        )
        for key in candidate_keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in container.values():
            if isinstance(value, dict):
                nested = _extract_display_std(value)
                if nested:
                    return nested
        return None

    std_candidate = _extract_display_std(metadata) or _extract_display_std(lecture_record)
    if not std_candidate and isinstance(resolved_std, str) and resolved_std.strip():
        std_candidate = resolved_std.strip()
    if not std_candidate:
        std_candidate = lecture_row.get("std") if isinstance(lecture_row.get("std"), str) else None
    std_value = std_candidate.strip() if isinstance(std_candidate, str) and std_candidate.strip() else None

    duration_seconds: Optional[int] = None
    duration_value = (
        lecture_record.get("estimated_duration")
        if isinstance(lecture_record, dict)
        else None
    ) or (
        lecture_record.get("requested_duration")
        if isinstance(lecture_record, dict)
        else None
    )

    if duration_value is not None:
        try:
            duration_seconds = int(float(duration_value) * 60)
        except (ValueError, TypeError):
            try:
                duration_seconds = int(duration_value)
            except (ValueError, TypeError):
                duration_seconds = None

    thumbnail_url: Optional[str] = None
    thumbnail_candidates: List[Optional[str]] = []

    if cover_photo_url_value:
        thumbnail_candidates.append(cover_photo_url_value)

    if isinstance(metadata, dict):
        thumbnail_candidates.extend(
            [
                metadata.get("thumbnail_url"),
                metadata.get("thumbnail"),
                metadata.get("cover_image"),
                metadata.get("cover"),
                metadata.get("poster"),
            ]
        )

    if isinstance(lecture_record, dict):
        thumbnail_candidates.extend(
            [
                lecture_record.get("thumbnail_url"),
                lecture_record.get("thumbnail"),
                lecture_record.get("cover_image"),
            ]
        )

        slides = lecture_record.get("slides")
        if isinstance(slides, list):
            for slide in slides:
                if not isinstance(slide, dict):
                    continue
                image_candidate = (
                    slide.get("image")
                    or slide.get("image_url")
                    or slide.get("thumbnail")
                    or slide.get("thumbnail_url")
                )
                if image_candidate:
                    thumbnail_candidates.append(image_candidate)
                    break

    for candidate in thumbnail_candidates:
        if isinstance(candidate, str) and candidate.strip():
            thumbnail_url = candidate.strip()
            break

    settings = get_settings()
    if not thumbnail_url:
        thumbnail_url = (
            getattr(settings, "default_lecture_thumbnail", None)
            or settings.dict().get("default_lecture_thumbnail")
            or "/static/images/lecture-placeholder.png"
        )

    # Persist uploaded recording to local storage
    saved = await save_uploaded_file(
        file,
        "videos",
        allowed_extensions={".mp4", ".mov", ".mkv", ".webm"},
        allowed_types={
            "video/mp4",
            "video/x-m4v",
            "video/quicktime",
            "video/webm",
            "video/x-matroska",
            "application/octet-stream",
        },
        max_size=5 * 1024 * 1024 * 1024,
    )

    video_url_value = saved["file_path"]
    thumbnail_url_value = thumbnail_url

    s3_service = None
    if settings.s3_enabled and settings.aws_s3_bucket_name:
        try:
            s3_service = get_s3_service(settings)
        except Exception as exc:  # pragma: no cover - S3 init
            logger.warning("Unable to initialize S3 service for lecture recordings: %s", exc)
            s3_service = None

    if s3_service:
        abs_video_path = os.path.abspath(saved["file_path"])
        if os.path.exists(abs_video_path):
            try:
                video_content_type = (
                    file.content_type
                    or mimetypes.guess_type(abs_video_path)[0]
                    or "video/mp4"
                )
                upload_result = s3_service.upload_file_from_path(
                    file_path=abs_video_path,
                    folder=f"lectures/{admin_id}/{lecture_id}",
                    content_type=video_content_type,
                    public=True,
                )
                video_url_value = upload_result.get("s3_url", video_url_value)
            except Exception as exc:  # pragma: no cover - S3 upload
                logger.warning("Failed to upload lecture recording to S3: %s", exc)

        if thumbnail_url and not str(thumbnail_url).lower().startswith(("http://", "https://", "s3://")):
            candidate_path = thumbnail_url
            if candidate_path.startswith("./"):
                candidate_path = candidate_path[2:]
            abs_thumb_path = os.path.abspath(candidate_path)
            if os.path.exists(abs_thumb_path):
                try:
                    thumb_content_type = mimetypes.guess_type(abs_thumb_path)[0] or "image/jpeg"
                    thumb_result = s3_service.upload_file_from_path(
                        file_path=abs_thumb_path,
                        folder=f"lectures/{admin_id}/{lecture_id}/thumbnails",
                        content_type=thumb_content_type,
                        public=True,
                    )
                    thumbnail_url_value = thumb_result.get("s3_url", thumbnail_url_value)
                except Exception as exc:  # pragma: no cover - S3 upload
                    logger.warning("Failed to upload lecture thumbnail to S3: %s", exc)

    video_record = student_portal_video_repository.create_video(
        admin_id=admin_id,
        std=std_value,
        subject=subject,
        title=title,
        description=f"Recorded lecture: {title}",
        chapter_name=chapter_name,
        duration_seconds=duration_seconds,
        video_url=video_url_value,
        thumbnail_url=thumbnail_url_value,
    )
    if isinstance(video_record, dict):
        video_record["cover_photo_url"] = cover_photo_url_value or thumbnail_url_value

    return {
        "status": True,
        "message": "Recording uploaded and shared to student portal successfully",
        "data": {
            "lecture_id": lecture_id,
            "std": std_value,
            "video": video_record,
        },
    }

@router.get(
    "/shared",
    response_model=List[SharedLectureSummary],
    summary="List all shared lectures",
    description="Retrieve lectures that have been shared with students",
)
async def list_shared_lectures(
    std: Optional[str] = Query(None, description="Filter by standard/class"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    current_user: Dict[str, Any] = Depends(member_required(WorkType.LECTURE)),
    share_service: LectureShareService = Depends(get_share_service),
) -> List[SharedLectureSummary]:
    admin_id = current_user.get("admin_id")
    records = await share_service.list_shared_lectures(
        admin_id=admin_id,
        std=std,
        subject=subject,
    )
    return [SharedLectureSummary(**record) for record in records]


@router.delete(
    "/{lecture_id}/shared",
    response_model=LectureShareDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete shared lecture records",
    description="Remove all share entries for a lecture and mark it as not shared",
)
async def delete_shared_lecture(
    lecture_id: str,
    current_user: Dict[str, Any] = Depends(member_required(WorkType.LECTURE)),
    share_service: LectureShareService = Depends(get_share_service),
) -> LectureShareDeleteResponse:
    admin_id = current_user.get("admin_id")
    result = await share_service.delete_shared_lecture(
        lecture_id=lecture_id,
        admin_id=admin_id,
    )
    return LectureShareDeleteResponse(**result)



@router.get(

    "/search",

    response_model=LectureListResponse,

    summary="Search lectures by title",

    description="Search lectures for the authenticated admin by title. Supports Gujarati titles and WhatsApp-style Roman queries via backend normalization.",

)

async def search_lectures(

    title: str = Query(..., description="Lecture title or keywords to search"),

    language: Optional[str] = Query(None, description="Filter by language"),

    std: Optional[str] = Query(None, description="Filter by standard/class"),


    subject: Optional[str] = Query(None, description="Filter by subject"),

    division: Optional[str] = Query(None, description="Filter by division/section"),

    limit: int = Query(100, ge=1, le=500, description="Maximum results"),

    offset: int = Query(0, ge=0, description="Pagination offset"),

    current_user: Dict[str, Any] = Depends(get_current_user),

    repository: LectureRepository = Depends(get_repository),

) -> LectureListResponse:

    try:

        # Extract admin_id: for admins it's in "id", for members it's in "admin_id"

        if current_user.get("role") == "admin":

            admin_id = current_user.get("id")

        else:

            admin_id = current_user.get("admin_id")



        lectures = await repository.search_lectures_by_title(

            query=title,

            language=language,

            std=std,

            subject=subject,

            division=division,

            limit=limit,

            offset=offset,
            admin_id=admin_id,

        ) 

        lecture_summaries = [LectureSummaryResponse(**lecture) for lecture in lectures]
        return LectureListResponse(
            status=True,
            message="Lecture search fetched successfully",
            data=lecture_summaries,
     )



    except Exception as e:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=f"Error searching lectures: {str(e)}",

        )


@router.get(
    "/{lecture_id}",
    response_model=LectureResponse,
    summary="Get lecture by ID",
    description="Retrieve complete lecture data including all slides"
)
async def get_lecture(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> LectureResponse:
    """
    Retrieve a lecture by its unique ID.
    
    - **lecture_id**: Unique lecture identifier
    """
    try:
        record = await repository.get_lecture(lecture_id)
        return LectureResponse(**record)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving lecture: {str(e)}"
        )


@router.get(
    "",
    response_model=LectureListResponse,
    summary="List lectures for the authenticated admin",
    description="Get a list of lectures generated within the current admin account with optional filtering"
)
async def list_lectures(
    language: Optional[str] = Query(None, description="Filter by language"),
    std: Optional[str] = Query(None, description="Filter by standard/class"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    division: Optional[str] = Query(None, description="Filter by division/section"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    repository: LectureRepository = Depends(get_repository),
) -> LectureListResponse:
    """
    List all lectures with pagination and filtering.
    - *language*: Optional filter by language (English, Hindi, Gujarati)
    - *std*: Optional filter by standard/class
    - *subject*: Optional filter by subject
    - *division*: Optional filter by division/section
    - *limit*: Maximum number of results (1-500)
    - *offset*: Number of results to skip for pagination
    """
    try:
        # Extract admin_id: for admins it's in "id", for members it's in "admin_id"
        if current_user.get("role") == "admin":
            admin_id = current_user.get("id")
        else:
            admin_id = current_user.get("admin_id")

        lectures = await repository.list_lectures(
            language=language,
            std=std,
            subject=subject,
            division=division,
            limit=limit,
            offset=offset,
            admin_id=admin_id,
        )
        lecture_summaries = [LectureSummaryResponse(**lecture) for lecture in lectures]
        return LectureListResponse(
            status=True,
            message="Lecture filters fetched successfully",
            data=lecture_summaries
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing lectures: {str(e)}"
        )


@router.delete(
    "/{lecture_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Delete a lecture",
    description="Permanently delete a lecture and all its associated files"
)
async def delete_lecture(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """
    Delete a lecture permanently.
    - **lecture_id**: Lecture to delete
    """
    try:
        deleted = await repository.delete_lecture(lecture_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lecture {lecture_id} not found"
            )

        return {
            "status": True,
            "message": "Lectures deleted successfully",
            "data": {
                "deleted_count": 1,
                "lectures": [
                    {"lecture_id": lecture_id},
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting lecture: {str(e)}"
        )
@router.delete(
    "",
    response_model=Dict[str, Any],
    summary="Delete lectures by class and subject",
    description="Remove all lectures for a given standard and subject (optionally division)."
)
async def bulk_delete_lectures(
    std: str = Query(..., description="Class/standard identifier"),
    subject: str = Query(..., description="Subject identifier"),
    division: Optional[str] = Query(None, description="Division/section identifier"),
    lecture_id: Optional[str] = Query(None, description="Specific lecture ID to delete within the filters"),
    current_user: Dict[str, Any] = Depends(member_required(WorkType.LECTURE)),
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    try:
        deleted = await repository.delete_lectures_by_metadata(
            std=std,
            subject=subject,
            division=division,
            lecture_id=lecture_id,
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No lectures found for provided filters",
            )
        return {
            "status": True,
            "message": "Lectures deleted successfully",
            "data": {
                "deleted_count": len(deleted),
                "lectures": deleted,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting lectures: {exc}",
        )
# ============================================================================
# QUESTION & ANSWER ENDPOINTS
# ============================================================================
@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question about a lecture",
    description="Get AI-powered answers to questions about lecture content"
)
async def ask_question(
    request: AskQuestionRequest,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> AnswerResponse:
    """
    Ask a question about a lecture or edit slide content.
    
    - **lecture_id**: Lecture to query
    - **question**: Question text or edit command
    - **answer_type**: Response format (text or json)
    - **is_edit_command**: Whether this is an edit command
    - **context_override**: Optional context override
    """
    try:
        # Get lecture for context
        record = await repository.get_lecture(request.lecture_id)
        
        context = request.context_override or record.get("context", "")
        language = record.get("language", "English")
        
        # Get answer from AI
        response = await groq_service.answer_question(
            question=request.question,
            context=context,
            language=language,
            answer_type=request.answer_type,
            is_edit_command=request.is_edit_command,
        )
        
        # Handle different response types
        if isinstance(response, dict):
            return AnswerResponse(**response)
        else:
            return AnswerResponse(answer=str(response))
            
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {request.lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )
# ============================================================================
# SLIDE MANAGEMENT ENDPOINTS
# ============================================================================
@router.get(
    "/{lecture_id}/slides/{slide_number}",
    summary="Get a specific slide",
    description="Retrieve details of a specific slide from a lecture"
)
async def get_slide(
    lecture_id: str,
    slide_number: int,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """
    Get a specific slide from a lecture.
    
    - **lecture_id**: Lecture containing the slide
    - **slide_number**: Slide number (1-indexed)
    """
    try:
        record = await repository.get_lecture(lecture_id)
        slides = record.get("slides", [])
        
        for slide in slides:
            if slide.get("number") == slide_number:
                return slide
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide {slide_number} not found in lecture {lecture_id}"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving slide: {str(e)}"
        )
@router.patch(
    "/{lecture_id}/slides/{slide_number}",
    summary="Update a slide",
    description="Update specific fields of a slide"
)
async def update_slide(
    lecture_id: str,
    slide_number: int,
    updates: Dict[str, Any],
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """
    Update a slide's content.
    
    - **lecture_id**: Lecture containing the slide
    - **slide_number**: Slide number to update
    - **updates**: Dictionary of fields to update
    """
    try:
        record = await repository.update_slide(
            lecture_id=lecture_id,
            slide_number=slide_number,
            slide_updates=updates,
        )
        
        # Return updated slide
        for slide in record.get("slides", []):
            if slide.get("number") == slide_number:
                return slide
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide {slide_number} not found"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating slide: {str(e)}"
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get(
    "/stats/summary",
    summary="Get lecture statistics",
    description="Get overall statistics about stored lectures"
)
async def get_stats(
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Get statistics about all lectures."""
    try:
        stats = await repository.get_lecture_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@router.get(
    "/{lecture_id}/source",
    summary="Get source text",
    description="Retrieve the original source text used to generate the lecture"
)
async def get_source_text(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, str]:
    """
    Get the original source text for a lecture.
    
    - **lecture_id**: Lecture ID
    """
    try:
        text = await repository.get_source_text(lecture_id)
        return {"lecture_id": lecture_id, "source_text": text}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source text not found for lecture {lecture_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving source text: {str(e)}"
        )


@router.get(
    "/{lecture_id}/play",
    summary="Prepare playback payload",
    description="Return narration script and slide segments for lecture playback"
)
async def get_playback_payload(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Return structured playback information for the requested lecture."""
    try:
        await repository.record_play(lecture_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error preparing playback payload: {exc}"
        ) from exc

    record = await repository.get_lecture(lecture_id)
    # slides = record.get("slides", []) or []
    # playback_segments: List[Dict[str, Any]] = []

    # for slide in slides:
    #     narration = slide.get("narration")
    #     if not narration:
    #         continue

    #     playback_segments.append(
    #         {
    #             "slide_number": slide.get("number"),
    #             "title": slide.get("title"),
    #             "narration": narration,
    #         }
    #     )

    # if not playback_segments:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Lecture does not contain narration to play"
    #     )

    # combined_script = "\n\n".join(
    #     f"Slide {segment.get('slide_number')}: {segment.get('title')}\n{segment.get('narration')}"
    #     for segment in playback_segments
    # )

    # return {
    #     "lecture_id": lecture_id,
    #     "title": record.get("title"),
    #     "language": record.get("language"),
    #     "metadata": record.get("metadata", {}),
    #     "playback": {
    #         "segments": playback_segments,
    #         "combined_script": combined_script,
    #     },
    #     "lecture_url": record.get("lecture_url"),

    requested_id = str(lecture_id)
    lecture_url = record.get("lecture_url")

    # Derive duration in minutes for playback payload using existing metadata
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




    if lecture_url:
        prefix, sep, filename = lecture_url.rpartition("/")
        if sep:
            name, dot, ext = filename.partition(".")
            if dot:
                lecture_url = f"{prefix}/{requested_id}.{ext}"
            else:
                lecture_url = f"{prefix}/{requested_id}"
    if not lecture_url:
        metadata = record.get("metadata") or {}
        std_value = (metadata.get("std") or metadata.get("class") or "general").strip().lower().replace(" ", "_")
        subject_value = (metadata.get("subject") or "lecture").strip().lower().replace(" ", "_")
        lecture_url = f"/lectures/{std_value}/{subject_value}/{requested_id}.json"

    return {
        "lecture_id": requested_id,
        "title": record.get("title"),
        "cover_photo_url": record.get("cover_photo_url"),
        "lecture_url": lecture_url,
        "duration": duration_minutes,
    }


@router.post(
    "/{lecture_id}/regenerate",
    response_model=LectureResponse,
    summary="Regenerate lecture content",
    description="Regenerate a lecture using the same source text but with new AI generation"
)
async def regenerate_lecture(
    lecture_id: str,
    language: Optional[str] = None,
    duration: Optional[int] = None,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> LectureResponse:
    """
    Regenerate a lecture with new content.
    
    - **lecture_id**: Lecture to regenerate
    - **language**: Optional new language (uses original if not provided)
    - **duration**: Optional new duration (uses original if not provided)
    """
    try:
        # Get original lecture
        original = await repository.get_lecture(lecture_id)
        source_text = await repository.get_source_text(lecture_id)
        
        # Use provided params or fall back to originals
        new_language = language or original.get("language", "English")
        new_duration = duration or original.get("requested_duration", 30)
        
        # Generate new content
        lecture_data = await groq_service.generate_lecture_content(
            text=source_text,
            language=new_language,
            duration=new_duration,
            style=original.get("style", "storytelling"),
        )
        
        slides = lecture_data.get("slides", [])
        context = "\n\n".join(
            slide.get("narration", "")
            for slide in slides
            if slide.get("narration")
        )
        
        # Update lecture
        updates = {
            "language": new_language,
            "requested_duration": new_duration,
            "estimated_duration": lecture_data.get("estimated_duration"),
            "slides": slides,
            "context": context,
            "total_slides": len(slides),
            "fallback_used": lecture_data.get("fallback_used", False),
        }
        
        record = await repository.update_lecture(lecture_id, updates)
        
        # Generate JSON URL again after regeneration
        metadata = original.get("metadata", {})
        std = metadata.get("std") or metadata.get("class") or "general"
        subject = metadata.get("subject") or "lecture"
        
        std_slug = std.replace(" ", "_").lower()
        subject_slug = subject.replace(" ", "_").lower()
        
        # JSON file URL format
        lecture_json_url = f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"
        
        record["lecture_url"] = lecture_json_url
        
        # Print to terminal
        print(f"\n{'='*60}")
        print(f"🔄 LECTURE REGENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Lecture ID: {lecture_id}")
        print(f"Class: {std}")
        print(f"Subject: {subject}")
        print(f"New Language: {new_language}")
        print(f"JSON URL: {lecture_json_url}")
        print(f"Full Path: https://yourdomain.com{lecture_json_url}")
        print(f"{'='*60}\n")
        
        return LectureResponse(**record)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error regenerating lecture: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the lecture service is running"
)
async def health_check(
    groq_service: GroqService = Depends(get_groq_service),
) -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "lecture_generation",
        "groq_configured": groq_service.configured,
    }