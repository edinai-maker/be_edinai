"""Public lecture routes for serving generated lecture JSON files."""
from __future__ import annotations

from typing import Any, Dict, Optional, DefaultDict, Set

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.repository.lecture_repository import LectureRepository
from app.database import get_db
from app.models.chapter_material import LectureGen
from app.utils.dependencies import get_current_user

router = APIRouter(
    prefix="/lectures",
    tags=["public lectures"],
    responses={
        404: {"description": "Lecture JSON not found"},
        500: {"description": "Internal server error"},
    },
)


def get_repository(db: Session = Depends(get_db)) -> LectureRepository:
    """Provide repository for accessing stored lectures."""
    return LectureRepository(db)


# ---------------------------------------------------------------------------
# PUBLIC LIST + FILTER ENDPOINTS
# ---------------------------------------------------------------------------


@router.get(
    "/",
    summary="List generated lectures with optional filters",
)
async def list_public_lectures(
    std: Optional[str] = Query(default=None, description="Filter by class/standard"),
    subject: Optional[str] = Query(default=None, description="Filter by subject"),
    division: Optional[str] = Query(default=None, description="Filter by division"),
    language: Optional[str] = Query(default=None, description="Filter by lecture language"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
    repository: LectureRepository = Depends(get_repository),
    
) -> Dict[str, Any]:
    # Extract admin_id: for admins it's in "id", for members it's in "admin_id"
    if current_user.get("role") == "admin":
        admin_id = current_user.get("id")
    else:
        admin_id = current_user.get("admin_id")
    lectures = await repository.list_lectures(
        std=std,
        subject=subject,
        division=division,
        language=language,
        limit=limit,
        offset=offset,
        admin_id=admin_id,
    )
    if not lectures:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No lectures found for provided filters",
        )
    return {
        "status": True,
        "message": "Lectures fetched successfully",
        "data": {"lectures": lectures, "count": len(lectures)},
    }

@router.get(
    "/filters",
    summary="Fetch available class and subject filters for authenticated admin",
)
async def get_lecture_filters(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # Extract admin_id: for admins it's in "id", for members it's in "admin_id"
    if current_user.get("role") == "admin":
        admin_id = current_user.get("id")
    else:
        admin_id = current_user.get("admin_id")
    
    rows = (
        db.query(LectureGen.std, LectureGen.subject)
        .filter(
            LectureGen.std.isnot(None),
            LectureGen.subject.isnot(None),
            LectureGen.admin_id == admin_id
        )
        .distinct()
        .all()
    )

    class_map: DefaultDict[str, Set[str]] = DefaultDict(set)
    for std, subject in rows:
        std_value = (std or "").strip()
        subject_value = (subject or "").strip()
        if not std_value or not subject_value:
            continue
        class_map[std_value].add(subject_value)

    std_subject_filters = [
        {
            "std": std_value,
            "subject": sorted(subjects),
        }
        for std_value, subjects in sorted(class_map.items())
    ]

    return {
        "status": True,
        "message": "Lecture filters fetched successfully",
        "data": std_subject_filters,
    }


@router.get(
    "/public_lecture/filters",
    summary="Fetch available class and subject filters for authenticated admin",
)

async def get_public_lecture_filters(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # Extract admin_id: for admins it's in "id", for members it's in "admin_id"
    if current_user.get("role") == "admin":
        admin_id = current_user.get("id")
    else:
        admin_id = current_user.get("admin_id")
    rows = (
        db.query(LectureGen.std, LectureGen.subject)
        .filter(
            LectureGen.std.isnot(None),
            LectureGen.subject.isnot(None),
            LectureGen.admin_id == admin_id
        )
        .distinct()
        .all()
    )
   
    class_map: DefaultDict[str, Set[str]] = DefaultDict(set)
    for std, subject in rows:
        std_value = (std or "").strip()
        subject_value = (subject or "").strip()
        if not std_value or not subject_value:
            continue
        class_map[std_value].add(subject_value)

    std_subject_filters = [
        {
            "std": std_value,
            "subject": sorted(subjects),
        }
        for std_value, subjects in sorted(class_map.items())
    ]
    return {
        "status": True,
        "message": "Lecture filters fetched successfully",
        "data": std_subject_filters,
    }
@router.get(
    "/public_lecture/played",
    summary="List played public lectures for the authenticated admin",
)
async def list_played_public_lectures(
    current_user: Dict[str, Any] = Depends(get_current_user),
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    admin_id = current_user.get("admin_id")
    # lectures = await repository.list_played_lectures(admin_id=admin_id)
    # Extract admin_id: for admins it's in "id", for members it's in "admin_id"
    if current_user.get("role") == "admin":
        admin_id = current_user.get("id")
    else:
        admin_id = current_user.get("admin_id")
    
    lectures = await repository.list_played_lectures(admin_id=admin_id)

    return {
        "status": True,
        "message": "Played lectures fetched successfully",
        "data": {"lectures": lectures, "count": len(lectures)},
    }


@router.get(
    "/{std}/{subject}/{lecture_id}.json",
    summary="Serve lecture JSON via public URL",
)
async def get_public_lecture_json(
    std: str,
    subject: str,
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> JSONResponse:
    """Return the stored lecture JSON so the /lectures URL works in browsers."""

    try:
        lecture_data: Dict[str, Any] = await repository.get_lecture(lecture_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Lecture file not found",
                "lecture_id": lecture_id,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading lecture: {exc}",
        ) from exc

    # Remove internal-only metadata keys user doesn't want exposed
    metadata = lecture_data.get("metadata") or {}
    for key in [
        "topics_source_file",
        "language_label",
        "language_code",
        "topics_override",
        "source_material_ids",
    ]:
        metadata.pop(key, None)

    lecture_data["metadata"] = metadata
    lecture_data.pop("lecture_url", None)
    
    slides = lecture_data.get("slides") or []
    for slide in slides:
        if isinstance(slide, dict):
            slide.pop("audio_path", None)

    return JSONResponse(content=lecture_data)

