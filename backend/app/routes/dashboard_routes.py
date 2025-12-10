"""Dashboard routes for admin/member dashboards."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas import ResponseBase, WorkType
from ..services import dashboard_service
from ..utils.dependencies import member_required, onboarding_completed_required

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/admin", response_model=ResponseBase)
async def get_admin_dashboard(current_user: dict = Depends(onboarding_completed_required)):
    try:
        data = dashboard_service.get_admin_dashboard(current_user["id"])
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")
    return ResponseBase(status=True, message="Admin dashboard data retrieved", data=data)


@router.get("/chapter", response_model=ResponseBase)
async def get_chapter_dashboard(current_user: dict = Depends(member_required(WorkType.CHAPTER))):
    data = dashboard_service.get_member_dashboard(
        member_id=current_user["id"],
        admin_id=current_user["admin_id"],
        work_type="chapter",
    )
    return ResponseBase(status=True, message="Chapter dashboard data retrieved", data=data)


@router.get("/student", response_model=ResponseBase)
async def get_student_dashboard(current_user: dict = Depends(member_required(WorkType.STUDENT))):
    data = dashboard_service.get_member_dashboard(
        member_id=current_user["id"],
        admin_id=current_user["admin_id"],
        work_type="student",
    )
    return ResponseBase(status=True, message="Student dashboard data retrieved", data=data)


@router.get("/lecture", response_model=ResponseBase)
async def get_lecture_dashboard(current_user: dict = Depends(member_required(WorkType.LECTURE))):
    data = dashboard_service.get_member_dashboard(
        member_id=current_user["id"],
        admin_id=current_user["admin_id"],
        work_type="lecture",
    )
    return ResponseBase(status=True, message="Lecture dashboard data retrieved", data=data)

@router.get("/admin/lectures", response_model=ResponseBase)
async def get_admin_lecture_dashboard(current_user: dict = Depends(onboarding_completed_required)):
    data = dashboard_service.get_admin_lecture_dashboard(current_user["id"])
    return ResponseBase(status=True, message="Admin lecture dashboard data retrieved", data=data)

@router.get("/summary", response_model=ResponseBase)
async def get_dashboard_summary(current_user: dict = Depends(onboarding_completed_required)):
    data = dashboard_service.get_summary(current_user["id"])
    return ResponseBase(status=True, message="Dashboard summary retrieved", data=data)
