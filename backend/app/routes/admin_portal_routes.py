from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from ..schemas import ChangePasswordRequest, MemberCreate, MemberUpdate, ResponseBase, WorkType
from ..services import admin_portal_service, auth_service, dashboard_service
from ..utils.dependencies import admin_required

router = APIRouter(prefix="/admin-portal", tags=["Admin Portal"])


def _parse_work_type(value: Optional[str]) -> Optional[WorkType]:
    if value is None:
        return None

    normalized = value.strip().lower()
    alias_map = {
        "chapter management": WorkType.CHAPTER.value,
        "chapter_management": WorkType.CHAPTER.value,
        "student management": WorkType.STUDENT.value,
        "student_management": WorkType.STUDENT.value,
        "lecture management": WorkType.LECTURE.value,
        "lecture_management": WorkType.LECTURE.value,
    }
    normalized = alias_map.get(normalized, normalized)
    try:
        return WorkType(normalized)
    except ValueError as exc:  # pragma: no cover - FastAPI handles response
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid work_type filter") from exc


def _resolve_admin_id(current_user: dict) -> int:
    return int(current_user["id"])


@router.get("/dashboard", response_model=ResponseBase)
async def get_admin_portal_dashboard(
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    try:
        admin_id = _resolve_admin_id(current_user)
        # Try to get dashboard data using the admin_id
        data = dashboard_service.get_admin_dashboard(admin_id)
    except ValueError:
        # If admin not found in dashboard repo, use the admin object from current_user
        # This handles portal admins that may not be in the legacy admins table
        admin_obj = current_user.get("user_obj")
        if admin_obj:
            data = dashboard_service.get_admin_dashboard_from_object(admin_obj)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

    return ResponseBase(status=True, message="Dashboard data retrieved successfully", data=data)


@router.get("/student-management/dashboard", response_model=ResponseBase)
async def get_student_management_dashboard(
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    summary = dashboard_service.get_student_management_dashboard(admin_id)
    return ResponseBase(
        status=True,
        message="Student management dashboard data retrieved",
        data=summary,
    )
    
@router.get("/profile", response_model=ResponseBase)

async def get_admin_profile(

    current_user: dict = Depends(admin_required),

) -> ResponseBase:

    admin_id = _resolve_admin_id(current_user)

    profile = admin_portal_service.get_admin_profile(admin_id)

    return ResponseBase(status=True, message="Admin profile fetched successfully", data={"profile": profile})

@router.put("/profile", response_model=ResponseBase)
async def update_admin_profile(
    full_name: str = Form(...),
    phone_number: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    profile = await admin_portal_service.update_admin_profile(
        admin_id,
        full_name=full_name,
        phone_number=phone_number,
        photo_file=photo,
    )
    return ResponseBase(status=True, message="Admin profile updated successfully", data={"profile": profile})

@router.post("/change-password")
async def change_admin_password(
    payload: ChangePasswordRequest,
    current_user: dict = Depends(admin_required),
) -> dict:
    auth_service.change_password(payload, current_user)
    return {"status": True, "message": "Password changed successfully"}


@router.get("/members", response_model=ResponseBase)
async def list_members(
    work_type: Optional[str] = Query(None),
    active_only: bool = False,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    target_admin_id = _resolve_admin_id(current_user)
    parsed_work_type = _parse_work_type(work_type)
    members = admin_portal_service.list_members(
        target_admin_id,
        work_type=parsed_work_type,
        active_only=active_only,
    )
    return ResponseBase(
        status=True,
        message="Members fetched successfully",
        data={"members": members, "admin_id": target_admin_id},
    )


@router.get("/members/{member_id}", response_model=ResponseBase)
async def get_member(
    member_id: int,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    member = admin_portal_service.get_member(member_id)
    target_admin_id = _resolve_admin_id(current_user)
    if member["admin_id"] != target_admin_id and not current_user.get("is_super_admin", False):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    return ResponseBase(status=True, message="Member fetched successfully", data={"member": member})


@router.post("/members", response_model=ResponseBase, status_code=status.HTTP_201_CREATED)
async def create_member(
    payload: MemberCreate,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    target_admin_id = _resolve_admin_id(current_user)
    member, created = admin_portal_service.create_member(target_admin_id, payload)
    message = "Member created successfully" if created else "Email already existed"
    return ResponseBase(
        status=True,
        message=message,
        data={"member": member},
    )


@router.post("/members/{target_admin_id}", response_model=ResponseBase, status_code=status.HTTP_201_CREATED)
async def create_member_for_specific_admin(
    target_admin_id: int,
    payload: MemberCreate,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    current_admin_id = _resolve_admin_id(current_user)
    if target_admin_id != current_admin_id and not current_user.get("is_super_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot create members for another admin")

    return await create_member(payload, current_user)

@router.put("/members/{target_admin_id}/{member_id}", response_model=ResponseBase)
async def update_member_for_admin(
    target_admin_id: int,
    member_id: int,
    payload: MemberUpdate,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    current_admin_id = _resolve_admin_id(current_user)
    if target_admin_id != current_admin_id and not current_user.get("is_super_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update members for another admin")

    member = admin_portal_service.update_member(target_admin_id, member_id, payload)
    return ResponseBase(status=True, message="Member updated successfully", data={"member": member})

@router.delete("/members/{target_admin_id}/{member_id}", response_model=ResponseBase)
async def delete_member_for_admin(
    target_admin_id: int,
    member_id: int,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    current_admin_id = _resolve_admin_id(current_user)
    if target_admin_id != current_admin_id and not current_user.get("is_super_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete members for another admin")

    admin_portal_service.delete_member(target_admin_id, member_id)
    return ResponseBase(status=True, message="Member deleted successfully", data=None)