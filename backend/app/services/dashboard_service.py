"""Dashboard domain services."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from ..repository import (
    dashboard_repository,
    lecture_credit_repository,
    lecture_share_repository,
    member_repository,
    student_management_repository as roster_repository,
    member_repository,
    student_portal_video_repository,
)
from ..repository.chapter_material_repository import get_chapter_overview_data
from ..schemas import AdminResponse, MemberResponse, PackageResponse, WorkType
from ..repository.chapter_material_repository import get_chapter_overview_data
from ..plan_limits import PLAN_CREDIT_LIMITS


def _serialize_admin(admin: Optional[dict]) -> Optional[dict]:
    return AdminResponse(**admin).model_dump() if admin else None


def _serialize_member(member: Optional[dict]) -> Optional[dict]:
    return MemberResponse(**member).model_dump() if member else None


def _serialize_package(package: Optional[dict]) -> Optional[dict]:
    return PackageResponse(**package).model_dump() if package else None


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_plan_label(raw_label: Optional[str]) -> Optional[str]:
    if not raw_label:
        return None

    label = raw_label.strip().lower()
    if not label:
        return None

    label = re.sub(r"[^0-9a-z]", "", label)
    for token in ("plan", "package", "credits"):
        label = label.replace(token, "")

    if "k" not in label:
        numeric = re.search(r"(\d+)", raw_label.lower())
        if numeric:
            multiplier = "k" if "k" in raw_label.lower() else ""
            label = f"{numeric.group(1)}{multiplier}" or label

    return label or None


def _compute_expiry(admin: dict, package: Optional[dict]) -> Optional[datetime]:
    expiry = admin.get("expiry_date")
    if expiry:
        return expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)

    start_date = admin.get("start_date")
    if not start_date or not package:
        return None

    duration = package.get("duration_days")
    if not duration:
        return None

    start_dt = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    return start_dt + timedelta(days=int(duration))


# -----------------------------------------------------------
# NEW: Accurate Lecture Metrics (from Code-2)
# -----------------------------------------------------------
def _collect_admin_lecture_metrics(admin_id: int) -> Dict[str, int]:
    """Collect lecture statistics for the admin."""
    return dashboard_repository.get_admin_lecture_metrics(admin_id)


def _select_reference_student_member(admin_id: int) -> Optional[dict]:
    """Pick a representative student-management member (if any) for the admin."""
    active_students = member_repository.list_members(admin_id, work_type="student", active_only=True)
    if active_students:
        return active_students[0]
    all_students = member_repository.list_members(admin_id, work_type="student", active_only=False)
    if all_students:
        return all_students[0]
    return None
def get_chapter_overview(admin_id: int) -> Dict[str, Any]:
    """Return flattened chapter overview lecture list wrapped in items container."""
    return {"items": get_chapter_overview_data(admin_id)}


def get_admin_dashboard_from_object(admin: dict) -> Dict[str, object]:
    """Build dashboard data from an admin object (used for portal admins)."""
    admin_id = admin.get("admin_id") or admin.get("admin_aid")
    if not admin_id:
        raise ValueError("Admin ID not found in admin object")
    
        # Normalize the admin object to ensure it has the expected fields

    normalized_admin = {

        "admin_id": admin_id,

        "name": admin.get("name") or admin.get("full_name"),

        "email": admin.get("email"),

        "package": admin.get("package") or admin.get("package_plan"),

        "start_date": admin.get("start_date"),

        "expiry_date": admin.get("expiry_date"),

        "has_inai_credentials": admin.get("has_inai_credentials", False),

        "active": admin.get("active", True),

        "is_super_admin": admin.get("is_super_admin", False),

        "created_at": admin.get("created_at"),

        "last_login": admin.get("last_login"),

    }



    package = (

        dashboard_repository.fetch_package(normalized_admin.get("package"))

        if normalized_admin.get("package")

        else None

    )



    plan_label = _normalize_plan_label(normalized_admin.get("package"))

    plan_credit_total = PLAN_CREDIT_LIMITS.get(plan_label)

    credit_usage = lecture_credit_repository.get_usage(admin_id)

    credit_remaining = (

        max(plan_credit_total - credit_usage["credits_used"], 0)

        if plan_credit_total is not None

        else None

    )

    post_limit_generated = (

        credit_usage["credits_used"] - plan_credit_total

        if plan_credit_total is not None and credit_usage["credits_used"] > plan_credit_total

        else 0

    )



    total_members = dashboard_repository.count_members(admin_id, active_only=False)

    active_members = dashboard_repository.count_members(admin_id, active_only=True)

    member_counts = dashboard_repository.count_members_by_work_type(admin_id)



    computed_expiry = _compute_expiry(normalized_admin, package)

    if computed_expiry:

        delta = computed_expiry - _aware_now()

        days_until_expiry = max(delta.days, 0)

        expiry_iso = computed_expiry.isoformat()

    else:

        days_until_expiry = 0

        expiry_iso = None



    week_ago = datetime.utcnow() - timedelta(days=7)

    recent_activity = dashboard_repository.recent_member_activity(admin_id, week_ago)

    lecture_metrics = _collect_admin_lecture_metrics(admin_id)



    package_usage: Dict[str, object] = {}

    if package:

        limit = package["video_limit"] if package["video_limit"] > 0 else "unlimited"

        usage_percentage = (

            active_members / package["video_limit"] * 100

            if isinstance(limit, (int, float)) and package["video_limit"] > 0

            else 0

        )

        package_usage = {

            "members_used": active_members,

            "members_limit": limit,

            "usage_percentage": usage_percentage,

        }



    return {

        "admin_info": _serialize_admin(normalized_admin),

        "package_info": _serialize_package(package),

        "member_statistics": {

            "total_members": total_members,

            "active_members": active_members,

            "inactive_members": total_members - active_members,

            "chapter_members": member_counts.get("chapter", 0),

            "student_members": member_counts.get("student", 0),

            "lecture_members": member_counts.get("lecture", 0),

            "lecture_metrics": lecture_metrics,

        },

        "account_status": {

            "days_until_expiry": days_until_expiry,

            "is_expired": days_until_expiry < 0,

            "expiry_date": expiry_iso,

            "subscription_status": "active" if days_until_expiry > 0 else "expired",

        },

        "lecture_credits": {

            "plan_label": plan_label,

            "total": plan_credit_total,

            "used": credit_usage["credits_used"],

            "remaining": credit_remaining,

            "post_limit_generated": post_limit_generated,

            "overflow_attempts": credit_usage["overflow_attempts"],

        },

        "recent_activity": recent_activity,

        "package_usage": package_usage,

    }

def get_admin_dashboard(admin_id: int) -> Dict[str, object]:
    admin = dashboard_repository.fetch_admin(admin_id)

    if not admin:

        raise ValueError("Admin not found")



    package = (

        dashboard_repository.fetch_package(admin.get("package"))

        if admin.get("package")

        else None

    )



    plan_label = _normalize_plan_label(admin.get("package"))

    plan_credit_total = PLAN_CREDIT_LIMITS.get(plan_label)

    credit_usage = lecture_credit_repository.get_usage(admin_id)

    credit_remaining = (

        max(plan_credit_total - credit_usage["credits_used"], 0)

        if plan_credit_total is not None

        else None

    )

    post_limit_generated = (

        credit_usage["credits_used"] - plan_credit_total

        if plan_credit_total is not None and credit_usage["credits_used"] > plan_credit_total

        else 0

    )



    total_members = dashboard_repository.count_members(admin_id, active_only=False)

    active_members = dashboard_repository.count_members(admin_id, active_only=True)

    member_counts = dashboard_repository.count_members_by_work_type(admin_id)



    computed_expiry = _compute_expiry(admin, package)

    if computed_expiry:

        delta = computed_expiry - _aware_now()

        days_until_expiry = max(delta.days, 0)

        expiry_iso = computed_expiry.isoformat()

    else:

        days_until_expiry = 0

        expiry_iso = None



    week_ago = datetime.utcnow() - timedelta(days=7)

    recent_activity = dashboard_repository.recent_member_activity(admin_id, week_ago)

    lecture_metrics = _collect_admin_lecture_metrics(admin_id)



    package_usage: Dict[str, object] = {}

    if package:

        limit = package["video_limit"] if package["video_limit"] > 0 else "unlimited"

        usage_percentage = (

            active_members / package["video_limit"] * 100

            if isinstance(limit, (int, float)) and package["video_limit"] > 0

            else 0

        )

        package_usage = {

            "members_used": active_members,

            "members_limit": limit,

            "usage_percentage": usage_percentage,

        }



    return {

        "admin_info": _serialize_admin(admin),

        "package_info": _serialize_package(package),

        "member_statistics": {

            "total_members": total_members,

            "active_members": active_members,

            "inactive_members": total_members - active_members,

            "chapter_members": member_counts.get("chapter", 0),

            "student_members": member_counts.get("student", 0),

            "lecture_members": member_counts.get("lecture", 0),

            "lecture_metrics": lecture_metrics,

        },

        "account_status": {

            "days_until_expiry": days_until_expiry,

            "is_expired": days_until_expiry < 0,

            "expiry_date": expiry_iso,

            "subscription_status": "active" if days_until_expiry > 0 else "expired",

        },

        "lecture_credits": {

            "plan_label": plan_label,

            "total": plan_credit_total,

            "used": credit_usage["credits_used"],

            "remaining": credit_remaining,

            "post_limit_generated": post_limit_generated,

            "overflow_attempts": credit_usage["overflow_attempts"],

        },

        "recent_activity": recent_activity,

        "package_usage": package_usage,

    }


def get_admin_lecture_dashboard(admin_id: int) -> Dict[str, object]:
    """Return lecture team overview for the admin."""

    metrics = _collect_admin_lecture_metrics(admin_id)

    lecture_members = member_repository.list_members(
        admin_id,
        work_type=WorkType.LECTURE.value,
        active_only=False,
    )

    members_payload: List[Dict[str, Any]] = []
    assigned_totals = {
        "total_lectures": 0,
        "played_lectures": 0,
        "shared_lectures": 0,
        "pending_lectures": 0,
        "qa_sessions": 0,
    }

    for member in lecture_members:
        member_id = member.get("member_id")
        if member_id is None:
            continue

        counts = dashboard_repository.get_member_lecture_metrics(admin_id, member_id)

        assigned_totals["total_lectures"] += counts["total_lectures"]
        assigned_totals["played_lectures"] += counts["played_lectures"]
        assigned_totals["shared_lectures"] += counts["shared_lectures"]
        assigned_totals["pending_lectures"] += counts["pending_lectures"]
        assigned_totals["qa_sessions"] += counts["qa_sessions"]

        members_payload.append(
            {
                "member_id": member_id,
                "name": member.get("name"),
                "email": member.get("email"),
                "designation": member.get("designation"),
                "active": bool(member.get("active", True)),
                "total_lectures": counts["total_lectures"],
                "played_lectures": counts["played_lectures"],
                "shared_lectures": counts["shared_lectures"],
                "qa_sessions": counts["qa_sessions"],
                "last_login": member.get("last_login"),
            }
        )

    unassigned_counts = {
        "total_lectures": max(metrics.get("total_lectures", 0) - assigned_totals["total_lectures"], 0),
        "played_lectures": max(metrics.get("played_lectures", 0) - assigned_totals["played_lectures"], 0),
        "shared_lectures": max(metrics.get("shared_lectures", 0) - assigned_totals["shared_lectures"], 0),
        "pending_lectures": max(metrics.get("pending_lectures", 0) - assigned_totals["pending_lectures"], 0),
        "qa_sessions": max(metrics.get("qa_sessions", 0) - assigned_totals["qa_sessions"], 0),
    }

    totals = {
        "total_lectures": metrics.get("total_lectures", 0),
        "played_lectures": metrics.get("played_lectures", 0),
        "shared_lectures": metrics.get("shared_lectures", 0),
        "qa_sessions": metrics.get("qa_sessions", 0),
        "pending_lectures": metrics.get("pending_lectures", 0),
    }

    return {
        "admin_id": admin_id,
        "generated_at": _aware_now().isoformat(),
        "team_size": len(members_payload),
        "totals": totals,
        "members": members_payload,
        "unassigned": {
            "label": "unassigned",
            "total_lectures": unassigned_counts.get("total_lectures", 0),
            "played_lectures": unassigned_counts.get("played_lectures", 0),
            "shared_lectures": unassigned_counts.get("shared_lectures", 0),
            "qa_sessions": 0,
            "pending_lectures": unassigned_counts.get("pending_lectures", 0),
        },
    }


def get_member_dashboard(*, member_id: int, admin_id: int, work_type: str) -> Dict[str, object]:
    member = dashboard_repository.fetch_member(member_id)
    if not member:
        raise ValueError("Member not found")

    admin = dashboard_repository.fetch_admin(admin_id)
    package = (
        dashboard_repository.fetch_package(admin.get("package"))
        if admin and admin.get("package")
        else None
    )

    if work_type == "chapter":
        total_students = dashboard_repository.count_members(admin_id, work_type="student", active_only=True)
        total_lectures = dashboard_repository.count_members(admin_id, work_type="lecture", active_only=True)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_student_activity = dashboard_repository.count_members_since(
            admin_id,
            since=week_ago,
            work_type="student",
            field="last_login",
        )
        chapter_overview = get_chapter_overview_data(admin_id)
        payload = {
            "chapter_metrics": {
                "total_students": total_students,
                "total_lectures": total_lectures,
                "recent_student_activity": recent_student_activity,
            },
            "chapter_overview": chapter_overview
        }
    elif work_type == "student":
        # total_chapters = dashboard_repository.count_members(admin_id, work_type="chapter", active_only=True)
        totals = student_portal_video_repository.get_video_totals_for_std(admin_id=admin_id, std=None)
        total_lectures = totals.get("total_videos", 0) # Student dashboard must mirror legacy SM API (static lecture counts)
        total_students = roster_repository.count_roster_students(admin_id, member_id=member_id)
        payload = {
            "student_metrics": {
                # "total_chapters": total_chapters,
                "total_lectures": total_lectures,
                "total_students": total_students,
                "total_paid": 0,
                "progress": {
                    "completed_lectures": 0,
                    "total_available_lectures": total_lectures,
                    "progress_percentage": 0,
                    "last_activity": member["last_login"].isoformat() if member.get("last_login") else None,
                },
            }
        }
    else:  # lecture
        # Lecture dashboard for a lecture member should reflect real shared lectures.
        total_lectures = dashboard_repository.count_total_lectures(admin_id)
        played_lectures = dashboard_repository.count_played_lectures(admin_id)
        pending_lectures = max(total_lectures - played_lectures, 0)
        # Reuse the same source of truth as the `/lectures/shared` API so
        # that the dashboard shows exactly the lectures that have been
        # shared with students for this admin.
        shared_lecture_records = lecture_share_repository.list_shared_lectures(
            admin_id=admin_id,
            std=None,
            subject=None,
        )
        shared_lecture_count = len(shared_lecture_records)
        
        return {
            "total_lectures": total_lectures,
            "played_lectures": played_lectures,
            "shared_lectures": shared_lecture_count,
            "pending_lectures": pending_lectures,
            "qa_sessions": 0,
        }

    plan_label = _normalize_plan_label(admin.get("package") or admin.get("package_plan"))
    return {
        "member_info": _serialize_member(member),
        "admin_info": {
            "name": admin["name"],
            "email": admin["email"],
            "plan_label": plan_label or admin.get("package") or admin.get("package_plan"),
        }
        if admin
        else None,
        "package_info": _serialize_package(package),
        "work_type": work_type,
        **payload,
    }

def get_student_management_dashboard(admin_id: int) -> Dict[str, object]:
    """Return condensed student management metrics for the admin portal dashboard."""

    total_rostered_students = roster_repository.count_roster_students(admin_id)
    student_members = member_repository.list_members(admin_id, work_type="student", active_only=True)
    if not student_members:
        student_members = member_repository.list_members(admin_id, work_type="student", active_only=False)

    reference_member = student_members[0] if student_members else None

    admin = dashboard_repository.fetch_admin(admin_id)
    package = (
        dashboard_repository.fetch_package(admin.get("package"))
        if admin and admin.get("package")
        else None
    )

    total_chapters = dashboard_repository.count_members(admin_id, work_type="chapter", active_only=True)
    total_lectures = dashboard_repository.count_total_lectures(admin_id)

    last_activity = (
        reference_member["last_login"].isoformat()
        if reference_member and reference_member.get("last_login")
        else None
    )

    student_metrics = {
        "total_chapters": total_chapters,
        "total_lectures": total_lectures,
        "total_students": total_rostered_students,
        "total_paid": 0,
        "progress": {
            "completed_lectures": 0,
            "total_available_lectures": total_lectures,
            "progress_percentage": 0,
            "last_activity": last_activity,
        },
    }

    return {
        "package_info": _serialize_package(package),
        "work_type": "student",
        "student_metrics": student_metrics,
    }
def get_summary(admin_id: int) -> Dict[str, object]:
    summary = dashboard_repository.dashboard_summary(admin_id)
    summary["alerts"] = []
    if summary["days_until_expiry"] <= 30:
        summary["alerts"].append(
            {
                "type": "warning",
                "message": f"Subscription expires in {summary['days_until_expiry']} days",
            }
        )
    if summary["total_members"] == 0:
        summary["alerts"].append(
            {
                "type": "info",
                "message": "No members added yet. Add your first team member!",
            }
        )
    return summary
def get_student_management_summary(admin_id: int) -> Dict[str, int]:
    """Return only roster-based student counts for the admin."""
    roster_students = roster_repository.count_roster_students(admin_id)
    return {
        # "roster_students": roster_students,
        "total_students": roster_students,
        "total_lectures": 0,
        "total_paid": 0,
    }