"""Auth dependencies (ported from universal_jwt_handler)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from ..config import settings
from ..repository import auth_repository, member_repository, registration_repository
from ..schemas import WorkType

ALGORITHM = "HS256"
_security = HTTPBearer()
def _as_aware(dt: Optional[datetime]) -> Optional[datetime]:

    if dt is None:

        return None
    
    if dt.tzinfo is None:

        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire_delta = expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": datetime.utcnow() + expire_delta})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except JWTError as exc:  # pragma: no cover - bubbled to FastAPI
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _normalize_admin_record(admin: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(admin)
    if "admin_id" not in normalized:
        alt_id = normalized.get("admin_aid") or normalized.get("id")
        if alt_id is not None:
            normalized["admin_id"] = alt_id
    normalized.setdefault("active", True)
    if "has_inai_credentials" not in normalized:
        normalized["has_inai_credentials"] = bool(
            normalized.get("inai_email") and normalized.get("inai_password_encrypted")
        )
    normalized.setdefault("is_super_admin", False)
    return normalized


def _get_admin_record(user_id: int) -> Optional[Dict[str, Any]]:
    admin = auth_repository.get_admin_by_id(user_id)
    if admin:
        return admin
    admin = registration_repository.get_admin_by_id(user_id)
    return admin


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_user_payload(token: str) -> Dict[str, Any]:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    payload = verify_token(token)
    role = payload.get("role")
    user_id = payload.get("id")

    if role is None or user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    if role == "admin":
        admin = _get_admin_record(user_id)
        if not admin:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin not found")

        admin = _normalize_admin_record(admin)
        expiry = admin.get("expiry_date")
        if expiry and datetime.now(timezone.utc) > _ensure_utc(expiry):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Admin account expired on {expiry.strftime('%Y-%m-%d')}",
            )
        if not admin.get("active", True):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin account is inactive")
        return {
            "role": "admin",
            "id": admin["admin_id"],
            "package": admin.get("package") or admin.get("package_plan"),
            "has_inai_credentials": admin.get("has_inai_credentials", False),
            "is_super_admin": admin.get("is_super_admin", False),
            "user_obj": admin,
        }

    if role == "member":
        member = member_repository.get_member_by_id(user_id)
        if not member:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member not found")
        if not member["active"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member account is inactive")

        admin = auth_repository.get_admin_by_id(member["admin_id"])
        now_utc = datetime.now(timezone.utc)
        admin_expiry = admin.get("expiry_date") if admin else None
        if not admin or not admin["active"] or (admin_expiry and now_utc > _ensure_utc(admin_expiry)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Associated admin account is inactive or expired",
            )
        return {
            "role": "member",
            "id": member["member_id"],
            "work_type": member["work_type"],
            "admin_id": member["admin_id"],
            "user_obj": member,
        }

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user role")

def resolve_user_from_token(token: str) -> Dict[str, Any]:
    """Utility for non-request contexts (e.g., Socket.IO handshake)."""
    return _resolve_user_payload(token)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> Dict[str, Any]:
    return _resolve_user_payload(credentials.credentials)

def admin_required(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


def onboarding_required(current_user: Dict[str, Any] = Depends(admin_required)) -> Dict[str, Any]:
    """Allow admins through even if onboarding not yet completed."""

    return current_user


def onboarding_completed_required(current_user: Dict[str, Any] = Depends(admin_required)) -> Dict[str, Any]:
    if current_user.get("is_super_admin"):
        return current_user
    if not current_user.get("has_inai_credentials", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Please complete onboarding first")
    return current_user


def member_required(required_work_type: Optional[WorkType] = None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def _member_required(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if current_user["role"] != "member":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Member access required")
        if required_work_type and current_user["work_type"] != required_work_type.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied for this role")
        return current_user

    return _member_required


def admin_or_chapter_member(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Allow admin or chapter members."""
    if current_user["role"] == "admin":
        return current_user
    if current_user["role"] == "member" and current_user.get("work_type") == "chapter":
        return current_user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin or chapter member access required")


def admin_or_lecture_member(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Allow admin or lecture members."""
    if current_user["role"] == "admin":
        return current_user
    if current_user["role"] == "member" and current_user.get("work_type") == "lecture":
        return current_user
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin or lecture member access required")
