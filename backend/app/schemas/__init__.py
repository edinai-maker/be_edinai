"""Pydantic schemas for the modular example backend."""

from .user_schema import UserCreate, UserResponse
from .student_schema import StudentCreate, StudentResponse
from .teacher_schema import TeacherCreate, TeacherResponse
from .admin_schema import AdminResponse, MemberCreate, MemberResponse, MemberUpdate, PackageResponse, WorkType
from .auth_schema import (
    LoginRequest,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from .contact_schema import ContactCreate, ContactRead, ContactResponse
from .chapter_material_schema import ChapterMaterialCreate, ChapterMaterialResponse
from .education_center_schema import (
    CompleteOnboardingRequest,
    CompleteOnboardingResponse,
    EducationCenterResponse,
)
from .lecture_schema import LectureCreate, LectureDashboardResponse, LectureResponse, LectureUpdate
from .response import ErrorResponse, ResponseEnvelope, ResponseBase
from .student_portal_schema import (
    SendChatMessageRequest,
    StudentChangePasswordRequest,
    StudentLoginRequest,
    StudentLoginResponse,
    StudentVideoCommentRequest,
    StudentVideoLikeRequest,
    StudentVideoSubscribeRequest,
    StudentVideoWatchRequest,
    StudentVideoShareRequest,
    StudentProfileCreate,
    StudentProfileResponse,
    StudentSignupRequest,
)

__all__ = [
    "UserCreate",
    "UserResponse",
    "StudentCreate",
    "StudentResponse",
    "TeacherCreate",
    "TeacherResponse",
    "AdminResponse",
    "MemberCreate",
    "MemberResponse",
    "MemberUpdate",
    "PackageResponse",
    "WorkType",
    "ContactCreate",
    "ContactRead",
    "ContactResponse",
    "ChapterMaterialCreate",
    "ChapterMaterialResponse",
    "LectureCreate",
    "LectureUpdate",
    "LectureResponse",
    "LectureDashboardResponse",
    "EducationCenterResponse",
    "CompleteOnboardingRequest",
    "CompleteOnboardingResponse",
    "ChangePasswordRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    "StudentProfileCreate",
    "StudentProfileResponse",
    "StudentSignupRequest",
    "StudentLoginRequest",
    "StudentLoginResponse",
    "StudentChangePasswordRequest",
    "StudentVideoWatchRequest",
    "StudentVideoLikeRequest",
    "StudentVideoSubscribeRequest",
    "StudentVideoCommentRequest",
    "StudentVideoShareRequest",
    "SendChatMessageRequest",
    "ResponseEnvelope",
    "ResponseBase",
    "ErrorResponse",
]
