"""Pydantic schemas for the student portal module."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from attr.converters import optional
from pydantic import BaseModel, EmailStr, Field, HttpUrl


class StudentProfileBase(BaseModel):
    first_name: str = Field(..., max_length=255)
    father_name: str = Field(..., max_length=255)
    class_stream: str = Field(..., max_length=255)
    division: Optional[str] = Field(None, max_length=255)
    class_head: Optional[str] = Field(None, max_length=255)
    enrollment_number: str = Field(..., max_length=255)
    mobile_number: Optional[str] = Field(None, max_length=20)
    parents_number: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None


class StudentProfileCreate(StudentProfileBase):
    photo_path: Optional[str] = None


class StudentProfileResponse(StudentProfileBase):
    id: int
    photo_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StudentSignupRequest(BaseModel):
    enrollment_number: str = Field(..., max_length=255, alias="enrolment_number")
    password: str = Field(..., min_length=6, max_length=128)

    class Config:
        allow_population_by_field_name = True


class StudentLoginRequest(BaseModel):
    enrollment_number: str = Field(..., alias="enrolment_number")
    password: str

    class Config:
        allow_population_by_field_name = True


class StudentLoginResponse(BaseModel):
    status: bool
    message: str
    token: Optional[str] = None
    profile_complete: Optional[bool] = None


class StudentChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6, max_length=128)


class StudentVideoWatchRequest(BaseModel):
    watch_seconds: int = Field(..., ge=1, le=60 * 60)
    duration_seconds: Optional[int] = Field(None, ge=1, le=60 * 60 * 24)


class StudentVideoUploadRequest(BaseModel):
    title: str = Field(..., max_length=255)
    subject: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    std: Optional[str] = Field(None, max_length=50)


class StudentVideoExternalRequest(BaseModel):
    title: str = Field(..., max_length=255)
    subject: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    std: Optional[str] = Field(None, max_length=50)
    video_url: HttpUrl
    thumbnail_url: Optional[HttpUrl] = None


class StudentVideoLikeRequest(BaseModel):
    liked: bool = Field(...)


class StudentVideoSubscribeRequest(BaseModel):
    subscribed: bool = Field(...)


class StudentVideoCommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, max_length=2000)


class SendChatMessageRequest(BaseModel):
    peer_enrollment: str = Field(..., max_length=255)
    message: Optional[str] = Field(None, max_length=2_000)

class StudentVideoShareRequest(BaseModel):
    peer_enrollment: str = Field(..., max_length=255)
    message: Optional[str] = Field(None, max_length=500)