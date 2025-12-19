"""Dataclass and ORM models representing database tables."""

from .user import User
from .student import Student
from .teacher import Teacher
from .chapter_material import ChapterMaterial, LectureGen, LectureChatbot

__all__ = [
    "User",
    "Student",
    "Teacher",
    "ChapterMaterial",
    "LectureGen",
    "LectureChatbot",
]
