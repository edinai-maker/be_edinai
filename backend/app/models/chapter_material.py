"""SQLAlchemy model for uploaded chapter materials."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Integer, String, func, JSON, Text

from app.database import Base


class ChapterMaterial(Base):
    __tablename__ = "chapter_materials"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, nullable=False, index=True)
    std = Column(String(32), nullable=False)
    subject = Column(String(128), nullable=False)
    sem = Column(String(32), nullable=True)
    board = Column(String(64), nullable=True)
    chapter_number = Column(String(128), nullable=False)
    chapter_title = Column(String(255), nullable=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(BigInteger, nullable=False, default=0)
    is_global = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "admin_id": self.admin_id,
            "std": self.std,
            "subject": self.subject,
            "sem": self.sem,
            "board": self.board,
            "chapter_number": self.chapter_number,
            "chapter_title": self.chapter_title,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

class LectureChatbot(Base):
    """Store chat interactions and audio assets generated for lectures."""
    __tablename__ = "lecture_chatbot"
    id = Column(Integer, primary_key=True, index=True)
    lecture_id = Column(String(64), nullable=False, index=True)
    question = Column(Text, nullable=True)
    response_text = Column(Text, nullable=True)
    audio_url = Column(String(512), nullable=True)
    language = Column(String(64), nullable=True)
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lecture_id": self.lecture_id,
            "question": self.question,
            "response_text": self.response_text,
            "audio_url": self.audio_url,
            "language": self.language,
            "extra_data": self.extra_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

class LectureGen(Base):
    """Model for storing generated lectures"""
    __tablename__ = "lecture_gen"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, nullable=False, index=True)
    material_id = Column(Integer, nullable=False, index=True)
    lecture_uid = Column(String(64), nullable=False, unique=True, index=True)
    # The underlying column is still named "lecture_title" in the database; keep
    # that name until a migration is run, but expose it in code as chapter_title.
    chapter_title = Column("lecture_title", String(255), nullable=False)
    lecture_link = Column(String(512), nullable=False)  # JSON URL will be stored here
    lecture_data = Column(JSON, nullable=True)
    cover_photo_url = Column(String(512), nullable=True)

    subject = Column(String(128), nullable=True)
    std = Column(String(32), nullable=True)
    sem = Column(String(32), nullable=True)
    board = Column(String(64), nullable=True)
    lecture_shared = Column(Boolean, nullable=False, default=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "admin_id": self.admin_id,
            "material_id": self.material_id,
            "lecture_uid": self.lecture_uid,
            "chapter_title": self.chapter_title,
            "lecture_link": self.lecture_link,
            "subject": self.subject,
            "std": self.std,
            "sem": self.sem,
            "board": self.board,
            "cover_photo_url":self.cover_photo_url,
            "lecture_shared": self.lecture_shared,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }