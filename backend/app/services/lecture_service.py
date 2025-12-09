"""High-level service that orchestrates lecture generation and storage."""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.repository.lecture_repository import LectureRepository
from app.services.lecture_generation_service import GroqService
from app.services.tts_service import GoogleTTSService


logger = logging.getLogger(__name__)


class LectureService:
    """Provide a cohesive API for lecture CRUD and AI interactions."""

    def __init__(
        self,
        *,
        db: Session,
        groq_api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        inferred_api_key = (
            groq_api_key
            or getattr(settings, "groq_api_key", None)
            or settings.dict().get("GROQ_API_KEY")
        )
        # Use absolute path for storage to ensure consistency regardless of working directory
        default_storage = str(Path(__file__).parent.parent.parent / "storage" / "chapter_lectures")
        storage_root = getattr(settings, "chapter_lecture_storage_root", None) or default_storage

        self._repository = LectureRepository(db)
        self._generator = GroqService(api_key=inferred_api_key or "")
        self._tts_service = GoogleTTSService(
            storage_root=storage_root,
            credentials_path=getattr(settings, "gcp_tts_credentials_path", None),
        )
        self._public_base_url = (
            settings.public_base_url.rstrip("/") if getattr(settings, "public_base_url", None) else None
        )
        self._audio_storage_root = Path(storage_root)

    @property
    def repository(self) -> LectureRepository:
        return self._repository

    @property
    def generator(self) -> GroqService:
        return self._generator

    async def create_lecture_from_text(
        self,
        *,
        text: str,
        language: str,
        duration: int,
        style: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        
    ) -> Dict[str, Any]:
        if not self._generator.configured:
            raise RuntimeError("Groq service is not configured")

        lecture_payload = await self._generator.generate_lecture_content(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )

        slides: List[Dict[str, Any]] = lecture_payload.get("slides", [])  # type: ignore[assignment]
        if not slides:
            raise RuntimeError("Lecture generation produced no slides")

        context = "\n\n".join(
            filter(None, (slide.get("narration", "") for slide in slides))
        )

        record = await self._repository.create_lecture(
            title=title,
            language=language,
            style=style,
            duration=duration,
            slides=slides,
            context=context,
            text=text,
            metadata=metadata,
            fallback_used=lecture_payload.get("fallback_used", False),
        )

        return await self._attach_slide_audio(record)

    async def answer_question(
        self,
        *,
        lecture_id: str,
        question: str,
        answer_type: Optional[str] = None,
        is_edit_command: bool = False,
        context_override: Optional[str] = None,
    ) -> Any:
        record = await self._repository.get_lecture(lecture_id)
        context = context_override or record.get("context", "")
        language = record.get("language", "English")

        return await self._generator.answer_question(
            question=question,
            context=context,
            language=language,
            answer_type=answer_type,
            is_edit_command=is_edit_command,
        )

    async def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        record = await self._repository.get_lecture(lecture_id)
        return await self._attach_slide_audio(record)

    async def list_lectures(
        self,
        *,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        std: Optional[str] = None,
        subject: Optional[str] = None,
        division: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._repository.list_lectures(
            language=language,
            limit=limit,
            offset=offset,
            std=std,
            subject=subject,
            division=division,
        )

    async def delete_lecture(self, lecture_id: str) -> bool:
        return await self._repository.delete_lecture(lecture_id)

    async def get_class_subject_filters(self) -> Dict[str, Any]:
        return await self._repository.get_class_subject_filters()

    async def _attach_slide_audio(self, record: Dict[str, Any]) -> Dict[str, Any]:
        lecture_id = str(record.get("lecture_id") or "").strip()
        slides: List[Dict[str, Any]] = record.get("slides") or []
        if not lecture_id or not slides:
            return self._sanitize_audio_metadata(record)

        language = record.get("language", "English")
        updated = False
        metadata_changed = False

        for index, slide in enumerate(slides, start=1):
            tts_text = self._compose_slide_tts_text(slide)
            if not tts_text:
                continue

            filename = f"slide-{slide.get('number') or index}.mp3"
            existing_file = self._audio_storage_root / lecture_id / "audio" / filename

            audio_url = self._build_audio_url(lecture_id, filename)
            audio_download_url = self._build_audio_download_url(lecture_id, filename)

            if slide.get("audio_url") != audio_url or slide.get("audio_download_url") != audio_download_url:
                metadata_changed = True

            slide["audio_url"] = audio_url
            slide["audio_download_url"] = audio_download_url

            if existing_file.is_file():
                logger.info("Slide audio ready (existing): %s", slide["audio_url"])
                continue

            audio_path = await self._tts_service.synthesize_text(
                lecture_id=lecture_id,
                text=tts_text,
                language=language,
                filename=filename,
                subfolder="audio",
            )

            if not audio_path:
                continue

            logger.info("Slide audio generated: %s", slide["audio_url"])
            updated = True

        if not (updated or metadata_changed):
            return self._sanitize_audio_metadata(record)

        updates = {
            "slides": slides,
            "audio_generated": True,
        }
        updated_record = await self._repository.update_lecture(lecture_id, updates)
        return self._sanitize_audio_metadata(updated_record)

    def _build_audio_url(self, lecture_id: str, filename: str) -> str:
        relative_url = f"/chapter-materials/chapter_lecture/audio/{lecture_id}/{filename}"
        if self._public_base_url:
            return f"{self._public_base_url}{relative_url}"
        return relative_url

    def _build_audio_download_url(self, lecture_id: str, filename: str) -> str:
        relative_url = f"/chapter-materials/chapter_lecture/audio/{lecture_id}/{filename}/download"
        if self._public_base_url:
            return f"{self._public_base_url}{relative_url}"
        return relative_url

    def _sanitize_audio_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not record:
            return record
        sanitized = deepcopy(record)
        slides = sanitized.get("slides") or []
        for slide in slides:
            if isinstance(slide, dict):
                slide.pop("audio_path", None)
        return sanitized

    def _compose_slide_tts_text(self, slide: Dict[str, Any], *, language: str = "English") -> str:
            """Build a narration string that includes title, bullets, narration, and questions."""
            if not isinstance(slide, dict):
                return ""
            sections: List[str] = []
            bullets = [
                (bullet or "").strip()
                for bullet in slide.get("bullets") or []
                if (bullet or "").strip()
            ]
            if bullets:
                sections.append(self._format_bullet_summary(bullets, language=language))
            narration = (slide.get("narration") or "").strip()
            if narration:
                sections.append(narration)
            question = (slide.get("question") or "").strip()
            if question:
                sections.append(question)
            return " ".join(sections)
    def _format_bullet_summary(self, bullets: List[str], *, language: str = "English") -> str:
            """Create a natural sentence summarizing slide bullets, localized to the lecture language."""
            topic_list = self._human_join(bullets).rstrip(". ")
            language = (language or "English").strip()
            templates = {
                "Hindi": "आज हम {topics} के बारे में सीखेंगे.",
                "Gujarati": "આજે આપણે {topics} વિશે શીખીશું.",
            }
            template = templates.get(language, "Today, we will learn about {topics}.")
            return template.format(topics=topic_list)
    @staticmethod
    def _human_join(items: List[str]) -> str:
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return f"{', '.join(items[:-1])}, and {items[-1]}"