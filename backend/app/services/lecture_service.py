"""High-level service that orchestrates lecture generation and storage."""
from __future__ import annotations

import logging
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4
import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.repository.lecture_repository import LectureRepository
from app.services.lecture_generation_service import GroqService
from app.services.tts_service import GoogleTTSService
from app.utils.s3_file_handler import get_s3_service

from app.services.runway_image_service import STATIC_IMAGE_URLS
from app.services.runway_video_service import STATIC_VIDEO_URLS
logger = logging.getLogger(__name__)


STATIC_IMAGE_SLIDE_NUMBERS = (1, 2, 3, 7, 8)
STATIC_VIDEO_SLIDE_NUMBERS = (4, 5, 6)


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
       
        self._s3_service = get_s3_service(settings)
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
        model: Optional[str] = None,
        reuse_existing: bool = False,
    ) -> Dict[str, Any]:
        if not self._generator.configured:
            raise RuntimeError("Groq service is not configured")

        metadata_payload: Dict[str, Any] = dict(metadata or {})
        if model:
            metadata_payload["model"] = model

        lecture_payload = await self._generator.generate_lecture_content(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )

        slides: List[Dict[str, Any]] = lecture_payload.get("slides", [])  # type: ignore[assignment]
        if not slides:
            raise RuntimeError("Lecture generation produced no slides")
        estimated_duration=lecture_payload.get("estimated_duration")
        self._attach_static_image_links(slides)
        self._attach_static_video_links(slides)


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
            metadata=metadata_payload,
            fallback_used=lecture_payload.get("fallback_used", False),
            estimated_duration=estimated_duration,
            reuse_existing=reuse_existing,
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
        lecture_record: Optional[Dict[str, Any]] = None,
    ) -> Any:

        record = lecture_record or await self._repository.get_lecture(lecture_id)
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

    async def _attach_slide_audio(
        self,
        record: Dict[str, Any],
        *,
        force_regeneration: bool = False,
    ) -> Dict[str, Any]:
        """Ensure each slide has up-to-date audio, optionally forcing regeneration."""
        lecture_id = str(record.get("lecture_id") or "").strip()
        slides: List[Dict[str, Any]] = record.get("slides") or []
        if not lecture_id or not slides:
            return self._sanitize_audio_metadata(record)

        if force_regeneration:
            audio_dir = self._audio_storage_root / lecture_id / "audio"
            try:
                if audio_dir.exists():
                    shutil.rmtree(audio_dir)
                    logger.info("Cleared existing audio directory for lecture %s", lecture_id)
            except OSError as exc:
                logger.warning("Failed to clean audio directory for %s: %s", lecture_id, exc)

        language = record.get("language", "English")
        voice_model = (record.get("metadata") or {}).get("model")
        updated = False
        metadata_changed = False

        for index, slide in enumerate(slides, start=1):
            tts_text = self._compose_slide_tts_text(slide, language=language)
            # Skip audio generation if no content available
            if not tts_text or len(tts_text.strip()) < 10:
                logger.warning(
                    "Slide %d has insufficient content for TTS, skipping audio generation",
                    slide.get('number', index)
                )
                continue
            filename = f"slide-{slide.get('number') or index}.mp3"
            existing_file = self._audio_storage_root / lecture_id / "audio" / filename

            must_regenerate_audio = force_regeneration or not existing_file.is_file()

            if must_regenerate_audio:
                audio_path = await self._tts_service.synthesize_text(
                    lecture_id=lecture_id,
                    text=tts_text,
                    language=language,
                    filename=filename,
                    subfolder="audio",
                    model=voice_model,
                )

                if not audio_path:
                    logger.error("Failed to generate audio for slide %d", slide.get('number', index))
                    continue

                # Remove audio_version - we don't need versioning
                slide.pop("audio_version", None)
                logger.info("Slide audio generated: %s", filename)
                updated = True
            else:
                # Remove audio_version for existing files too
                slide.pop("audio_version", None)
                logger.info("Slide audio ready (existing): %s", filename)

            # Build URLs WITHOUT version parameter
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
                model=voice_model,
            )

            if not audio_path:
                logger.error("Failed to generate audio for slide %d", slide.get('number', index))
                continue
            logger.info("Slide audio generated: %s", slide["audio_url"])
            updated = True

        if not (updated or metadata_changed):
            return self._sanitize_audio_metadata(record)

        # Update the in-memory record with changes
        record["slides"] = slides
        record["audio_generated"] = True
        record["updated_at"] = datetime.utcnow().isoformat()
        
        # Only persist to database if we have a db_record_id (meaning it was already created)
        if record.get("db_record_id"):
            updates = {
                "slides": slides,
                "audio_generated": True,
            }
            try:
                updated_record = await self._repository.update_lecture(lecture_id, updates)
                return self._sanitize_audio_metadata(updated_record)
            except FileNotFoundError:
                logger.warning("Could not update lecture %s in database, returning in-memory record", lecture_id)
                return self._sanitize_audio_metadata(record)
        
        return self._sanitize_audio_metadata(record)

    def _build_audio_url(
        self,
        lecture_id: str,
        filename: str,
        *,
        subfolder: str = "audio",
    ) -> str:
        """Build audio URL WITHOUT version parameter."""
        relative_url = f"/chapter-materials/chapter_lecture/{subfolder}/{lecture_id}/{filename}"
        if self._public_base_url:
            return f"{self._public_base_url}{relative_url}"
        return relative_url

    def _build_audio_download_url(
        self,
        lecture_id: str,
        filename: str,
        *,
        subfolder: str = "audio",
    ) -> str:
        """Build download URL WITHOUT version parameter."""
        relative_url = f"/chapter-materials/chapter_lecture/{subfolder}/{lecture_id}/{filename}/download"
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
    def _attach_static_image_links(self, slides: List[Dict[str, Any]]) -> None:

        """Assign static image URLs to predefined slides."""

        if not slides:

            return



        for slide_number, url in zip(STATIC_IMAGE_SLIDE_NUMBERS, STATIC_IMAGE_URLS):

            index = slide_number - 1

            if index < 0 or index >= len(slides):

                continue

            slide = slides[index]

            if not isinstance(slide, dict):

                continue

            slide.pop("runway_image", None)

            slide["image_url"] = url

            slide["static_image"] = {

                "url": url,

                "assigned_slide": slide_number,

                "source": "static",

            }



    def _attach_static_video_links(self, slides: List[Dict[str, Any]]) -> None:

        """Assign static video URLs to predefined slides."""

        if not slides:

            return



        for slide_number, url in zip(STATIC_VIDEO_SLIDE_NUMBERS, STATIC_VIDEO_URLS):

            index = slide_number - 1

            if index < 0 or index >= len(slides):

                continue

            slide = slides[index]

            if not isinstance(slide, dict):

                continue

            slide.pop("runway_video", None)

            slide["video_url"] = url

            slide["static_video"] = {

                "url": url,

                "assigned_slide": slide_number,

                "source": "static",

            }    
        
    async def synthesize_chat_answer_audio(
        self,
        *,
        lecture_id: str,
        text: str,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a short audio clip for an assistant response, upload to S3, and return URL."""
        normalized_text = (text or "").strip()
        if not normalized_text:
            return None
        filename = f"chat-{uuid4().hex}.mp3"
        try:
            audio_path = await self._tts_service.synthesize_text(
                lecture_id=str(lecture_id),
                text=normalized_text,
                language=language or "English",
                filename=filename,
                subfolder="chat",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Chat TTS failed for lecture %s: %s", lecture_id, exc)
            return None
        if not audio_path:
            return None
        try:
            audio_bytes = audio_path.read_bytes()

        except OSError as exc:
            logger.error("Failed reading synthesized chat audio for %s: %s", lecture_id, exc)
            return None
        s3_folder = f"audio/chat/{lecture_id}"
        try:
            upload_result = self._s3_service.upload_file(
                file_content=audio_bytes,
                file_name=filename,
                folder=s3_folder,
                content_type="audio/mpeg",
                public=True,
            )
        except Exception as exc:
            logger.warning("Uploading chat audio to S3 failed for lecture %s: %s", lecture_id, exc)
            return None
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
        return upload_result.get("s3_url")

    async def save_chat_interaction(
        self,
        *,
        lecture_id: str,
        question: Optional[str],
        response_text: Optional[str],
        audio_url: Optional[str],
        language: Optional[str],
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist chatbot exchange for a lecture."""
        return await self._repository.create_chatbot_entry(
            lecture_id=lecture_id,
            question=question,
            response_text=response_text,
            audio_url=audio_url,
            language=language,
            extra_data=extra_data,
        )
    def build_pause_prompt_message(self, language: str = "English") -> str:
        """Return a localized pause/resume prompt shown when the lecture is paused."""
        normalized = (language or "English").strip().lower()
        templates = {
            "hindi": "कृपया आगे बढ़ने के लिए तैयार हों। क्या आप अगले भाग के लिए तैयार हैं?",
            "gujarati": "મહેરબાની કરીને આગળનો ભાગ શરૂ કરવા તૈયાર રહો. શું તમે તૈયાર છો?",
            "english": "Please get ready to continue. Let me know when you want to resume.",
        }
        return templates.get(normalized, templates["english"])

    def _compose_slide_tts_text(self, slide: Dict[str, Any], *, language: str = "English") -> str:
        """Build a narration string that includes title, bullets, narration, and questions."""
        if not isinstance(slide, dict):
            return ""
        sections: List[str] = []
        # Add title as introduction
        title = (slide.get("title") or "").strip()
        if title:
            sections.append(title)
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
            # Add a prefix for questions
            question_intro = self._get_question_intro(language)
            sections.append(f"{question_intro} {question}")
        return " ".join(sections)

    def _get_question_intro(self, language: str) -> str:
        """Get localized question introduction."""
        intros = {
            "Hindi": "अब कुछ सवाल:",
            "Gujarati": "હવે કેટલાક પ્રશ્નો:",
            "English": "Now, some questions:"
        }
        return intros.get(language, intros["English"])
    def _format_bullet_summary(self, bullets: List[str], *, language: str = "English") -> str:
            """Create a natural sentence summarizing slide bullets, localized to the lecture language."""
            if not bullets:
                return ""
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