"""High-level service that orchestrates lecture generation and storage."""
from __future__ import annotations

import logging
import shutil
import mimetypes
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4
import re
import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.repository.lecture_repository import LectureRepository
from app.services.lecture_generation_service import GroqService
from app.services.tts_service import GoogleTTSService
from app.utils.s3_file_handler import get_s3_service
from app.services.runway_video_service import RunwayVideoService
from app.services.runway_image_service import RunwayImageService
logger = logging.getLogger(__name__)


class LectureService:
    """Provide a cohesive API for lecture CRUD and AI interactions."""

    def __init__(
        self,
        *,
        db: Session,
        groq_api_key: Optional[str] = None,
        runway_service: Optional[RunwayVideoService] = None,
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
        self._runway_service = runway_service or RunwayVideoService(settings=settings)
        self._runway_image_service = RunwayImageService(settings=settings)
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
        await self._maybe_attach_runway_media(
            slides=slides,
            lecture_title=title,
            language=language,
            media_triggers = self._collect_media_triggers(slides)
        )


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
            reuse_existing=reuse_existing,
            media_triggers=media_triggers,
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
    async def generate_runway_video_for_slide(

        self,
        *,
        lecture_record: Optional[Dict[str, Any]] = None,
        lecture_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ensure slide 5 has a Runway video and persist the updated slides."""
        if lecture_record is None:
            if not lecture_id:
                raise ValueError("lecture_id is required to regenerate the video")
            lecture_record = await self._repository.get_lecture(lecture_id)
        target_lecture_id = lecture_record.get("lecture_id") or lecture_id
        slides = lecture_record.get("slides") or []
        if len(slides) < 5:
            raise ValueError("Lecture does not contain a fifth slide")

        await self._maybe_attach_runway_video(
            slides=slides,
            lecture_title=lecture_record.get("title") or "",
            language=lecture_record.get("language") or "English",
        )

        slide = slides[4]
        if not slide.get("video_url"):
            raise RuntimeError("Runway video generation failed")

        if target_lecture_id:
            await self._repository.update_lecture(target_lecture_id, {"slides": slides})
        return slide

    async def _maybe_attach_runway_media(
        self,
        *,
        slides: List[Dict[str, Any]],
        lecture_title: str,
        language: str,
    ) -> None:
        """Generate requested Runway assets (images/videos) for configured slide numbers."""
        if not slides:
            return

        await self._maybe_attach_runway_image(
            slides=slides,
            lecture_title=lecture_title,
            language=language,
            slide_numbers=(2, 3, 7),
        )
        await self._maybe_attach_runway_video(
            slides=slides,
            lecture_title=lecture_title,
            language=language,
            slide_numbers=(4, 5, 6),
        )

    async def _maybe_attach_runway_image(
        self,
        *,
        slides: List[Dict[str, Any]],
        lecture_title: str,
        language: str,
        slide_numbers: Sequence[int],
    ) -> None:
        """Generate Runway images for the specified slides and upload them to S3."""
        if (
            not self._runway_image_service
            or not self._runway_image_service.configured
            or not slide_numbers
        ):
            return

        for slide_number in slide_numbers:
            index = slide_number - 1
            if index < 0 or index >= len(slides):
                continue
            slide = slides[index]
            if slide.get("image_url"):
                continue

            prompt = self._build_runway_image_prompt(slide, lecture_title, language)
            if not prompt:
                continue

            try:
                result = await self._runway_image_service.create_text_to_image(
                    prompt_text=prompt,
                )
            except Exception as exc:
                logger.warning(
                    "Runway image generation failed for slide %s: %s",
                    slide_number,
                    exc,
                    exc_info=True,
                )
                continue

            runway_url = result.get("image_url")
            stored_url = await self._store_runway_asset_in_s3(
                runway_url,
                asset_type="image",
                slide_number=slide_number,
            )
            if not stored_url:
                continue

            metadata = {
                "task_id": result.get("task_id"),
                "image_url": stored_url,
                "source_url": runway_url,
                "script": prompt,
                "generated_at": datetime.utcnow().isoformat(),
                "model": getattr(self._runway_image_service, "_default_model", "runway"),
                "slide_number": slide_number,
            }
            slide["image_url"] = stored_url
            slide["runway_image"] = metadata
            slide["image_trigger_points"] = self._build_trigger_points(prompt, kind="image_prompt")
            logger.info(
                "Runway image stored for slide %s (task: %s)",
                slide_number,
                metadata["task_id"],

            )

    async def _maybe_attach_runway_video(
        self,
        *,
        slides: List[Dict[str, Any]],
        lecture_title: str,
        language: str,
        slide_numbers: Sequence[int],
    ) -> None:
        """Generate Runway videos for the specified slides."""
        if (
            not self._runway_service
            or not self._runway_service.configured
            or not slide_numbers
        ):
            return

        variant_specs = (
            (
                "cinematic",
                "Emphasize cinematic camera moves with sweeping transitions and classroom depth while keeping the scene completely free of people or characters.",
            ),
            (
                "closeup",
                "Highlight close-up shots of key concepts with smooth motion graphics overlays, focusing only on objects and diagrams without any characters.",
            ),
            (
                "chalkboard",
                "Use chalkboard-style animations with step-by-step illustrations and guiding arrows, ensuring the visuals contain no human figures.",
            ),
        )

        for slide_number in slide_numbers:
            index = slide_number - 1
            if index < 0 or index >= len(slides):
                continue
            slide = slides[index]
            base_script = self._build_runway_script(slide, lecture_title, language)
            if not base_script:
                continue

            existing_variants = slide.get("video_variants")
            variant_map: Dict[str, Dict[str, Any]] = {}
            if isinstance(existing_variants, list):
                for entry in existing_variants:
                    if isinstance(entry, dict) and entry.get("variant"):
                        variant_map[entry["variant"]] = entry

            generated_any = False
            for variant_key, variant_hint in variant_specs:
                cached_variant = variant_map.get(variant_key)
                if cached_variant and cached_variant.get("video_url"):
                    continue

                instruction_suffix = (
                    "Generate a cinematic 1 minute educational video with smooth transitions, no on-screen text or subtitles, and absolutely no people, characters, or human figures in view."
                )
                variant_prompt = f"{base_script} {variant_hint} {instruction_suffix}".strip()
                if len(variant_prompt) > 980:
                    variant_prompt = variant_prompt[:980].rsplit(" ", 1)[0] + "..."

                try:
                    result = await self._runway_service.create_text_to_video(
                        prompt_text=variant_prompt,
                        duration_seconds=60,
                    )
                except Exception as exc:
                    logger.warning(
                        "Runway video generation failed for slide %s variant %s: %s",
                        slide_number,
                        variant_key,
                        exc,
                        exc_info=True,
                    )
                    continue

                metadata = {
                    "task_id": result.get("task_id"),
                    "video_url": result.get("video_url"),
                    "duration_seconds": 60,
                    "script": base_script,
                    "generated_at": datetime.utcnow().isoformat(),
                    "model": getattr(self._runway_service, "_default_model", "runway"),
                    "slide_number": slide_number,
                    "variant": variant_key,
                }
                trigger_points = result.get("trigger_points") or self._build_trigger_points(
                    base_script,
                    duration=metadata["duration_seconds"],
                    kind="video_segment",
                )

                variant_map[variant_key] = {
                    "variant": variant_key,
                    "video_url": result.get("video_url"),
                    "trigger_points": trigger_points,
                    "metadata": metadata,
                }
                generated_any = True
                logger.info(
                    "Runway video generated for slide %s (variant: %s, lecture: %s, task: %s)",
                    slide_number,
                    variant_key,
                    lecture_title,
                    metadata["task_id"],
                )

            if not variant_map:
                continue
            ordered_variants: List[Dict[str, Any]] = []
            for variant_key, _ in variant_specs:
                entry = variant_map.get(variant_key)
                if entry:
                    ordered_variants.append(entry)
            for key, entry in variant_map.items():
                if key not in {spec[0] for spec in variant_specs}:
                    ordered_variants.append(entry)

            slide["video_variants"] = ordered_variants
            slide["runway_videos"] = [v.get("metadata") for v in ordered_variants if v.get("metadata")]

            primary = ordered_variants[0]
            slide["video_url"] = primary.get("video_url")
            slide["runway_video"] = primary.get("metadata")
            slide["video_trigger_points"] = primary.get("trigger_points")
            slide["video_trigger_sets"] = [
                {
                    "variant": variant.get("variant"),
                    "video_url": variant.get("video_url"),
                    "trigger_points": variant.get("trigger_points", []),
                }
                for variant in ordered_variants
                if variant.get("trigger_points")
            ]

            if generated_any:
                slide.pop("video_error", None)

    def _build_runway_image_prompt(self, slide: Dict[str, Any], lecture_title: str, language: str) -> str:
        """Create a descriptive visual prompt from slide content."""
        segments: List[str] = []
        title = (slide.get("title") or "").strip()
        narration = (slide.get("narration") or "").strip()
        bullets = [
            (bullet or "").strip()
            for bullet in slide.get("bullets") or []
            if (bullet or "").strip()
        ]
        question = (slide.get("question") or "").strip()

        if lecture_title:
            segments.append(f"Lecture topic: {lecture_title.strip()}.")
        if title:
            segments.append(f"Focus: {title}.")
        if narration:
            segments.append(narration)
        if bullets:
            segments.append("Key ideas: " + "; ".join(bullets[:4]))
        if question:
            segments.append(f"Reflection prompt: {question}")

        prompt = " ".join(segments).strip()
        if not prompt:
            return ""

        if len(prompt) > 980:
            prompt = prompt[:980].rsplit(" ", 1)[0] + "..."

        prompt += " Generate a single detailed educational illustration with clear visuals."
        return prompt.strip()

    def _build_runway_script(self, slide: Dict[str, Any], lecture_title: str, language: str) -> str:
        """Compose a concise script for Runway prompt based on slide 5 content."""
        segments: List[str] = []
        title = (slide.get("title") or "").strip()
        narration = (slide.get("narration") or "").strip()
        bullets = [
            (bullet or "").strip()
            for bullet in slide.get("bullets") or []
            if (bullet or "").strip()
        ]
        question = (slide.get("question") or "").strip()

        if lecture_title:
            segments.append(f"Lecture title: {lecture_title.strip()}.")
        if title:
            segments.append(f"Slide focus: {title}.")
        if narration:
            segments.append(narration)
        if bullets:
            segments.append("Key points: " + "; ".join(bullets[:4]))
        if question:
            segments.append(f"Prompt for thought: {question}")

        script = " ".join(segments).strip()
        if not script:
            return ""

        # Ensure script length within Runway limits.
        if len(script) > 980:
            script = script[:980].rsplit(" ", 1)[0] + "..."

        script += " Generate a cinematic 1 minute educational video with smooth transitions and no on-screen text or subtitles."
        return script.strip()

    def _collect_media_triggers(self, slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        triggers: Dict[str, Any] = {
            "videos": [],
            "images": [],
        }

        for slide in slides:
            if not isinstance(slide, dict):
                continue

            number = slide.get("number")

            video_variants = slide.get("video_variants")
            if isinstance(video_variants, list) and video_variants:
                for variant in video_variants:
                    trigger_points = (variant or {}).get("trigger_points")
                    if not trigger_points:
                        continue
                    triggers["videos"].append(
                        {
                            "slide_number": number,
                            "variant": variant.get("variant"),
                            "trigger_points": trigger_points,
                            "video_url": variant.get("video_url"),
                        }
                    )
            else:
                video_points = slide.get("video_trigger_points")
                if video_points:
                    triggers["videos"].append(
                        {
                            "slide_number": number,
                            "variant": "default",
                            "trigger_points": video_points,
                            "video_url": slide.get("video_url"),
                        }
                    )

            image_points = slide.get("image_trigger_points")
            if image_points:
                triggers["images"].append(
                    {
                        "slide_number": number,
                        "trigger_points": image_points,
                        "image_url": slide.get("image_url"),
                    }
                )

        if not triggers["videos"] and not triggers["images"]:
            return {}

        return triggers

    def _build_trigger_points(
        self,
        text: str,
        *,
        duration: Optional[int] = None,
        kind: str = "narration_segment",
    ) -> List[Dict[str, Any]]:
        normalized = (text or "").strip()
        if not normalized:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [normalized]

        trigger_points: List[Dict[str, Any]] = []
        interval = None
        if duration and duration > 0:
            interval = duration / max(len(sentences), 1)

        for index, sentence in enumerate(sentences, start=1):
            offset = None
            if interval is not None:
                offset = round(min(duration, (index - 1) * interval), 2)
            trigger_points.append(
                {
                    "id": f"segment_{index}",
                    "offset_seconds": offset,
                    "text": sentence,
                    "type": kind,
                }
            )

        if interval is not None:
            trigger_points.append(
                {
                    "id": "segment_end",
                    "offset_seconds": duration,
                    "text": "Segment completed",
                    "type": kind,
                }
            )

        return trigger_points