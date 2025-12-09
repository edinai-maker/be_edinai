"""Utilities for generating lecture audio via Google Cloud Text-to-Speech."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Iterator, Optional, Tuple

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

DEFAULT_CREDENTIALS_FILENAME = ".json"


def resolve_gcp_credentials_path(configured_path: str | None = None) -> Optional[str]:
    """Return a usable credentials path, falling back to the repo-level file."""
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.is_file():
            logger.info("Using GCP credentials from configured path: %s", configured_path)
            return str(candidate.resolve())
        logger.warning(
            "Configured GCP credentials path %s is not a file. "
            "Falling back to default credential detection.",
            configured_path,
        )

    backend_root = Path(__file__).resolve().parents[2]
    fallback = backend_root / DEFAULT_CREDENTIALS_FILENAME
    if fallback.is_file():
        logger.info("Using GCP credentials from fallback path: %s", fallback)
        return str(fallback.resolve())

    logger.info(
        "No GCP credentials file found. Will use default credential detection "
        "(e.g., GOOGLE_APPLICATION_CREDENTIALS env var or Application Default Credentials)."
    )
    return None


class GoogleTTSService:
    """Wrapper around Google Cloud Text-to-Speech client."""

    _CHUNK_CHAR_LIMIT = 4000

    def __init__(
        self,
        storage_root: str = "./storage/chapter_lectures",
        *,
        credentials_path: Optional[str] = None,
    ) -> None:
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)

        resolved_credentials = resolve_gcp_credentials_path(credentials_path)
        self._client = self._build_client(resolved_credentials)

    async def synthesize_text(
        self,
        *,
        lecture_id: str,
        text: str,
        language: str,
        filename: str,
        subfolder: str | None = None,
    ) -> Optional[Path]:
        """Generate an MP3 file for the provided text chunk."""
        normalized_text = (text or "").strip()
        if not normalized_text:
            logger.info(
                "Skipping TTS for lecture %s (%s) because text is empty.",
                lecture_id,
                filename,
            )
            return None

        target_path = self._build_audio_path(lecture_id, filename, subfolder=subfolder)
        voice = self._voice_for_language(language)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._write_audio_file,
            normalized_text,
            voice,
            target_path,
        )

    def _write_audio_file(
        self,
        text: str,
        voice: Tuple[str, str],
        target_path: Path,
    ) -> Optional[Path]:
        language_code, voice_name = voice
        try:
            with open(target_path, "wb") as audio_file:
                for chunk_text in self._chunk_text(text):
                    response = self._client.synthesize_speech(
                        input=texttospeech.SynthesisInput(text=chunk_text),
                        voice=texttospeech.VoiceSelectionParams(
                            language_code=language_code,
                            name=voice_name,
                        ),
                        audio_config=texttospeech.AudioConfig(
                            audio_encoding=texttospeech.AudioEncoding.MP3
                        ),
                    )
                    audio_file.write(response.audio_content)
            logger.info("Generated lecture audio at %s", target_path)
            return target_path
        except (GoogleAPICallError, OSError, ValueError) as exc:
            logger.error(
                "Failed to synthesize audio for lecture %s: %s",
                target_path.name,
                exc,
            )
            logger.exception("Full TTS error details for %s:", target_path.name)
            try:
                if target_path.exists():
                    target_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            return None

    def _build_client(self, credentials_path: Optional[str]) -> texttospeech.TextToSpeechClient:
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            return texttospeech.TextToSpeechClient(credentials=credentials)
        # Clear GOOGLE_APPLICATION_CREDENTIALS if no valid credentials path is found
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        return texttospeech.TextToSpeechClient()

    def _build_audio_path(self, lecture_id: str, filename: str, *, subfolder: str | None) -> Path:
        lecture_dir = self._storage_root / str(lecture_id)
        if subfolder:
            lecture_dir = lecture_dir / subfolder
        lecture_dir.mkdir(parents=True, exist_ok=True)
        return lecture_dir / filename

    @staticmethod
    def _voice_for_language(language: str) -> Tuple[str, str]:
        mapping = {
            "English": ("en-US", "en-IN-Chirp3-HD-Achernar"),
            "Hindi": ("hi-IN", "hi-IN-Chirp3-HD-Achernar"),
            "Gujarati": ("gu-IN", "gu-IN-Chirp3-HD-Achernar"),
        }
        return mapping.get(language, ("en-US", "en-US-Wavenet-A"))

    def _chunk_text(self, text: str) -> Iterator[str]:
        normalized = text.strip()
        if len(normalized) <= self._CHUNK_CHAR_LIMIT:
            yield normalized
            return

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self._CHUNK_CHAR_LIMIT:
                if current_chunk:
                    yield " ".join(current_chunk)
                    current_chunk = []
                    current_length = 0
                for start in range(0, len(sentence), self._CHUNK_CHAR_LIMIT):
                    yield sentence[start : start + self._CHUNK_CHAR_LIMIT]
                continue

            additional_length = len(sentence) + (1 if current_chunk else 0)
            if current_length + additional_length <= self._CHUNK_CHAR_LIMIT:
                current_chunk.append(sentence)
                current_length += additional_length
            else:
                yield " ".join(current_chunk)
                current_chunk = [sentence]
                current_length = len(sentence)

        if current_chunk:
            yield " ".join(current_chunk)