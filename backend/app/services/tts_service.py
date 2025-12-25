"""Utilities for generating lecture audio via Google Cloud Text-to-Speech."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import texttospeech
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleTTSService:
    """Wrapper around Google Cloud Text-to-Speech client."""

    _CHUNK_CHAR_LIMIT = 2500
    _SENTENCE_CHAR_LIMIT = 200
    _SENTENCE_ENDINGS = ".!?।！？"
    _SENTENCE_DELIMITER_PATTERN = re.compile(r"(?<=[.!?।！？])\s+|[\r\n]+")
    _BULLET_PREFIX_PATTERN = re.compile(r"^[\-\u2010-\u2015\u2022\u25CF\u25CB\u25A0\*]+\s+")
    _UNWANTED_SPOKEN_CHARS_PATTERN = re.compile(r"\$+|\*{2,}")

    _MATH_SYMBOL_SPOKEN_MAP = {
        "+": "plus",
        "-": "minus",
        "−": "minus",
        "*": "times",
        "×": "times",
        "/": "divided by",
        "÷": "divided by",
        "=": "equals",
        "^": "to the power of",
        "%": "percent",
    }

    _MATH_SYMBOL_PATTERN = re.compile(r"(?P<symbol>[+\-*/=×÷−^%])")

    def __init__(
        self,
        storage_root: str = "./storage/chapter_lectures",
        *,
        credentials_path: Optional[str] = None,
    ) -> None:
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)

        resolved_credentials = self._resolve_credentials_path(credentials_path)
        self._client = self._build_client(resolved_credentials)

    async def synthesize_text(
        self,
        *,
        lecture_id: str,
        text: str,
        language: str,
        filename: str,
        subfolder: str | None = None,
        model: str | None = None,
    ) -> Optional[Path]:
        """Generate an MP3 file for the provided text chunk."""
        normalized_text = self._sanitize_text((text or "").strip())
        if not normalized_text:
            logger.info(
                "Skipping TTS for lecture %s (%s) because text is empty.",
                lecture_id,
                filename,
            )
            return None

        target_path = self._build_audio_path(lecture_id, filename, subfolder=subfolder)
        voice_params = self._voice_for_voice_model(language, model)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._write_audio_file,
            normalized_text,
            voice_params,
            target_path,
        )

    # def _write_audio_file(
    #     self,
    #     text: str,
    #     voice: texttospeech.VoiceSelectionParams,
    #     target_path: Path,
    # ) -> Optional[Path]:
    #     temp_path: Optional[Path] = None
    #     try:
    #         target_path.parent.mkdir(parents=True, exist_ok=True)
    #         with tempfile.NamedTemporaryFile(
    #             mode="wb",
    #             delete=False,
    #             dir=target_path.parent,
    #             suffix=".tmp",
    #         ) as temp_file:
    #             temp_path = Path(temp_file.name)
    #             for chunk_text in self._chunk_text(text):
    #                 response = self._client.synthesize_speech(
    #                     input=texttospeech.SynthesisInput(text=chunk_text),
    #                     voice=voice,
    #                     audio_config=texttospeech.AudioConfig(
    #                         audio_encoding=texttospeech.AudioEncoding.MP3
    #                     ),
    #                 )
    #                 temp_file.write(response.audio_content)
    #         temp_path.replace(target_path)
    #         logger.info("Generated lecture audio at %s", target_path)
    #         return target_path
    #     except (GoogleAPICallError, OSError, ValueError) as exc:
    #         logger.error(
    #             "Failed to synthesize audio for lecture %s: %s",
    #             target_path.name,
    #             exc,
    #         )
    #         logger.exception("Full TTS error details for %s:", target_path.name)
    #         for path in filter(None, [temp_path, target_path]):
    #             try:
    #                 if path.exists():
    #                     path.unlink(missing_ok=True)
    #             except Exception:  # pragma: no cover - best effort cleanup
    #                 pass
    #         return None
    def _write_audio_file(
        self,
        text: str,
        voice: texttospeech.VoiceSelectionParams,
        target_path: Path,
    ) -> Optional[Path]:
        try:
            with open(target_path, "wb") as audio_file:
                for chunk_text in self._chunk_text(text):
                    response = self._client.synthesize_speech(
                        input=texttospeech.SynthesisInput(text=chunk_text),
                        voice=voice,
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

    def _build_client(self, credentials_path: str) -> texttospeech.TextToSpeechClient:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return texttospeech.TextToSpeechClient(credentials=credentials)

    def _resolve_credentials_path(self, credentials_path: Optional[str]) -> str:
        if not credentials_path:
            raise ValueError(
                "Google Cloud TTS credentials path is required and must point to a JSON file."
            )
        candidate = Path(credentials_path).expanduser()
        if not candidate.is_absolute():
            backend_root = Path(__file__).resolve().parents[2]
            candidate = backend_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Google Cloud TTS credentials file not found at {candidate}"
            )
        logger.info("Using GCP credentials from %s", candidate)
        return str(candidate)

    def _build_audio_path(self, lecture_id: str, filename: str, *, subfolder: str | None) -> Path:
        lecture_dir = self._storage_root / str(lecture_id)
        if subfolder:
            lecture_dir = lecture_dir / subfolder
        lecture_dir.mkdir(parents=True, exist_ok=True)
        return lecture_dir / filename

    @staticmethod
    def _voice_for_voice_model(language: str, model: str | None) -> texttospeech.VoiceSelectionParams:

        language = (language or "English").strip()
        normalized_model = (model or "").strip().lower()

        default_voice = {
            "English": ("en-in", "en-IN-Chirp3-HD-Achernar", texttospeech.SsmlVoiceGender.NEUTRAL),
            "Hindi": ("hi-in", "hi-IN-Chirp3-HD-Achernar", texttospeech.SsmlVoiceGender.NEUTRAL),
            "Gujarati": ("gu-in", "gu-IN-Chirp3-HD-Achernar", texttospeech.SsmlVoiceGender.NEUTRAL),
        }
        if normalized_model == "inai":
            language_code, voice_name, gender = default_voice.get(language, default_voice["English"])
            return texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=gender,
            )

        if normalized_model == "vinai":
            vinai_voice = {
                "English": ("en-in", "en-IN-Wavenet-B", texttospeech.SsmlVoiceGender.MALE),
                "Hindi": ("hi-in", "hi-IN-Wavenet-B", texttospeech.SsmlVoiceGender.MALE),
                "Gujarati": ("gu-in", "gu-IN-Wavenet-D", texttospeech.SsmlVoiceGender.MALE),
            }
            language_code, voice_name, gender = vinai_voice.get(language, vinai_voice["English"])
            return texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=gender,
            )

        if normalized_model == "aira":
            aira_voice = {
                "English": ("en-in", "en-IN-Wavenet-A", texttospeech.SsmlVoiceGender.FEMALE),
                "Hindi": ("hi-in", "hi-IN-Wavenet-A", texttospeech.SsmlVoiceGender.FEMALE),
                "Gujarati": ("gu-in", "gu-IN-Wavenet-A", texttospeech.SsmlVoiceGender.FEMALE),
            }
            language_code, voice_name, gender = aira_voice.get(language, aira_voice["English"])
            return texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=gender,
            )

        language_code, voice_name, gender = default_voice.get(language, default_voice["English"])
        return texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=gender,
        )

    def _chunk_text(self, text: str) -> Iterator[str]:
        normalized = text.strip()
        if not normalized:
            return

        segments = self._SENTENCE_DELIMITER_PATTERN.split(normalized)
        segments = [
            self._normalize_segment(segment)
            for segment in segments
            if segment.strip()
        ]
        if not segments:
            return

        current_chunk: list[str] = []
        current_length = 0

        for segment in segments:
            for safe_segment in self._split_long_segment(segment):
                addition = len(safe_segment) + (1 if current_chunk else 0)
                if current_length + addition <= self._CHUNK_CHAR_LIMIT:
                    current_chunk.append(safe_segment)
                    current_length += addition
                else:
                    if current_chunk:
                        yield " ".join(current_chunk)
                    current_chunk = [safe_segment]
                    current_length = len(safe_segment)

        if current_chunk:
            yield " ".join(current_chunk)

    def _split_long_segment(self, segment: str) -> Iterator[str]:
        """Ensure no single sentence exceeds the per-request sentence limit."""
        sentence = self._normalize_segment(segment)
        if not sentence:
            return

        if len(sentence) <= self._SENTENCE_CHAR_LIMIT:
            yield self._ensure_sentence_ending(sentence)
            return

        words = sentence.split()
        if not words:
            for start in range(0, len(sentence), self._SENTENCE_CHAR_LIMIT):
                snippet = sentence[start : start + self._SENTENCE_CHAR_LIMIT]
                yield self._ensure_sentence_ending(snippet)
            return

        current_words: list[str] = []
        current_length = 0

        for word in words:
            addition = len(word) + (1 if current_words else 0)
            if addition > self._SENTENCE_CHAR_LIMIT:
                if current_words:
                    chunk = " ".join(current_words)
                    yield self._ensure_sentence_ending(chunk)
                    current_words = []
                    current_length = 0
                for start in range(0, len(word), self._SENTENCE_CHAR_LIMIT):
                    snippet = word[start : start + self._SENTENCE_CHAR_LIMIT]
                    yield self._ensure_sentence_ending(snippet)
                continue

            if current_length + addition <= self._SENTENCE_CHAR_LIMIT:
                current_words.append(word)
                current_length += addition
            else:
                chunk = " ".join(current_words)
                yield self._ensure_sentence_ending(chunk)
                current_words = [word]
                current_length = len(word)

        if current_words:
            chunk = " ".join(current_words)
            yield self._ensure_sentence_ending(chunk)

    def _normalize_segment(self, segment: str) -> str:
        trimmed = (segment or "").strip()
        if not trimmed:
            return ""
        normalized = self._BULLET_PREFIX_PATTERN.sub("", trimmed)
        return self._sanitize_text(normalized)

    def _sanitize_text(self, text: str) -> str:

        """Remove characters that create noisy pronunciations in TTS output."""
        if not text:
            return ""
        sanitized = self._UNWANTED_SPOKEN_CHARS_PATTERN.sub(" ", text)
        sanitized = self._MATH_SYMBOL_PATTERN.sub(
            lambda match: f" {self._MATH_SYMBOL_SPOKEN_MAP.get(match.group('symbol'), match.group('symbol'))} ",
            sanitized,
        )
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        return sanitized.strip()   

    @classmethod
    def _ensure_sentence_ending(cls, sentence: str) -> str:
        trimmed = sentence.strip()
        if not trimmed:
            return sentence
        if trimmed[-1] in cls._SENTENCE_ENDINGS:
            return trimmed
        return f"{trimmed}."



        