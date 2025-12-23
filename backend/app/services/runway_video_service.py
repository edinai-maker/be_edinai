"""Async client for Runway text-to-video generations."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class RunwayVideoService:
    """Thin wrapper around Runway's text-to-video API."""

    def __init__(self, *, settings=None, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings or get_settings()
        self._api_key = self._settings.runway_api_key
        self._text_to_video_url = self._settings.runway_text_to_video_url
        self._tasks_base_url = self._settings.runway_tasks_base_url.rstrip("/")
        self._api_version = self._settings.runway_api_version
        self._default_model = self._settings.runway_text_to_video_model
        self._default_ratio = self._settings.runway_text_to_video_ratio
        self._default_duration = self._settings.runway_text_to_video_duration
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def close(self) -> None:
        await self._client.aclose()

    async def create_text_to_video(
        self,
        *,
        prompt_text: str,
        duration_seconds: Optional[int] = None,
        ratio: Optional[str] = None,
        model: Optional[str] = None,
        poll_interval_seconds: float = 5.0,
        max_wait_seconds: int = 180,
    ) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError("Runway API is not configured")

        resolved_model = model or self._default_model
        resolved_ratio = ratio or self._default_ratio
        resolved_duration = duration_seconds or self._default_duration

        if "api.dev.runwayml.com" in self._text_to_video_url:
            dev_allowed_ratios = {"1280:720", "720:1280", "1080:1920", "1920:1080"}
            if resolved_ratio not in dev_allowed_ratios:
                logger.debug(
                    "Runway dev API only accepts %s ratios; falling back from %s to 1280:720",
                    ", ".join(sorted(dev_allowed_ratios)),
                    resolved_ratio,
                )
                resolved_ratio = "1280:720"

            dev_allowed_durations = {4, 6, 8}
            if resolved_duration not in dev_allowed_durations:
                logger.debug(
                    "Runway dev API only accepts durations %s seconds; falling back from %s to 8",
                    ", ".join(str(v) for v in sorted(dev_allowed_durations)),
                    resolved_duration,
                )
                resolved_duration = 8

        payload = {
            "model": resolved_model,
            "promptText": prompt_text[:1000],
            "duration": resolved_duration,
            "ratio": resolved_ratio,
            "audio": False,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Runway-Version": self._api_version,
        }

        response = await self._client.post(self._text_to_video_url, json=payload, headers=headers)
        response.raise_for_status()
        task = response.json()
        task_id = task.get("id")
        if not task_id:
            raise RuntimeError("Runway API did not return a task id")

        deadline = asyncio.get_event_loop().time() + max_wait_seconds
        status = task.get("status")
        last_payload: Dict[str, Any] = task

        while status not in {"SUCCEEDED", "FAILED"}:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Runway task {task_id} timed out")
            await asyncio.sleep(poll_interval_seconds)
            poll_url = f"{self._tasks_base_url}/{task_id}"
            poll_resp = await self._client.get(poll_url, headers=headers)
            poll_resp.raise_for_status()
            last_payload = poll_resp.json()
            status = last_payload.get("status")

        if status != "SUCCEEDED":
            raise RuntimeError(f"Runway task {task_id} failed: {last_payload}")

        output_urls = last_payload.get("output") or []
        if not output_urls:
            raise RuntimeError(f"Runway task {task_id} succeeded without output URLs")

        trigger_points = self._compute_trigger_points(payload["promptText"], resolved_duration)

        return {
            "task_id": task_id,
            "video_url": output_urls[0],
            "output": output_urls,
            "raw": last_payload,
            "trigger_points": trigger_points,
        }

    def _compute_trigger_points(self, script: str, duration_seconds: int) -> List[Dict[str, Any]]:
        """Derive simple trigger points from the generated script."""
        normalized_script = (script or "").strip()
        if not normalized_script or duration_seconds <= 0:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized_script)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [normalized_script]

        interval = duration_seconds / max(len(sentences), 1)
        trigger_points: List[Dict[str, Any]] = []
        for index, sentence in enumerate(sentences, start=1):
            offset = round(min(duration_seconds, (index - 1) * interval), 2)
            trigger_points.append(
                {
                    "id": f"segment_{index}",
                    "offset_seconds": offset,
                    "text": sentence,
                    "type": "narration_segment",
                }
            )

        trigger_points.append(
            {
                "id": "video_end",
                "offset_seconds": duration_seconds,
                "text": "Segment completed",
                "type": "narration_segment",
            }
        )

        return trigger_points


runway_video_service = RunwayVideoService()
