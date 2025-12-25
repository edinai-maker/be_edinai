"""Static Runway video service replacement."""

from __future__ import annotations

import itertools
import uuid
from typing import Dict, Iterable, Optional


STATIC_VIDEO_URLS: tuple[str, ...] = (
    "https://drive.google.com/file/d/1gNVVuiS85GqSU-J_YYMkNJMLiY178sfJ/view?usp=sharing",
    "https://drive.google.com/file/d/1Yf__ujQD_3jv5xLTCeAZGKjIsFfkd242/view?usp=sharing",
    "https://drive.google.com/file/d/1tMEdSeFXi5QGXN3URnQFM5aN_DpZacG8/view?usp=sharing",
)


class RunwayVideoService:
    """Return pre-defined static video URLs while preserving original API surface."""

    def __init__(self, *, static_urls: Optional[Iterable[str]] = None) -> None:
        urls = tuple(static_urls) if static_urls is not None else STATIC_VIDEO_URLS
        if not urls:
            raise ValueError("At least one static video URL must be provided")
        self._urls = urls
        self._url_cycle = itertools.cycle(self._urls)

    @property
    def configured(self) -> bool:
        return True

    async def create_text_to_video(
        self,
        *,
        prompt_text: str,
        duration_seconds: Optional[int] = None,
        ratio: Optional[str] = None,
        model: Optional[str] = None,
        poll_interval_seconds: float = 5.0,
        max_wait_seconds: int = 180,
    ) -> Dict[str, object]:
        del prompt_text, duration_seconds, ratio, model, poll_interval_seconds, max_wait_seconds
        video_url = next(self._url_cycle)
        task_id = uuid.uuid4().hex
        return {
            "task_id": task_id,
            "video_url": video_url,
            "output": [video_url],
            "raw": {
                "status": "SUCCEEDED",
                "task_id": task_id,
                "output": [video_url],
                "static": True,
            },
        }

    async def close(self) -> None:  # pragma: no cover - interface compatibility
        return None


runway_video_service = RunwayVideoService()