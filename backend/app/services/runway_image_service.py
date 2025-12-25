"""Static Runway image service replacement."""

from __future__ import annotations

import itertools
import uuid
from typing import Dict, Iterable, Optional


STATIC_IMAGE_URLS: tuple[str, ...] = (
    "https://drive.google.com/file/d/1-ld3bTOR50jRXsP9h8erYRSMpBVGgTPA/view?usp=sharing",
    "https://drive.google.com/file/d/1zSd6owxP6FsXdX1FXUmaHng9t5xwA0mC/view?usp=sharing",
    "https://drive.google.com/file/d/1sZYcQI9Vlr7bsnMBuGEzXBFiTCrzWs2c/view?usp=sharing",
    "https://drive.google.com/file/d/1YHnp8J5zgDhcFH37vWIitjUT-wQsIcvG/view?usp=sharing",
    "https://drive.google.com/file/d/1LKJk1XR0JKGEYHL7VKTqE79rn63d5Mlb/view?usp=sharing",
)


class RunwayImageService:
    """Return pre-defined static image URLs while preserving original API surface."""

    def __init__(self, *, static_urls: Optional[Iterable[str]] = None) -> None:
        urls = tuple(static_urls) if static_urls is not None else STATIC_IMAGE_URLS
        if not urls:
            raise ValueError("At least one static image URL must be provided")
        self._urls = urls
        self._url_cycle = itertools.cycle(self._urls)

    @property
    def configured(self) -> bool:
        return True

    async def create_text_to_image(
        self,
        *,
        prompt_text: str,
        ratio: Optional[str] = None,
        model: Optional[str] = None,
        reference_images: Optional[Iterable[Dict[str, str]]] = None,
        seed: Optional[int] = None,
        poll_interval_seconds: float = 3.0,
        max_wait_seconds: int = 120,
    ) -> Dict[str, object]:
        del prompt_text, ratio, model, reference_images, seed, poll_interval_seconds, max_wait_seconds
        image_url = next(self._url_cycle)
        task_id = uuid.uuid4().hex
        return {
            "task_id": task_id,
            "image_url": image_url,
            "output": [image_url],
            "raw": {
                "status": "SUCCEEDED",
                "task_id": task_id,
                "output": [image_url],
                "static": True,
            },
        }

    async def close(self) -> None:  # pragma: no cover - interface compatibility
        return None


runway_image_service = RunwayImageService()