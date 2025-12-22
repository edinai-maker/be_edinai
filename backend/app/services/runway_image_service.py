"""Async client for Runway text/image-to-image generations."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class RunwayImageService:
    """Wrapper around Runway's text-to-image (a.k.a. text/image-to-image) endpoint."""

    def __init__(self, *, settings=None, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings or get_settings()
        self._api_key = self._settings.runway_api_key
        self._text_to_image_url = self._settings.runway_text_to_image_url
        self._tasks_base_url = self._settings.runway_tasks_base_url.rstrip("/")
        self._api_version = self._settings.runway_api_version
        self._default_model = self._settings.runway_text_to_image_model
        self._default_ratio = self._settings.runway_text_to_image_ratio
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def close(self) -> None:
        await self._client.aclose()

    async def create_text_to_image(
        self,
        *,
        prompt_text: str,
        ratio: Optional[str] = None,
        model: Optional[str] = None,
        reference_images: Optional[Sequence[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        poll_interval_seconds: float = 3.0,
        max_wait_seconds: int = 120,
    ) -> Dict[str, Any]:
        """Submit an image generation task and wait for its output."""
        if not self.configured:
            raise RuntimeError("Runway API is not configured")

        payload: Dict[str, Any] = {
            "model": model or self._default_model,
            "promptText": prompt_text[:1000],
            "ratio": ratio or self._default_ratio,
        }

        if reference_images:
            normalized_refs: List[Dict[str, Any]] = []
            for ref in reference_images:
                uri = (ref or {}).get("uri")
                if not uri:
                    continue
                entry = {"uri": uri}
                if ref.get("tag"):
                    entry["tag"] = ref["tag"]
                normalized_refs.append(entry)
            if normalized_refs:
                payload["referenceImages"] = normalized_refs[:3]

        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Runway-Version": self._api_version,
        }

        response = await self._client.post(self._text_to_image_url, json=payload, headers=headers)
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

        return {
            "task_id": task_id,
            "image_url": output_urls[0],
            "output": output_urls,
            "raw": last_payload,
        }


runway_image_service = RunwayImageService()
