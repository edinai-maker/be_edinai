"""Async client for Runway text/image-to-image generations."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

import httpx

try:  # Optional SDK import; falls back to HTTP client if unavailable
    from runwayml import RunwayML  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    RunwayML = None

from app.config import get_settings

logger = logging.getLogger(__name__)


class RunwayImageService:
    """Wrapper around Runway's text-to-image (a.k.a. text/image-to-image) endpoint."""

    def __init__(self, *, settings=None, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings or get_settings()
        self._api_key = self._settings.runway_api_key
        self._enabled = bool(self._settings.runway_enabled and self._api_key)
        self._text_to_image_url = self._settings.runway_text_to_image_url
        self._tasks_base_url = self._settings.runway_tasks_base_url.rstrip("/")
        self._api_version = self._settings.runway_api_version
        self._default_model = self._settings.runway_text_to_image_model
        self._default_ratio = self._settings.runway_text_to_image_ratio
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._sdk_client = RunwayML(api_key=self._api_key) if (RunwayML and self._api_key) else None

    @property
    def configured(self) -> bool:
        return self._enabled

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

        resolved_model = model or self._default_model
        resolved_ratio = ratio or self._default_ratio

        if self._sdk_client:
            task_id, status, output_urls, raw = await self._create_with_sdk(
                prompt_text=prompt_text,
                model=resolved_model,
                ratio=resolved_ratio,
                reference_images=reference_images,
                seed=seed,
            )
        else:
            task_id, status, output_urls, raw = await self._create_with_http(
                prompt_text=prompt_text,
                model=resolved_model,
                ratio=resolved_ratio,
                reference_images=reference_images,
                seed=seed,
                poll_interval_seconds=poll_interval_seconds,
                max_wait_seconds=max_wait_seconds,
            )

        if status != "SUCCEEDED":
            raise RuntimeError(f"Runway task {task_id} failed: {raw}")

        if not output_urls:
            raise RuntimeError(f"Runway task {task_id} succeeded without output URLs")

        return {
            "task_id": task_id,
            "image_url": output_urls[0],
            "output": output_urls,
            "raw": raw,
        }

    async def _create_with_sdk(
        self,
        *,
        prompt_text: str,
        model: str,
        ratio: str,
        reference_images: Optional[Sequence[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ) -> tuple[str, str, List[str], Dict[str, Any]]:
        """Use the official Runway SDK (blocking) in a thread to create + wait for output."""
        if not self._sdk_client:
            raise RuntimeError("Runway SDK client is not initialized")

        def _run() -> tuple[str, str, List[str], Dict[str, Any]]:
            task = self._sdk_client.text_to_image.create(
                model=model,
                prompt_text=prompt_text[:1000],
                ratio=ratio,
                seed=seed,
                reference_images=reference_images,
            ).wait_for_task_output()
            output_urls = getattr(task, "output", None) or []
            task_id = getattr(task, "id", None) or getattr(task, "task_id", None) or ""
            status = getattr(task, "status", None) or ""
            raw = task.to_dict() if hasattr(task, "to_dict") else getattr(task, "__dict__", {}) or {}
            return task_id, status, output_urls, raw

        return await asyncio.to_thread(_run)

    async def _create_with_http(
        self,
        *,
        prompt_text: str,
        model: str,
        ratio: str,
        reference_images: Optional[Sequence[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        poll_interval_seconds: float = 3.0,
        max_wait_seconds: int = 120,
    ) -> tuple[str, str, List[str], Dict[str, Any]]:
        """Fallback HTTP client for Runway image generation."""
        payload: Dict[str, Any] = {
            "model": model,
            "promptText": prompt_text[:1000],
            "ratio": ratio,
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

        try:
            response = await self._client.post(self._text_to_image_url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                self._enabled = False
                logger.error("Runway image API returned 401; disabling further image generation attempts")
                raise RuntimeError("Runway image API unauthorized (check RUNWAY_API_KEY)") from exc
            raise
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

        output_urls = last_payload.get("output") or []
        return task_id, status or "", output_urls, last_payload


runway_image_service = RunwayImageService()