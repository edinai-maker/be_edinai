"""File handling utilities for uploads."""
from __future__ import annotations

import base64
import binascii
import io
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional, Set

from fastapi import HTTPException, UploadFile
from PIL import Image

BASE_BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_BACKEND_DIR / "uploads"))
DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
DEFAULT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".svg", ".webp"}
DEFAULT_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/svg+xml", "image/webp"}

ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_PDF_TYPES = {"application/pdf"}

DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[\w.+\-\/]+);base64,(?P<data>.+)$")
MIMETYPE_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


def _resolve_allowed_extensions(allowed_extensions: Optional[Set[str]]) -> Set[str]:
    if allowed_extensions is not None:
        return allowed_extensions
    return DEFAULT_ALLOWED_EXTENSIONS | ALLOWED_PDF_EXTENSIONS


def _resolve_allowed_types(allowed_types: Optional[Set[str]]) -> Set[str]:
    if allowed_types is not None:
        return allowed_types
    return DEFAULT_ALLOWED_TYPES | ALLOWED_PDF_TYPES


def _save_bytes_to_disk(content: bytes, *, mime_type: str, subfolder: str = "", extension: str) -> dict:
    if len(content) > DEFAULT_MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed size is {DEFAULT_MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    ensure_upload_dir()
    save_dir = UPLOAD_DIR
    if subfolder:
        save_dir = os.path.join(UPLOAD_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)

    saved_filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(save_dir, saved_filename)

    if mime_type.startswith("image/") and mime_type != "image/svg+xml":
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
        except Exception as exc:  # pragma: no cover - PIL-specific error path
            raise HTTPException(status_code=400, detail="Invalid image file") from exc

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "filename": saved_filename,
        "saved_filename": saved_filename,
        "file_path": os.path.relpath(file_path, "."),
        "file_size": len(content),
    }


def ensure_upload_dir() -> None:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

def ensure_upload_subdir(subfolder: str) -> str:
    """Ensure a specific subdirectory inside the upload root exists."""
    if not subfolder:
        return UPLOAD_DIR
    ensure_upload_dir()
    save_dir = os.path.join(UPLOAD_DIR, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def validate_file(
    file: UploadFile,
    *,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_types: Optional[Set[str]] = None,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> bool:
    extensions = allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS
    content_types = allowed_types or DEFAULT_ALLOWED_TYPES

    if hasattr(file, "size") and file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum allowed size is {max_size // (1024 * 1024)}MB",
        )

    if file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed extensions: {', '.join(sorted(extensions))}",
            )

    if file.content_type not in content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(sorted(content_types))}",
        )

    return True


async def save_uploaded_file(
    file: UploadFile,
    subfolder: str = "",
    *,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_types: Optional[Set[str]] = None,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> dict:
    validate_file(
        file,
        allowed_extensions=allowed_extensions,
        allowed_types=allowed_types,
        max_size=max_size,
    )

    ensure_upload_dir()

    save_dir = UPLOAD_DIR
    if subfolder:
        save_dir = os.path.join(UPLOAD_DIR, subfolder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if file.filename:
        file_ext = os.path.splitext(file.filename)[1]
    else:
        file_ext = sorted(allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)[0]
    saved_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(save_dir, saved_filename)

    try:
        content = await file.read()

        if file.content_type.startswith("image/") and file.content_type != "image/svg+xml":
            try:
                image = Image.open(io.BytesIO(content))
                image.verify()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")

        with open(file_path, "wb") as f:
            f.write(content)

        relative_path = os.path.relpath(file_path, ".")

        return {
            "filename": file.filename,
            "saved_filename": saved_filename,
            "file_path": relative_path,
            "file_size": len(content),
        }

    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc


async def save_multiple_files(
    files: List[UploadFile],
    subfolder: str = "",
    *,
    allowed_extensions: Optional[Set[str]] = None,
    allowed_types: Optional[Set[str]] = None,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
) -> List[dict]:
    saved_files: List[dict] = []
    try:
        for upload in files:
            file_info = await save_uploaded_file(
                upload,
                subfolder,
                allowed_extensions=allowed_extensions,
                allowed_types=allowed_types,
                max_size=max_size,
            )
            saved_files.append(file_info)
        return saved_files
    except Exception:
        for info in saved_files:
            path = info["file_path"]
            if os.path.exists(path):
                os.remove(path)
        raise


def delete_file(file_path: str) -> bool:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def get_file_url(file_path: str) -> str:
    url_path = file_path.replace("\\", "/")
    if url_path.startswith("./"):
        url_path = url_path[2:]
    return f"/{url_path}"
