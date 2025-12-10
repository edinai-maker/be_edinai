"""FastAPI application factory for the modular backend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from socketio import ASGIApp

from .config import settings
from .database import init_db
from .routes import (
    admin_portal_router,
    auth_router,
    chapter_material_router,
    contact_router,
    dashboard_router,
    lecture_router,
    public_lecture_router,
    registration_router,
    student_portal_router,
    student_management_router,
    student_router,
    superadministration_portal_router,
    teacher_router,
    user_router,
)
from .utils.file_handler import UPLOAD_DIR, ensure_upload_dir, ensure_upload_subdir
from .services.auth_service import ensure_dev_admin_account
from .realtime.socket_server import sio


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    # ----------------------------------
    # CORS CONFIG
    # ----------------------------------
    allow_all_origins = settings.cors_origins == ["*"]

    default_allowed_origins = [
        "https://staticfile-shubhamc080.wasmer.app/examples/mp3.html",
        "https://staticfile-shubhamc080.wasmer.app/",
        "https://edinai.inaiverse.com",
        "https://api.edinai.inaiverse.com",
        "https://xlhyimcoyqyp.id.wasmer.app/",
      
    ]

    cors_origins = ["*"] if allow_all_origins else default_allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------
    # DATABASE + DIRECTORIES
    # ----------------------------------
    init_db()
    ensure_upload_dir()
    ensure_upload_subdir("videos")
    ensure_upload_subdir("static_videos")

    uploads_dir = Path(UPLOAD_DIR).resolve()
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

    # Storage for lecture audio/files
    storage_dir = (Path(__file__).parent.parent / "storage").resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/storage", StaticFiles(directory=storage_dir), name="storage")

    ensure_dev_admin_account()

    # ----------------------------------
    # ROUTERS
    # ----------------------------------
    app.include_router(auth_router)
    app.include_router(admin_portal_router)
    app.include_router(superadministration_portal_router)
    app.include_router(contact_router)
    app.include_router(user_router)
    app.include_router(student_router)
    app.include_router(teacher_router)
    app.include_router(dashboard_router)
    app.include_router(lecture_router)
    app.include_router(public_lecture_router)
    app.include_router(registration_router)
    app.include_router(student_portal_router)
    app.include_router(student_management_router)
    app.include_router(chapter_material_router)

    # Do NOT include contact_router twice!

    # ----------------------------------
    # ROOT HEALTH CHECK
    # ----------------------------------
    @app.get("/", tags=["System"])
    async def root():
        return {"status": True, "message": "Modular backend ready"}

    return app


# ----------------------------------
# SOCKET.IO + FASTAPI COMBINED ASGI APP
# ----------------------------------
fastapi_app = create_app()
app = ASGIApp(sio, fastapi_app)
