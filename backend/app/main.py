"""FastAPI application factory for the modular backend."""
from __future__ import annotations
import time
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from socketio import ASGIApp
# :white_check_mark: PROMETHEUS IMPORTS (ADDED)
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
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
    vision_router,
)
from .utils.file_handler import UPLOAD_DIR, ensure_upload_dir, ensure_upload_subdir
from .services.auth_service import ensure_dev_admin_account
from .realtime.socket_server import sio
from .routes.chapter_material_routes import chapter_material_http_exception_handler
# ----------------------------------
# PROMETHEUS METRICS (ADDED)
# ----------------------------------
HTTP_REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["path"],
)
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
    # PROMETHEUS MIDDLEWARE (ADDED)
    # ----------------------------------
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        start_time = time.time()
        try:
            response = await call_next(request)
        except HTTPException as exc:
            if request.url.path.startswith("/chapter-materials"):
                response = await chapter_material_http_exception_handler(request, exc)
            else:
                raise
        response = await call_next(request)
        duration = time.time() - start_time
        HTTP_REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        ).inc()
        HTTP_REQUEST_LATENCY.labels(
            path=request.url.path
        ).observe(duration)
        return response
    # ----------------------------------
    # DATABASE + DIRECTORIES
    # ----------------------------------
    init_db()
    ensure_upload_dir()
    ensure_upload_subdir("videos")
    ensure_upload_subdir("static_videos")
    uploads_dir = Path(UPLOAD_DIR).resolve()
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
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
    app.include_router(vision_router)
    # ----------------------------------
    # ROOT + HEALTH
    # ----------------------------------
    @app.get("/", tags=["System"])
    async def root():
        return {"status": True, "message": "Modular backend ready"}
    @app.get("/health", tags=["System"])
    def health():
        return {"status": "ok"}
    # ----------------------------------
    # PROMETHEUS METRICS ENDPOINT (ADDED)
    # ----------------------------------
    @app.get("/metrics", tags=["Monitoring"])
    def metrics():
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    return app
# ----------------------------------
# SOCKET.IO + FASTAPI COMBINED ASGI APP
# ----------------------------------
fastapi_app = create_app()
app = ASGIApp(sio, fastapi_app)