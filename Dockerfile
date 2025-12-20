FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    # Optional: make ffmpeg path explicit (your app can read this)
    FFMPEG_PATH=/usr/bin/ffmpeg

WORKDIR /app

# =============================
# System deps (runtime + build only where needed)
# =============================
RUN apt-get update && apt-get install --no-install-recommends -y \
    # build deps (needed for some pip wheels)
    build-essential \
    gcc \
    # postgres client libs
    libpq-dev \
    libpq5 \
    # common runtime libs (helpful for opencv / image processing deps)
    libgl1 \
    libglib2.0-0 \
    # pdf + media tooling
    poppler-utils \
    ghostscript \
    qpdf \
    pngquant \
    ffmpeg \
    # misc
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# =============================
# Install python requirements first (cache-friendly)
# =============================
COPY backend/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 # cleanup build tools to reduce image size (keep runtime libs)
 && apt-get purge -y --auto-remove build-essential gcc curl \
 && rm -rf /tmp/requirements.txt

# =============================
# Copy backend source
# =============================
COPY backend ./backend

# =============================
# Non-root user (recommended)
# =============================
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# IMPORTANT:
# - workers=2 for c7i-flex.large (2 vCPU). Increase only if CPU allows.
# - timeout=600 for long tasks; once OCR is async job-based, reduce.
CMD ["gunicorn","app.main:app","-k","uvicorn.workers.UvicornWorker","--workers","1","--worker-connections","1000","--timeout","600","--bind","0.0.0.0:8000","--access-logfile","-","--error-logfile","-"]
