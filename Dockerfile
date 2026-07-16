# ─────────────────────────────────────────────
#  Stage 1: Python dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries required by OpenCV, InsightFace, MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# ─────────────────────────────────────────────
#  Stage 2: Runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PATH="/root/.local/bin:$PATH"

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Non-root user for security
RUN groupadd --gid 1001 rex && useradd --uid 1001 --gid rex --shell /bin/bash rex
RUN chown -R rex:rex /app
USER rex

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8011/health/live || exit 1

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011", "--workers", "2", "--loop", "uvloop"]
