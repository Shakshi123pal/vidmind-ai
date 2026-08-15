# VideoRAG Dockerfile - CPU-only, production-ready
# Optimized for GCP Cloud Run / Hugging Face Spaces

FROM python:3.11-slim-bookworm

# ─── Labels ───────────────────────────────────────────────────────────────────
LABEL maintainer="VideoRAG"
LABEL description="CPU-only Video Q&A RAG System"

# ─── Build Args ───────────────────────────────────────────────────────────────
ARG DEBIAN_FRONTEND=noninteractive

# ─── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio/Video processing
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    # Build tools
    build-essential \
    cmake \
    git \
    # Networking
    curl \
    wget \
    ca-certificates \
    # espeak for pyttsx3 fallback
    espeak \
    espeak-ng \
    libespeak-ng1 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ─── Python Environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ─── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Install PyTorch CPU first (large, separate layer for cache) ──────────────
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ─── Install Python Dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ─── Application Code ─────────────────────────────────────────────────────────
COPY . .

# ─── Directory Setup ─────────────────────────────────────────────────────────
RUN mkdir -p audio_outputs faiss_indexes temp static

# ─── Copy Frontend ───────────────────────────────────────────────────────────
COPY static/ static/

# ─── Environment Variables ───────────────────────────────────────────────────
ENV HOST=0.0.0.0
ENV WORKERS=1

# ─── Expose Port ─────────────────────────────────────────────────────────────
EXPOSE 10000

# ─── Health Check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# ─── Startup Command ─────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --timeout-keep-alive 75 --log-level info"]
