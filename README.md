# VidMind AI

Professional Video Q&A — retrieve concise, sourced answers from any video with optional synthesized audio.

VidMind AI provides a scalable CPU-friendly RAG pipeline that converts video audio into searchable knowledge, retrieves relevant context, and generates clean, voice-enabled answers. It's designed for local development, containerized deployments, and lightweight production environments.

---

## Features

- Fast CPU transcription using optimized Whisper variants for cost-effective processing
- Semantic chunking + vector search (FAISS) for precise context retrieval
- Configurable embedding models and LLM backend (Gemini-compatible by default)
- Generated answers paired with source chunks and timestamps for auditability
- Optional TTS output for audio replies (Kokoro or configurable TTS)
- Simple REST API for ingestion, query, and management
- Docker-ready for quick deployment and reproducible builds

---

## Architecture

```
Video URL
   │
   ▼
yt-dlp (audio download)
   │
   ▼
Transcription (faster-whisper / Whisper)
   │
   ▼
Semantic Chunker (configurable tokens + overlap)
   │
   ▼
Embeddings (sentence-transformers or configurable)
   │
   ▼
FAISS (persistent per-video index)
   │
   ├──── Query ────────────────────────────────────┐
   │                                               │
User Question → Embed → Top-K Retrieval → Context │
                                                   ▼
                                       LLM (Gemini or configured model)
                                                   │
                                                   ▼
                                          Optional TTS (audio)
                                                   │
                                                   ▼
                                       JSON: answer + sources + audio
```

---

## Quick Start

1) Clone and configure

```bash
git clone <your-repo-url>
cd vidmind
cp backend/.env.example backend/.env
# Edit backend/.env and set GEMINI_API_KEY (and other variables as needed)
```

2) Docker (recommended)

```bash
# Build and run with environment key
GEMINI_API_KEY=your_key_here docker-compose up --build

# Open the UI or API at http://localhost:8000
```

3) Local development

```bash
cd backend
pip install -r requirements.txt
# On Windows PowerShell
$Env:GEMINI_API_KEY = 'your_key_here'
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### POST /process_video

Download, transcribe, and index a video (returns `video_id` and processing status).

```bash
curl -X POST http://localhost:8000/process_video \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### POST /ask

Ask a question about a processed video. Returns an answer, retrieved source chunks, and optional audio asset metadata.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "yt_dQw4w9WgXcQ",
    "question": "What is the main topic of this video?",
    "top_k": 5
  }'
```

### GET /status

```bash
curl http://localhost:8000/status
```

### DELETE /video/{video_id}

```bash
curl -X DELETE http://localhost:8000/video/yt_dQw4w9WgXcQ
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Video Download | yt-dlp |
| Transcription | faster-whisper / Whisper (CPU-optimized) |
| Embeddings | sentence-transformers (configurable) |
| Vector DB | FAISS |
| LLM | Gemini-compatible (configurable) |
| TTS | Kokoro TTS (optional) |
| Audio | soundfile, librosa |

---

## Deployment

### Container (GCP Cloud Run / any Docker host)

```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT/vidmind ./backend

# Push
docker push gcr.io/YOUR_PROJECT/vidmind

# Deploy (example for Cloud Run)
gcloud run deploy vidmind \
  --image gcr.io/YOUR_PROJECT/vidmind \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=your_key
```

---

## Performance Notes

- Transcription: optimized Whisper variants can process long audio on CPU with acceptable throughput
- Embedding batching reduces overhead for large videos
- FAISS provides low-latency retrieval for typical per-video index sizes
- Recommend 4GB+ RAM for small production workloads; scale resources for high concurrency

---

## Project Structure

```
.
├── docker-compose.yml
├── README.md
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── .env.example
    ├── main.py          ← FastAPI app + endpoints
    ├── transcribe.py    ← yt-dlp + faster-whisper
    ├── embed.py         ← chunking + embeddings
    ├── rag.py           ← retrieval + generation pipeline
    ├── tts.py           ← Kokoro TTS integration
    ├── utils.py         ← shared utilities
    └── static/
        └── index.html   ← UI frontend
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Gemini / LLM API key |
| `WHISPER_MODEL` | ❌ | `base` | Whisper model size |
| `TTS_VOICE` | ❌ | `af_heart` | Kokoro voice ID |
| `PORT` | ❌ | `8000` | Server port |

---

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests, and follow standard GitHub workflows for pull requests. Keep changes focused and include tests where practical.

---

## License

MIT
