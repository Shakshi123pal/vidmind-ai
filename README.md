# VideoRAG 🎬🤖

**CPU-only Video Q&A system using RAG + Gemini Flash 2.5 + Kokoro TTS**

Ask anything about any video. Get text + voice answers.

---

## Architecture

```
Video URL
   │
   ▼
yt-dlp (audio download)
   │
   ▼
faster-whisper (CPU transcription)
   │
   ▼
Semantic Chunker (400 tok, 80 overlap)
   │
   ▼
sentence-transformers/all-MiniLM-L6-v2 (batch embeddings)
   │
   ▼
FAISS IndexFlatIP (persistent per-video index)
   │
   ├──── Query ────────────────────────────────────┐
   │                                               │
User Question → Embed → Top-K Retrieval → Context │
                                                   ▼
                                          Gemini Flash 2.5
                                                   │
                                                   ▼
                                          Kokoro TTS (audio)
                                                   │
                                                   ▼
                                       JSON: answer + chunks + audio
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo>
cd videorag
cp backend/.env.example backend/.env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Docker (recommended)

```bash
# Build and run
GEMINI_API_KEY=your_key_here docker-compose up --build

# Open browser
open http://localhost:8000
```

### 3. Local Development

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `POST /process_video`

Download, transcribe, and index a video.

```bash
curl -X POST http://localhost:8000/process_video \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Response:**
```json
{
  "video_id": "yt_dQw4w9WgXcQ",
  "transcript_length": 12500,
  "num_chunks": 31,
  "message": "Video successfully processed and indexed.",
  "status": "success"
}
```

---

### `POST /ask`

Ask a question about a processed video.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "yt_dQw4w9WgXcQ",
    "question": "What is the main topic of this video?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "This video is about...",
  "retrieved_chunks": [
    {
      "chunk_id": 3,
      "text": "...",
      "start_time": 42.5,
      "end_time": 87.3,
      "similarity_score": 0.892
    }
  ],
  "audio_file_path": "/app/audio_outputs/yt_dQw4w9WgXcQ_abc12345.wav",
  "audio_url": "/audio/yt_dQw4w9WgXcQ_abc12345.wav",
  "video_id": "yt_dQw4w9WgXcQ"
}
```

---

### `GET /status`

```bash
curl http://localhost:8000/status
```

### `DELETE /video/{video_id}`

```bash
curl -X DELETE http://localhost:8000/video/yt_dQw4w9WgXcQ
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Video Download | yt-dlp |
| Transcription | faster-whisper (CPU, int8) |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector DB | FAISS IndexFlatIP |
| LLM | Gemini Flash 2.5 |
| TTS | Kokoro TTS |
| Audio | soundfile, librosa |

---

## Deployment

### GCP Cloud Run

```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT/videorag ./backend

# Push
docker push gcr.io/YOUR_PROJECT/videorag

# Deploy
gcloud run deploy videorag \
  --image gcr.io/YOUR_PROJECT/videorag \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=your_key
```

### Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Upload the `backend/` folder contents to the Space
3. Add `GEMINI_API_KEY` as a Secret in Space settings
4. The Space will auto-build using the Dockerfile

**Note:** Ensure your Space has enough CPU RAM (4GB+ recommended).

---

## Performance Notes

- **Whisper base** model: ~10x realtime on modern CPU (1hr video ≈ 6 min)
- **all-MiniLM-L6-v2**: batch embedding of 100 chunks ≈ 2-3 seconds on CPU
- **FAISS**: sub-millisecond search on CPU for typical video sizes
- **Cold start**: ~30-60 seconds for model loading (pre-baked in Docker image)
- Recommend at least **4GB RAM** for production use

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
    ├── embed.py         ← chunking + sentence-transformers
    ├── rag.py           ← FAISS + Gemini Flash 2.5
    ├── tts.py           ← Kokoro TTS
    ├── utils.py         ← shared utilities
    └── static/
        └── index.html   ← dark UI frontend
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `WHISPER_MODEL` | ❌ | `base` | Whisper model size |
| `TTS_VOICE` | ❌ | `af_heart` | Kokoro voice ID |
| `PORT` | ❌ | `8000` | Server port |

---

## License

MIT
