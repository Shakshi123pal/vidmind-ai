"""
VideoRAG - Production-ready Video Q&A System
FastAPI backend with RAG pipeline
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl

from transcribe import VideoTranscriber
from embed import EmbeddingEngine
from dotenv import load_dotenv
from tts import TTSEngine
from utils import ensure_dirs, cleanup_audio, get_video_id

# Load environment variables from backend/.env before importing modules that rely on them
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

# Import RAG after loading env so it can read GEMINI_API_KEY at import time
from rag import RAGPipeline

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("videorag")

# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VideoRAG API",
    description="CPU-optimized Video Q&A with RAG + TTS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directory Setup ──────────────────────────────────────────────────────────
# Base paths (resolve relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
AUDIO_DIR = BASE_DIR / "audio_outputs"
INDEX_DIR = BASE_DIR / "faiss_indexes"
TEMP_DIR = BASE_DIR / "temp"
STATIC_DIR = BASE_DIR / "utilities" / "static"

ensure_dirs([AUDIO_DIR, INDEX_DIR, TEMP_DIR, STATIC_DIR])

# Mount static files for audio playback and frontend
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global Model Cache ───────────────────────────────────────────────────────
_transcriber: Optional[VideoTranscriber] = None
_embedder: Optional[EmbeddingEngine] = None
_rag: Optional[RAGPipeline] = None
_tts: Optional[TTSEngine] = None


def get_transcriber() -> VideoTranscriber:
    global _transcriber
    if _transcriber is None:
        logger.info("Initializing transcriber...")
        _transcriber = VideoTranscriber()
    return _transcriber


def get_embedder() -> EmbeddingEngine:
    global _embedder
    if _embedder is None:
        logger.info("Initializing embedding engine...")
        _embedder = EmbeddingEngine()
    return _embedder


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        logger.info("Initializing RAG pipeline...")
        _rag = RAGPipeline(index_dir=INDEX_DIR)
    return _rag


def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        logger.info("Initializing TTS engine...")
        _tts = TTSEngine(output_dir=AUDIO_DIR)
    return _tts


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class ProcessVideoRequest(BaseModel):
    video_url: str
    force_reprocess: bool = False


class ProcessVideoResponse(BaseModel):
    video_id: str
    transcript_length: int
    num_chunks: int
    message: str
    status: str


class AskRequest(BaseModel):
    video_id: str
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[dict]
    audio_file_path: str
    audio_url: str
    video_id: str


class StatusResponse(BaseModel):
    status: str
    indexed_videos: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_path = STATIC_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return JSONResponse({"message": "VideoRAG API is running. Use /docs for API reference."})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "VideoRAG"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and indexed videos."""
    rag = get_rag()
    indexed = rag.list_indexed_videos()
    return StatusResponse(status="ready", indexed_videos=indexed)


@app.post("/process_video", response_model=ProcessVideoResponse)
async def process_video(request: ProcessVideoRequest):
    """
    Download, transcribe, chunk, embed, and index a video.
    
    Steps:
    1. Download audio from video URL
    2. Transcribe with faster-whisper
    3. Chunk transcript semantically
    4. Generate embeddings
    5. Store in FAISS index
    """
    video_url = request.video_url
    video_id = get_video_id(video_url)
    
    logger.info(f"Processing video: {video_id} from {video_url}")
    
    # Check if already indexed
    rag = get_rag()
    if not request.force_reprocess and rag.is_indexed(video_id):
        index_info = rag.get_index_info(video_id)
        return ProcessVideoResponse(
            video_id=video_id,
            transcript_length=index_info.get("transcript_length", 0),
            num_chunks=index_info.get("num_chunks", 0),
            message="Video already processed and indexed. Use force_reprocess=true to reindex.",
            status="cached"
        )
    
    try:
        # Step 1 & 2: Download and transcribe
        transcriber = get_transcriber()
        logger.info(f"Transcribing video {video_id}...")
        transcript, segments = transcriber.transcribe_url(video_url, temp_dir=TEMP_DIR)
        
        if not transcript:
            raise HTTPException(status_code=422, detail="Failed to transcribe video - empty transcript")
        
        logger.info(f"Transcript length: {len(transcript)} chars")
        
        # Step 3: Chunk transcript
        embedder = get_embedder()
        chunks = embedder.chunk_text(transcript, segments=segments)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4 & 5: Embed and index
        embeddings = embedder.embed_chunks(chunks)
        rag.index_video(
            video_id=video_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata={
                "url": video_url,
                "transcript_length": len(transcript),
                "num_chunks": len(chunks)
            }
        )
        
        logger.info(f"Video {video_id} fully processed and indexed")
        
        return ProcessVideoResponse(
            video_id=video_id,
            transcript_length=len(transcript),
            num_chunks=len(chunks),
            message="Video successfully processed and indexed.",
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, background_tasks: BackgroundTasks):
    """
    Ask a question about a processed video.
    
    Steps:
    1. Embed the question
    2. Retrieve top-K relevant chunks from FAISS
    3. Send context + question to Gemini Flash 2.5
    4. Convert answer to speech with Kokoro TTS
    5. Return answer + audio
    """
    video_id = request.video_id
    question = request.question
    top_k = min(request.top_k, 10)  # Cap at 10
    
    logger.info(f"Question for video {video_id}: {question[:100]}...")
    
    # Validate video is indexed
    rag = get_rag()
    if not rag.is_indexed(video_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{video_id}' not found. Please process the video first via /process_video"
        )
    
    try:
        # Step 1: Embed question
        embedder = get_embedder()
        q_embedding = embedder.embed_query(question)
        
        # Step 2: Retrieve chunks
        retrieved_chunks = rag.retrieve(video_id, q_embedding, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 3: Generate answer with Gemini
        answer = rag.generate_answer(question, retrieved_chunks)
        logger.info(f"Generated answer: {answer[:100]}...")
        
        # Step 4: TTS
        tts = get_tts()
        audio_filename = f"{video_id}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = AUDIO_DIR / audio_filename
        tts.synthesize(answer, str(audio_path))
        
        # Schedule cleanup of old audio files (keep last 50)
        background_tasks.add_task(cleanup_audio, AUDIO_DIR, keep_last=50)
        
        audio_url = f"/audio/{audio_filename}"
        
        return AskResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            audio_file_path=str(audio_path),
            audio_url=audio_url,
            video_id=video_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question for {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


@app.delete("/video/{video_id}")
async def delete_video_index(video_id: str):
    """Delete a video's FAISS index."""
    rag = get_rag()
    if not rag.is_indexed(video_id):
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    rag.delete_index(video_id)
    return {"message": f"Index for video '{video_id}' deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
