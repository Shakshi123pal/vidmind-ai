"""
rag.py - RAG pipeline using FAISS for retrieval + configured Gemini model for generation
Persistent FAISS indexes per video
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import time

logger = logging.getLogger("videorag.rag")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Model name can be overridden via the GEMINI_MODEL environment variable.
# Default uses a generic Gemini model identifier; set `GEMINI_MODEL` in
# `backend/.env` or the process environment to a supported model for your account.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")


class RAGPipeline:
    """
    Manages FAISS indexes per video and generates answers via the configured Gemini model.
    
    Index structure per video (stored in index_dir/{video_id}/):
      - index.faiss   : FAISS flat index
      - chunks.pkl    : List of chunk dicts
      - meta.json     : Metadata (url, transcript_length, etc.)
    """

    def __init__(self, index_dir: Path = Path("faiss_indexes")):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict = {}    # video_id -> faiss.Index
        self._chunks: dict = {}     # video_id -> list[dict]
        self._configure_gemini()
        logger.info("Using GEMINI_MODEL=%s", GEMINI_MODEL)
        logger.info(f"RAGPipeline initialized (index_dir={index_dir})")

    def _configure_gemini(self):
        # Read API key at runtime (allows loading .env before import)
        key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not key:
            logger.warning("GEMINI_API_KEY not set. LLM generation will fail.")
            return
        try:
            # Configure the genai client with the provided key
            genai.configure(api_key=key)
            logger.info("Gemini API configured.")
        except Exception as e:
            # Log and continue; downstream code will surface errors if calls fail
            logger.exception("Failed to configure Gemini API client: %s", e)

    def _video_dir(self, video_id: str) -> Path:
        return self.index_dir / video_id

    def _discover_compatible_model(self) -> Optional[str]:
        """Query the Gemini service for available models and pick a compatible one.

        Returns a model name string or None if nothing suitable is found.
        """
        try:
            # Attempt to list models from the SDK. Response shape may vary by SDK version.
            models = None
            if hasattr(genai, "list_models"):
                models = genai.list_models()
            elif hasattr(genai, "Models") and hasattr(genai.Models, "list"):
                models = genai.Models.list()

            if not models:
                logger.debug("No models returned by genai.list_models()")
                return None

            # models may be an iterable of objects or a mapping
            candidates = []
            for m in models:
                # Support dict-like or object-like
                name = None
                supported = None
                try:
                    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
                except Exception:
                    name = None
                try:
                    supported = m.get("supported_methods") if isinstance(m, dict) else getattr(m, "supported_methods", None)
                except Exception:
                    supported = None

                if not name:
                    # Try alternate keys
                    try:
                        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
                    except Exception:
                        name = None

                if name:
                    candidates.append((name, supported))

            # Prefer exact match
            for name, supported in candidates:
                if GEMINI_MODEL and GEMINI_MODEL in name:
                    logger.info(f"Discovered compatible model: {name}")
                    return name

            # Otherwise pick a model that supports generateContent or chat
            for name, supported in candidates:
                if not supported:
                    # Unknown supported list -> return first candidate
                    logger.info(f"Selecting model (unknown capabilities): {name}")
                    return name
                # supported could be a list or comma-separated string
                s = supported if isinstance(supported, (list, tuple)) else str(supported)
                if any(k in str(s).lower() for k in ["generatecontent", "generate", "chat", "text"]):
                    logger.info(f"Selecting supported model: {name} (supports={supported})")
                    return name

            logger.debug("No suitable model found in model list")
            return None
        except Exception:
            logger.exception("Failed to list/inspect Gemini models")
            return None

    def is_indexed(self, video_id: str) -> bool:
        """Check if a video has a FAISS index."""
        if video_id in self._indexes:
            return True
        video_dir = self._video_dir(video_id)
        return (video_dir / "index.faiss").exists()

    def list_indexed_videos(self) -> list[str]:
        """List all indexed video IDs."""
        return [d.name for d in self.index_dir.iterdir() if d.is_dir() and (d / "index.faiss").exists()]

    def get_index_info(self, video_id: str) -> dict:
        """Get metadata for an indexed video."""
        meta_path = self._video_dir(video_id) / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def index_video(
        self,
        video_id: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        metadata: dict
    ):
        """
        Build and persist FAISS index for a video.
        Uses IndexFlatIP (inner product = cosine for normalized vectors).
        """
        if len(chunks) == 0:
            raise ValueError("Cannot index video with empty chunks")

        video_dir = self._video_dir(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Persist
        faiss.write_index(index, str(video_dir / "index.faiss"))
        with open(video_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        
        meta = {**metadata, "num_chunks": len(chunks), "embedding_dim": dim}
        (video_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Cache in memory
        self._indexes[video_id] = index
        self._chunks[video_id] = chunks

        logger.info(f"Indexed {len(chunks)} chunks for video {video_id} (dim={dim})")

    def _load_index(self, video_id: str):
        """Load index from disk into memory cache."""
        video_dir = self._video_dir(video_id)
        self._indexes[video_id] = faiss.read_index(str(video_dir / "index.faiss"))
        with open(video_dir / "chunks.pkl", "rb") as f:
            self._chunks[video_id] = pickle.load(f)
        logger.info(f"Loaded index for {video_id} ({self._indexes[video_id].ntotal} vectors)")

    def retrieve(self, video_id: str, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-K relevant chunks for a query.
        Returns list of chunk dicts with similarity scores.
        """
        if video_id not in self._indexes:
            self._load_index(video_id)

        index = self._indexes[video_id]
        chunks = self._chunks[video_id]

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        logger.info(f"Retrieved {len(results)} chunks (top score: {results[0]['similarity_score']:.3f})" if results else "No chunks retrieved")
        return results

    def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        Generate an answer using the configured Gemini model with retrieved context.
        """
        # Re-read the API key in case .env was loaded at runtime
        key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not key:
            return "Error: GEMINI_API_KEY not configured. Please set the environment variable."

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            time_info = ""
            if chunk.get("start_time") is not None:
                start = int(chunk["start_time"])
                end = int(chunk.get("end_time", start))
                time_info = f" [{start//60}:{start%60:02d} - {end//60}:{end%60:02d}]"
            context_parts.append(f"[Excerpt {i}{time_info}]\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are an expert video content analyst. You have been given excerpts from a video transcript and must answer the user's question in an engaging, insightful, and creative way.

TRANSCRIPT EXCERPTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based on the transcript excerpts provided
- Be conversational, engaging, and informative
- If the answer isn't in the excerpts, say so honestly
- Include specific details and timestamps when relevant
- Keep your answer focused and well-structured
- Aim for 2-4 paragraphs maximum

ANSWER:"""

        # Simple, robust generation path: try configured model, then discover alternatives.
        from google.api_core import exceptions as api_exceptions

        def try_model(model_name: str) -> Optional[str]:
            """Try to generate text with the given model name. Returns answer or None."""
            try:
                # ensure client configured
                try:
                    genai.configure(api_key=key)
                except Exception:
                    logger.debug("genai.configure() call failed or already configured", exc_info=True)

                # Prefer modern GenerativeModel if available
                if hasattr(genai, "GenerativeModel"):
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=1024,
                            top_p=0.9,
                        ),
                    )
                    answer = getattr(response, "text", None)
                    if not answer:
                        candidates_resp = getattr(response, "candidates", None)
                        if candidates_resp and len(candidates_resp) > 0:
                            cand = candidates_resp[0]
                            answer = getattr(cand, "output_text", None) or getattr(cand, "text", None) or getattr(cand, "content", None)
                    if answer:
                        return str(answer).strip()

                # Fallbacks for older SDK variants
                if hasattr(genai, "generate_text"):
                    resp = genai.generate_text(model=model_name, prompt=prompt, temperature=0.7, max_output_tokens=1024, top_p=0.9)
                    answer = getattr(resp, "text", None) or getattr(resp, "output", None)
                    if answer:
                        return str(answer).strip()

                if hasattr(genai, "generate"):
                    resp = genai.generate(model=model_name, prompt=prompt, temperature=0.7, max_output_tokens=1024)
                    answer = getattr(resp, "text", None) or getattr(resp, "output", None)
                    if answer:
                        return str(answer).strip()

                return None
            except api_exceptions.GoogleAPICallError:
                # propagate API call errors to caller for handling (NotFound, ResourceExhausted, etc.)
                raise
            except Exception:
                logger.exception("Error while generating with model %s", model_name)
                return None

        # First try the configured model
        try:
            ans = try_model(GEMINI_MODEL)
            if ans:
                return ans
        except api_exceptions.GoogleAPICallError as gerr:
            msg = str(gerr).lower()
            logger.exception("API error with configured model %s: %s", GEMINI_MODEL, repr(gerr))
            # If model not found, attempt discovery
            if isinstance(gerr, api_exceptions.NotFound) or "not found" in msg:
                logger.info("Configured model %s not found; attempting discovery", GEMINI_MODEL)
                try:
                    alt = self._discover_compatible_model()
                    if alt and alt != GEMINI_MODEL:
                        try:
                            ans = try_model(alt)
                            if ans:
                                logger.info("Generation succeeded with discovered model %s", alt)
                                return ans
                        except api_exceptions.GoogleAPICallError as g2:
                            logger.exception("API error with discovered model %s: %s", alt, repr(g2))
                            # fall through to friendly message below
                except Exception:
                    logger.exception("Model discovery failed")
            # Rate limit or quota errors
            if isinstance(gerr, api_exceptions.ResourceExhausted) or "rate" in msg or "429" in msg or "quota" in msg:
                logger.error("Rate limit or quota hit: %s", repr(gerr))
                return "Too many requests, please wait a moment."
            # Other API errors will fall through to fallback message
        except Exception:
            logger.exception("Unexpected error trying configured Gemini model %s", GEMINI_MODEL)

        # As a last effort, attempt model discovery proactively
        try:
            alt = self._discover_compatible_model()
            if alt and alt != GEMINI_MODEL:
                try:
                    ans = try_model(alt)
                    if ans:
                        logger.info("Generation succeeded with discovered model %s", alt)
                        return ans
                except api_exceptions.GoogleAPICallError as gerr2:
                    msg2 = str(gerr2).lower()
                    logger.exception("API error with discovered model %s: %s", alt, repr(gerr2))
                    if isinstance(gerr2, api_exceptions.ResourceExhausted) or "rate" in msg2 or "429" in msg2 or "quota" in msg2:
                        return "Too many requests, please wait a moment."
        except Exception:
            logger.exception("Model discovery failed during fallback")

        logger.error("All generation attempts exhausted")
        return (
            "Sorry, I couldn't generate an answer right now because the language model service failed. "
            "Please try again in a few moments. If the problem persists, check the server logs or your GEMINI_API_KEY configuration."
        )

    def delete_index(self, video_id: str):
        """Delete a video's FAISS index."""
        import shutil
        video_dir = self._video_dir(video_id)
        if video_dir.exists():
            shutil.rmtree(video_dir)
        self._indexes.pop(video_id, None)
        self._chunks.pop(video_id, None)
        logger.info(f"Deleted index for {video_id}")
