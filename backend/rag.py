"""
rag.py - RAG pipeline using FAISS for retrieval + configured Gemini model for generation
Persistent FAISS indexes per video
"""

import os
import re
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import time
from google import genai as google_genai
from google.genai import types as genai_types

logger = logging.getLogger("videorag.rag")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
# Optional override from environment; keep discovery dynamic and do not hardcode a specific Gemini family.
GEMINI_MODEL = DEFAULT_GEMINI_MODEL


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
        self._chunks: dict = {}    # video_id -> list[dict]
        self._client = None
        self._available_gemini_models: list[str] = []
        self._unavailable_gemini_models: set[str] = set()
        self._configure_gemini()
        self._selected_model = self._discover_compatible_model() or GEMINI_MODEL
        logger.info("Using selected Gemini model=%s", self._selected_model or "<none>")
        logger.info(f"RAGPipeline initialized (index_dir={index_dir})")

    def _configure_gemini(self):
        # Read API key at runtime (allows loading .env before import)
        key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not key:
            logger.warning("GEMINI_API_KEY not set. LLM generation will fail.")
            return
        try:
            self._client = google_genai.Client(api_key=key)
            logger.info("Gemini API configured with google-genai client.")
        except Exception as e:
            # Log and continue; downstream code will surface errors if calls fail
            logger.exception("Failed to configure Gemini API client: %s", e)

    def _video_dir(self, video_id: str) -> Path:
        return self.index_dir / video_id

    def _discover_compatible_model(self) -> Optional[str]:
        """Return the first currently supported Gemini model name for the configured API key."""
        candidates = self._discover_supported_models()
        return candidates[0] if candidates else None

    def _discover_supported_models(self) -> list[str]:
        """List Gemini model names that support content generation via the official google-genai SDK."""
        if self._client is None:
            return []

        try:
            models = list(self._client.models.list())
            candidates: list[str] = []

            for model in models:
                name = getattr(model, "name", None) or getattr(model, "model", None)
                if not name:
                    continue
                if name.startswith("models/"):
                    name = name.split("/", 1)[1]

                supported_actions = getattr(model, "supported_actions", None) or getattr(model, "supported_methods", None)
                if not supported_actions:
                    supported_actions = getattr(model, "supported_generation_methods", None)

                support_text = ""
                if isinstance(supported_actions, (list, tuple, set)):
                    support_text = " ".join(str(item) for item in supported_actions).lower()
                elif supported_actions is not None:
                    support_text = str(supported_actions).lower()

                if any(token in support_text for token in ["generatecontent", "generate", "chat", "text"]):
                    if name not in candidates:
                        candidates.append(name)

            env_model = (DEFAULT_GEMINI_MODEL or GEMINI_MODEL or "").strip()
            if env_model and env_model not in candidates:
                candidates.append(env_model)

            if not candidates:
                logger.debug("No generateContent-capable Gemini models were returned by the API")
                return []

            logger.info("Discovered Gemini generation models: %s", candidates)
            self._available_gemini_models = candidates
            return candidates
        except Exception:
            logger.exception("Failed to list/inspect Gemini models using google-genai")
            return []

    def _is_text_generation_model(self, model_name: str) -> bool:
        """Ignore Gemini preview, audio, image, robotics, embedding, and live-only models."""
        if not model_name:
            return False

        name = model_name.strip().lower()
        if name.startswith("models/"):
            name = name.split("/", 1)[1]

        blocked_tokens = (
            "embedding",
            "tts",
            "audio",
            "image",
            "robotics",
            "live",
            "preview",
            "omni",
        )
        if any(token in name for token in blocked_tokens):
            return False

        if name.startswith("aqa"):
            return False

        return True

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

        prompt = f"""Answer naturally in plain English.

Use the retrieved context to explain the video in a conversational way, as if you watched it and are describing it to someone.
Keep it to about 120-150 words in one paragraph.
Do not use headings, bullets, labels, or markdown.

CONTEXT:
{context}

QUESTION: {question}
"""

        preferred_model = self._selected_model or self._discover_compatible_model() or GEMINI_MODEL
        discovered_models = self._discover_supported_models()
        candidate_models = []
        for model_name in [preferred_model, *discovered_models]:
            if not model_name or model_name in candidate_models:
                continue
            if self._is_text_generation_model(model_name):
                candidate_models.append(model_name)

        def extract_plain_text(payload) -> Optional[str]:
            """Recursively unwrap Gemini SDK response objects to the plain generated text string."""
            if payload is None:
                return None

            if isinstance(payload, str):
                text = payload.strip()
                return text or None

            if isinstance(payload, dict):
                for key in ("text", "output_text", "content", "parts", "candidates"):
                    if key in payload:
                        extracted = extract_plain_text(payload[key])
                        if extracted:
                            return extracted
                return None

            if isinstance(payload, (list, tuple)):
                for item in payload:
                    extracted = extract_plain_text(item)
                    if extracted:
                        return extracted
                return None

            for attr in ("text", "output_text", "content", "parts", "candidates"):
                value = getattr(payload, attr, None)
                extracted = extract_plain_text(value)
                if extracted:
                    return extracted

            return None

        def sanitize_answer(answer: str) -> str:
            """Keep only the final human-readable answer and discard prompt scaffolding."""
            cleaned = str(answer or "").strip()
            if not cleaned:
                return ""

            cleaned = " ".join(cleaned.splitlines()).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            section_markers = (
                "question:",
                "constraint:",
                "constraints:",
                "excerpt:",
                "draft:",
                "word count:",
                "constraint check:",
                "self-correction:",
                "final check:",
                "final polish:",
                "final answer:",
                "answer:",
                "role:",
                "task:",
                "drafting:",
                "self-reflection:",
            )

            last_marker_index = -1
            last_marker = None
            lowered_text = cleaned.lower()
            for marker in section_markers:
                idx = lowered_text.rfind(marker)
                if idx > last_marker_index:
                    last_marker_index = idx
                    last_marker = marker

            if last_marker and last_marker_index != -1:
                cleaned = cleaned[last_marker_index + len(last_marker) :].strip()

            prompt_fragments = (
                "constraint 1:",
                "constraint 2:",
                "constraint 3:",
                "use only the provided transcript excerpts.",
                "answer based only on the excerpts.",
                "do not output prompt text.",
                "do not mention the prompt.",
                "role:",
                "task:",
                "constraints:",
                "drafting:",
                "self-reflection:",
            )
            lowered_cleaned = cleaned.lower()
            for fragment in prompt_fragments:
                if lowered_cleaned.startswith(fragment):
                    cleaned = cleaned[len(fragment) :].strip()
                    lowered_cleaned = cleaned.lower()

            cleaned = re.sub(
                r"^(?:question|constraint|constraints|excerpt|draft|word count|constraint check|self-correction|final check|final polish|role|task|answer|final answer)\s*[:\-]\s*",
                "",
                cleaned,
                flags=re.I,
            ).strip()
            cleaned = re.sub(
                r"\b(?:role|task|constraint|constraints|draft|excerpt|word count|final answer)\b\s*[:\-]\s*",
                "",
                cleaned,
                flags=re.I,
            )

            cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
            cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.M)
            cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+\.\s+)", "", cleaned, flags=re.M)
            cleaned = re.sub(r"^\s*[•\-]\s*", "", cleaned, flags=re.M)
            cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
            cleaned = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", cleaned)
            cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
            cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
            cleaned = re.sub(r"[*_`]+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            if re.search(
                r"(?i)\b(?:question|constraint|constraints|excerpt|draft|word count|self-reflection|final answer|final polish|role|task|transcript)\b\s*[:\-]",
                cleaned,
            ):
                return ""

            if re.search(r"(?i)\b(?:wait, the prompt says|let's think|reasoning)\b", cleaned):
                return ""

            if re.search(r"(?:^|\s)(?:[-*+]|\d+\.|#+)\s", cleaned):
                return ""

            cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()
            if not re.search(r"[.!?]$", cleaned):
                cleaned = f"{cleaned}."

            return cleaned.strip()

        def try_model(model_name: str) -> Optional[str]:
            """Try to generate text with the given model name. Returns answer or None."""
            try:
                if model_name in self._unavailable_gemini_models:
                    logger.info("Skipping cached unavailable Gemini model %s", model_name)
                    return None

                if self._client is None:
                    self._configure_gemini()
                if self._client is None:
                    return None

                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=2048,
                        top_p=0.9,
                    ),
                )

                answer = extract_plain_text(response)
                if answer:
                    final_answer = sanitize_answer(str(answer))
                    if final_answer:
                        word_count = len(re.findall(r"\b\w+\b", final_answer))
                        if 80 <= word_count <= 180:
                            return final_answer
                        logger.info(
                            "Discarded model %s response because it did not fit the accepted plain-English length window.",
                            model_name,
                        )
                        return None
                    logger.info("Discarded prompt-scaffolding response from model %s and will try the next compatible model.", model_name)
                    return None
                return None
            except Exception as exc:
                err_text = str(exc).lower()
                if "404" in err_text or "not found" in err_text or "429" in err_text or "quota" in err_text or "resource_exhausted" in err_text:
                    self._unavailable_gemini_models.add(model_name)
                    raise
                logger.exception("Error while generating with model %s", model_name)
                return None

        for model_name in candidate_models:
            try:
                ans = try_model(model_name)
                if ans:
                    logger.info("Generation succeeded with model %s", model_name)
                    return ans
            except Exception as gerr:
                msg = str(gerr).lower()
                logger.exception("API error with Gemini model %s: %s", model_name, repr(gerr))
                if "404" in msg or "not found" in msg or "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                    logger.info("Model %s unavailable; caching and trying next available Gemini model.", model_name)
                    continue
                if "rate" in msg:
                    logger.error("Rate limit or quota hit: %s", repr(gerr))
                    return "Too many requests, please wait a moment."

        logger.error("All generation attempts exhausted")
        return "Sorry, I couldn't generate a clean answer right now."

    def delete_index(self, video_id: str):
        """Delete a video's FAISS index."""
        import shutil
        video_dir = self._video_dir(video_id)
        if video_dir.exists():
            shutil.rmtree(video_dir)
        self._indexes.pop(video_id, None)
        self._chunks.pop(video_id, None)
        logger.info(f"Deleted index for {video_id}")
