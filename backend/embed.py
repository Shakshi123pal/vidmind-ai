"""
embed.py - Text chunking and embedding using sentence-transformers
CPU-optimized batch embedding with all-MiniLM-L6-v2
"""

import logging
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("videorag.embed")

CHUNK_SIZE = 400       # tokens ≈ chars / 4
CHUNK_OVERLAP = 80     # overlap in tokens
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


class EmbeddingEngine:
    """
    Handles text chunking and sentence-transformer based embeddings.
    Uses all-MiniLM-L6-v2 (384-dim, fast on CPU).
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        batch_size: int = BATCH_SIZE
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        logger.info(f"EmbeddingEngine initialized (model={model_name}, chunk={chunk_size}/{chunk_overlap})")

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._model.max_seq_length = 256
            logger.info("Embedding model loaded.")
        return self._model

    def chunk_text(self, text: str, segments: Optional[list] = None) -> list[dict]:
        """
        Split transcript into overlapping chunks.
        
        Strategy:
        - If segments available: group segments into chunks respecting token limits
        - Otherwise: character-based sliding window
        
        Returns list of chunk dicts with text, start, end, chunk_id
        """
        if segments and len(segments) > 1:
            return self._chunk_from_segments(segments)
        else:
            return self._chunk_text_sliding(text)

    def _chunk_from_segments(self, segments: list) -> list[dict]:
        """Group transcription segments into semantic chunks."""
        chunks = []
        current_texts = []
        current_start = None
        current_end = None
        current_tokens = 0
        chunk_idx = 0

        # Approximate token count: 4 chars ≈ 1 token
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        for seg in segments:
            seg_text = seg["text"].strip()
            seg_tokens = len(seg_text) // 4 + 1

            if current_start is None:
                current_start = seg.get("start", 0)

            # Flush chunk if adding this segment exceeds limit
            if current_tokens + seg_tokens > self.chunk_size and current_texts:
                chunk_text = " ".join(current_texts).strip()
                if chunk_text:
                    chunks.append({
                        "chunk_id": chunk_idx,
                        "text": chunk_text,
                        "start_time": current_start,
                        "end_time": current_end,
                        "token_count": current_tokens
                    })
                    chunk_idx += 1

                # Overlap: carry over last few segments
                overlap_tokens = 0
                overlap_segs = []
                for t in reversed(current_texts):
                    t_tokens = len(t) // 4 + 1
                    if overlap_tokens + t_tokens <= self.chunk_overlap:
                        overlap_segs.insert(0, t)
                        overlap_tokens += t_tokens
                    else:
                        break

                current_texts = overlap_segs
                current_tokens = overlap_tokens
                current_start = seg.get("start", current_end or 0)

            current_texts.append(seg_text)
            current_tokens += seg_tokens
            current_end = seg.get("end", 0)

        # Flush remaining
        if current_texts:
            chunk_text = " ".join(current_texts).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "start_time": current_start,
                    "end_time": current_end,
                    "token_count": current_tokens
                })

        logger.info(f"Created {len(chunks)} segment-based chunks")
        return chunks

    def _chunk_text_sliding(self, text: str) -> list[dict]:
        """Sliding window chunking on plain text."""
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        chunk_idx = 0
        start = 0

        while start < len(text):
            end = min(start + chars_per_chunk, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                for delim in [". ", "! ", "? ", "\n"]:
                    boundary = text.rfind(delim, start + overlap_chars, end)
                    if boundary != -1:
                        end = boundary + len(delim)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "start_time": None,
                    "end_time": None,
                    "token_count": len(chunk_text) // 4
                })
                chunk_idx += 1

            start = end - overlap_chars
            if start >= len(text):
                break

        logger.info(f"Created {len(chunks)} sliding-window chunks")
        return chunks

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Batch-embed all chunks.
        Returns numpy array of shape (num_chunks, embedding_dim).
        """
        model = self._load_model()
        texts = [c["text"] for c in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks in batches of {self.batch_size}...")
        
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Cosine similarity via dot product
            convert_to_numpy=True
        )
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        model = self._load_model()
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)
