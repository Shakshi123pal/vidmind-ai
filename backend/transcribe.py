"""
transcribe.py - Video download and transcription using yt-dlp + faster-whisper
CPU-optimized transcription pipeline
"""

import os
import re
import logging
import subprocess
import hashlib 
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("videorag.transcribe")


class VideoTranscriber:
    """
    Downloads video audio and transcribes using faster-whisper on CPU.
    Uses 'base' model for balance of speed and accuracy on CPU.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        logger.info(f"VideoTranscriber initialized (model={model_size}, device={device}, compute={compute_type})")

    def _load_model(self):
        """Lazy-load the whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{self.model_size}'...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2
            )
            logger.info("Whisper model loaded successfully.")
        return self._model

    def download_audio(self, url: str, output_dir: Path) -> str:
        """
        Download audio from video URL using yt-dlp.
        Returns path to downloaded audio file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use URL hash as filename to avoid duplicates
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        output_template = str(output_dir / f"{url_hash}.%(ext)s")
        audio_path = str(output_dir / f"{url_hash}.mp3")

        # Skip if already downloaded
        if Path(audio_path).exists():
            logger.info(f"Audio already cached: {audio_path}")
            return audio_path

        logger.info(f"Downloading audio from: {url}")
        
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "--output", output_template,
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio download timed out (>10 minutes)")

        if not Path(audio_path).exists():
            # Try to find any downloaded file
            candidates = list(output_dir.glob(f"{url_hash}.*"))
            if candidates:
                audio_path = str(candidates[0])
            else:
                raise RuntimeError(f"Audio download failed - no output file found. stderr: {result.stderr[:300]}")

        logger.info(f"Audio downloaded: {audio_path} ({Path(audio_path).stat().st_size / 1024:.1f} KB)")
        return audio_path

    def transcribe_file(self, audio_path: str) -> Tuple[str, list]:
        """
        Transcribe audio file using faster-whisper.
        Returns (full_transcript, segments_list)
        """
        model = self._load_model()
        logger.info(f"Transcribing: {audio_path}")

        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=3,
            best_of=3,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,
            language=None,  # Auto-detect
        )

        logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        segments_list = []
        full_text_parts = []

        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                segments_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text
                })
                full_text_parts.append(text)

        full_transcript = " ".join(full_text_parts)
        logger.info(f"Transcription complete: {len(segments_list)} segments, {len(full_transcript)} chars")
        
        return full_transcript, segments_list

    def transcribe_url(self, url: str, temp_dir: Path = Path("temp")) -> Tuple[str, list]:
        """
        Full pipeline: URL → audio download → transcription.
        Returns (transcript, segments)
        """
        audio_path = self.download_audio(url, temp_dir)
        transcript, segments = self.transcribe_file(audio_path)
        return transcript, segments
