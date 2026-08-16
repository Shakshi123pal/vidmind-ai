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

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
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
                num_workers=1
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

        cookies_path = None
        cookie_candidates = [
        Path("/etc/secrets/cookies.txt"),  # Render Secret File
        Path.cwd() / "cookies.txt",        # Local
        Path(__file__).resolve().parent.parent / "cookies.txt",
        output_dir / "cookies.txt",
        ]
        for candidate in cookie_candidates:
            if candidate.exists():
                cookies_path = candidate
                logger.info(f"Found yt-dlp cookies file: {cookies_path}")
                break

        if cookies_path is None:
            logger.info("No cookies.txt found; continuing without cookies.")

        common_args = [
            "--no-playlist",
        
            # Network stability
            "--socket-timeout", "20",
            "--retries", "1",
            "--fragment-retries", "1",
            "--force-ipv4",

            "--js-runtimes", "deno",
            "--remote-components", "ejs:github",
        
            # Audio
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",
        
            "--output", output_template,
        ]
        commands = []

        def build_cmd(*extra_args: str):
            cmd = ["yt-dlp"] + common_args
            if cookies_path is not None:
                cmd.extend(["--cookies", str(cookies_path)])
            cmd.extend([
                "--extractor-args",
                "youtube:player_client=default,web_embedded"
            ])

            cmd.extend(extra_args)
            cmd.append(url)
            return cmd

        commands.append(
            build_cmd("-f", "bestaudio/best")
        )

        result = None
        last_error = None

        retry_messages = (
            "Sign in to confirm you're not a bot",
            "Requested format is not available",
            "Only images are available for download",
            "YouTube is no longer supported in this application or device",
        )

        for cmd in commands:
            try:
                result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90
                )
            
                logger.info(f"yt-dlp return code: {result.returncode}")
                logger.info(f"yt-dlp stdout: {result.stdout[-2000:]}")
                logger.info(f"yt-dlp stderr: {result.stderr[-4000:]}")
                if result.returncode == 0:
                    break

                last_error = result.stderr.strip()
                retryable = any(msg in last_error for msg in retry_messages)
                if retryable:
                    logger.warning("YouTube extraction path failed; retrying with alternate yt-dlp settings.")
                    continue
                break
            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or ""
                stderr = e.stderr or ""
            
                last_error = (
                    f"yt-dlp command timed out after 90 seconds\n"
                    f"stdout: {stdout[-2000:]}\n"
                    f"stderr: {stderr[-4000:]}"
                )
            
                logger.error(last_error)
                logger.warning("yt-dlp timed out; stopping this attempt.")
                break

        if result is None or result.returncode != 0:
            if last_error and any(msg in last_error for msg in retry_messages):
                raise RuntimeError(
                    "YouTube audio download is unavailable for this video in the current environment. "
                    "The video may be restricted, require browser cookies, or only expose non-audio assets."
                )
            raise RuntimeError(f"yt-dlp failed: {last_error[:500] if last_error else 'unknown yt-dlp error'}")

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
            beam_size=1,
            best_of=1,
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
