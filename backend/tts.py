"""
tts.py - Text-to-Speech using Kokoro TTS (CPU-optimized)
Fallback to pyttsx3 if Kokoro unavailable
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger("videorag.tts")

MAX_TTS_CHARS = 2000  # Truncate very long answers for TTS


class TTSEngine:
    """
    Text-to-speech using Kokoro TTS with pyttsx3 fallback.
    Kokoro provides high-quality, fast, CPU-friendly TTS.
    """

    def __init__(self, output_dir: Path = Path("audio_outputs"), voice: str = "af_heart"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self._engine = None
        self._engine_type = None
        logger.info(f"TTSEngine initialized (output_dir={output_dir})")

    def _load_engine(self):
        """Lazy-load TTS engine with fallback chain."""
        if self._engine is not None:
            return

        # Try Kokoro first
        try:
            from kokoro import KPipeline
            self._engine = KPipeline(lang_code="a")  # 'a' = American English
            self._engine_type = "kokoro"
            logger.info("Kokoro TTS engine loaded successfully.")
            return
        except ImportError:
            logger.warning("Kokoro not available, trying pyttsx3...")
        except Exception as e:
            logger.warning(f"Kokoro failed to load: {e}, trying pyttsx3...")

        # Fallback to pyttsx3
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 175)
            self._engine.setProperty("volume", 0.9)
            self._engine_type = "pyttsx3"
            logger.info("pyttsx3 TTS fallback engine loaded.")
            return
        except Exception as e:
            logger.warning(f"pyttsx3 failed: {e}")

        # Final fallback: silent (gTTS requires internet, skip for offline mode)
        self._engine_type = "silent"
        logger.error("No TTS engine available. Audio will be silent placeholder.")

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS pronunciation."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)          # italic
        text = re.sub(r'`(.+?)`', r'\1', text)            # code
        text = re.sub(r'#{1,6}\s+', '', text)             # headers
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # links
        text = re.sub(r'\n+', ' ', text)                  # newlines
        text = re.sub(r'\s+', ' ', text)                  # whitespace
        # Replace bullet points
        text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    def _truncate_for_tts(self, text: str) -> str:
        """Truncate long texts for TTS (focus on key content)."""
        if len(text) <= MAX_TTS_CHARS:
            return text
        # Find last sentence boundary before limit
        truncated = text[:MAX_TTS_CHARS]
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )
        if last_period > MAX_TTS_CHARS * 0.5:
            truncated = truncated[:last_period + 1]
        return truncated + " ... (response truncated)"

    def synthesize(self, text: str, output_path: str) -> str:
        """
        Convert text to speech and save to output_path.
        Returns the output file path.
        """
        self._load_engine()

        clean_text = self._clean_text_for_tts(text)
        clean_text = self._truncate_for_tts(clean_text)

        if not clean_text:
            logger.warning("Empty text after cleaning, using placeholder")
            clean_text = "No response available."

        output_path = str(output_path)

        try:
            if self._engine_type == "kokoro":
                self._synthesize_kokoro(clean_text, output_path)
            elif self._engine_type == "pyttsx3":
                self._synthesize_pyttsx3(clean_text, output_path)
            else:
                self._create_silent_audio(output_path)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            self._create_silent_placeholder(output_path, str(e))

        logger.info(f"Audio saved: {output_path}")
        return output_path

    def _synthesize_kokoro(self, text: str, output_path: str):
        """Synthesize using Kokoro TTS."""
        import soundfile as sf
        import numpy as np

        audio_chunks = []
        generator = self._engine(text, voice=self.voice, speed=1.0)
        
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            sf.write(output_path, full_audio, 24000)
        else:
            self._create_silent_placeholder(output_path, "Kokoro returned empty audio")

    def _synthesize_pyttsx3(self, text: str, output_path: str):
        """Synthesize using pyttsx3."""
        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()
        
        # pyttsx3 may save in platform-native format; ensure it's accessible
        if not Path(output_path).exists():
            # Try with .wav extension
            wav_path = output_path.replace('.wav', '') + '.wav'
            if not Path(wav_path).exists():
                self._create_silent_placeholder(output_path, "pyttsx3 save failed")

    def _create_silent_placeholder(self, output_path: str, reason: str = ""):
        """Create a minimal valid WAV file as placeholder."""
        import struct
        import wave

        logger.warning(f"Creating silent WAV placeholder. Reason: {reason}")
        
        with wave.open(output_path, 'w') as wav:
            wav.setnchannels(1)     # Mono
            wav.setsampwidth(2)     # 16-bit
            wav.setframerate(22050)
            # 0.5 seconds of silence
            silence = b'\x00\x00' * 22050 // 2
            wav.writeframes(silence)
