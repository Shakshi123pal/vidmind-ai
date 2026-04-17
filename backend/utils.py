"""
utils.py - Shared utility functions for VideoRAG
"""

import re
import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("videorag.utils")


def ensure_dirs(dirs: list):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_video_id(url: str) -> str:
    """
    Extract a stable video ID from URL.
    For YouTube: extract video ID from URL params.
    For others: use MD5 hash of URL.
    """
    url = url.strip()
    
    # YouTube patterns
    yt_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in yt_patterns:
        match = re.search(pattern, url)
        if match:
            return f"yt_{match.group(1)}"
    
    # Vimeo
    vimeo_match = re.search(r'vimeo\.com\/(\d+)', url)
    if vimeo_match:
        return f"vm_{vimeo_match.group(1)}"
    
    # Generic: hash the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    # Use domain as prefix
    try:
        domain = urlparse(url).netloc.replace("www.", "").split(".")[0][:8]
        return f"{domain}_{url_hash}"
    except Exception:
        return f"vid_{url_hash}"


def cleanup_audio(audio_dir: Path, keep_last: int = 50):
    """Delete old audio files, keeping only the most recent N files."""
    audio_dir = Path(audio_dir)
    audio_files = sorted(
        audio_dir.glob("*.wav"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    
    files_to_delete = audio_files[keep_last:]
    for f in files_to_delete:
        try:
            f.unlink()
            logger.debug(f"Deleted old audio: {f.name}")
        except Exception as e:
            logger.warning(f"Could not delete {f}: {e}")
    
    if files_to_delete:
        logger.info(f"Cleaned up {len(files_to_delete)} old audio files")


def sanitize_filename(name: str) -> str:
    """Convert arbitrary string to safe filename."""
    return re.sub(r'[^\w\-_.]', '_', name)[:64]


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
