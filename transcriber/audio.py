import subprocess
import sys
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    if getattr(sys, "_MEIPASS", None):
        bundled_ffmpeg = Path(sys._MEIPASS) / "ffmpeg"
        if bundled_ffmpeg.exists():
            logger.debug(f"Using bundled ffmpeg: {bundled_ffmpeg}")
            return str(bundled_ffmpeg)

    system_ffmpeg = shutil.which("ffmpeg")
    if not system_ffmpeg:
        raise RuntimeError(
            "FFmpeg is not installed. It's needed to process audio files.\n"
            "Install it with:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: download from https://ffmpeg.org/download.html"
        )

    logger.debug(f"Using system ffmpeg: {system_ffmpeg}")
    return system_ffmpeg


def extract_wav(src_path: Path, dst_path: Path, sample_rate: int = 16000) -> None:
    logger.info(f"Converting {src_path} to WAV")
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-i", str(src_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(dst_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg failed:\n{result.stderr}")
        raise RuntimeError(f"FFmpeg conversion failed with code {result.returncode}")

    logger.info(f"Successfully created WAV file: {dst_path}")
