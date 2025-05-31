import subprocess, sys, shutil
from pathlib import Path
import numpy as np
import whisper
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def _ffmpeg_path() -> str:
    """
    Resolve the bundled `ffmpeg` if we're inside a PyInstaller executable,
    otherwise fall back to whatever is on PATH.
    """
    if getattr(sys, '_MEIPASS', None):
        candidate = Path(sys._MEIPASS) / "ffmpeg"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg") or "ffmpeg"

def load_audio(file_path: str) -> np.ndarray:
    """
    Load audio file using soundfile and convert to the format whisper expects.
    Falls back to whisper's load_audio if soundfile fails.
    """
    try:
        logger.debug(f"Attempting to load audio file: {file_path}")
        # Convert file path to string and normalize
        file_path = str(Path(file_path).resolve())
        logger.debug(f"Resolved file path: {file_path}")
        
        # Try loading with soundfile first
        try:
            # Load audio file
            data, sample_rate = sf.read(file_path)
            logger.debug(f"Successfully loaded audio with soundfile. Shape: {data.shape}, Sample rate: {sample_rate}")
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                # You might want to add resampling here
                logger.warning("Audio resampling not implemented, falling back to whisper")
                return whisper.load_audio(file_path)
            
            return data
            
        except Exception as e:
            logger.warning(f"Soundfile loading failed, falling back to whisper: {e}")
            return whisper.load_audio(file_path)
            
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise RuntimeError(f"Failed to load audio file: {e}")

def extract_wav(src_mp4: Path, dst_wav: Path, *, sample_rate=16000):
    """Convert MP4 (or any container) → mono WAV with FFmpeg."""
    cmd = [
        _ffmpeg_path(), "-y",
        "-i", str(src_mp4),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(dst_wav),
    ]
    logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
    
    # Run and capture stderr so we can show why it failed
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("FFmpeg stderr ↓↓↓")
            logger.error(proc.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        raise