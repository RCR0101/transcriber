import subprocess, sys, shutil
from pathlib import Path
import numpy as np
import whisper
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def get_ffmpeg_path() -> str:
    """Get path to ffmpeg executable, using bundled version if available."""
    if getattr(sys, '_MEIPASS', None):
        bundled_ffmpeg = Path(sys._MEIPASS) / "ffmpeg"
        if bundled_ffmpeg.exists():
            logger.debug(f"Using bundled ffmpeg: {bundled_ffmpeg}")
            return str(bundled_ffmpeg)
    
    system_ffmpeg = shutil.which("ffmpeg")
    if not system_ffmpeg:
        raise RuntimeError("ffmpeg not found in system PATH")
    
    logger.debug(f"Using system ffmpeg: {system_ffmpeg}")
    return system_ffmpeg

def load_audio(file_path: str) -> np.ndarray:
    """Load and normalize audio file for transcription."""
    try:
        file_path = str(Path(file_path).resolve())
        logger.info(f"Loading audio file: {file_path}")
        
        # Try soundfile first for better performance
        try:
            data, sample_rate = sf.read(file_path)
            logger.debug(f"Loaded audio: {data.shape}, {sample_rate}Hz")
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                logger.debug("Converted stereo to mono")
            
            # For now, fall back to whisper if resampling needed
            if sample_rate != 16000:
                logger.info("Sample rate not 16kHz, using whisper loader")
                return whisper.load_audio(file_path)
            
            return data
            
        except Exception as e:
            logger.info(f"Soundfile failed, using whisper loader: {e}")
            return whisper.load_audio(file_path)
            
    except Exception as e:
        logger.error(f"Failed to load audio: {e}", exc_info=True)
        raise RuntimeError(f"Could not load audio file: {e}") from e

def extract_wav(src_path: Path, dst_path: Path, sample_rate: int = 16000) -> None:
    """Convert audio/video file to WAV format using FFmpeg."""
    try:
        logger.info(f"Converting {src_path} to WAV")
        cmd = [
            get_ffmpeg_path(),
            "-y",  # Overwrite output
            "-i", str(src_path),
            "-ac", "1",  # Mono
            "-ar", str(sample_rate),
            "-f", "wav",
            str(dst_path)
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed with code {result.returncode}")
        
        logger.info(f"Successfully created WAV file: {dst_path}")
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}", exc_info=True)
        raise RuntimeError(f"Could not convert audio: {e}") from e