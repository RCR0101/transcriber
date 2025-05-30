import subprocess, sys, shutil
from pathlib import Path
from typing import List

def _ffmpeg_path() -> str:
    """
    Resolve the bundled `ffmpeg` if we’re inside a PyInstaller executable,
    otherwise fall back to whatever is on PATH.
    """
    if getattr(sys, "_MEIPASS", None):
        candidate = Path(sys._MEIPASS) / "ffmpeg"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg") or "ffmpeg"

def extract_wav(src_mp4: Path, dst_wav: Path, *, sample_rate=16000):
    """Convert MP4 (or any container) → mono WAV with FFmpeg."""
    cmd: List[str] = [
        _ffmpeg_path(), "-y",
        "-i", str(src_mp4),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        str(dst_wav),
    ]
    # Run and capture stderr so we can show why it failed
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("FFmpeg stderr ↓↓↓", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)