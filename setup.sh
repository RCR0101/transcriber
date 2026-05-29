#!/bin/bash
set -e

echo "=== Audio Transcriber Setup ==="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Install it from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "Found FFmpeg"
else
    echo "WARNING: FFmpeg is not installed. You'll need it to process audio files."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with: brew install ffmpeg"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "  Install with: sudo apt install ffmpeg"
    else
        echo "  Download from: https://ffmpeg.org/download.html"
    fi
    echo ""
fi

# Detect platform
ARCH=$(python3 -c "import platform; print(platform.machine())")
OS=$(python3 -c "import sys; print(sys.platform)")

if [[ "$OS" == "darwin" && "$ARCH" == "arm64" ]]; then
    WHISPER_BACKEND="mlx-whisper"
    echo "Detected Apple Silicon — will use mlx-whisper"
else
    WHISPER_BACKEND="faster-whisper>=1.0"
    echo "Detected $OS/$ARCH — will use faster-whisper"
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

echo "Installing dependencies (this may take a few minutes)..."
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q "$WHISPER_BACKEND"

# Set up .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Created .env file from .env.example"
    echo "IMPORTANT: Edit .env and add your HuggingFace token."
    echo "  Get one at: https://huggingface.co/settings/tokens"
else
    echo ".env file already exists"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To get started:"
echo "  1. Activate the environment:  source .venv/bin/activate"
echo "  2. Run the GUI:               python gui.py"
echo "  3. Or use the CLI:            transcribe recording.mp3"
echo ""
echo "If using speaker diarization, make sure to:"
echo "  - Add your HuggingFace token to .env"
echo "  - Accept model terms at:"
echo "    https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "    https://huggingface.co/pyannote/segmentation-3.0"
