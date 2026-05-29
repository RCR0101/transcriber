@echo off
echo === Audio Transcriber Setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed.
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%i
echo Found Python %PYVER%

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo WARNING: FFmpeg is not installed. You'll need it to process audio files.
    echo   Download from: https://ffmpeg.org/download.html
    echo.
) else (
    echo Found FFmpeg
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists
)

echo Installing dependencies (this may take a few minutes)...
.venv\Scripts\pip install -q --upgrade pip
.venv\Scripts\pip install -q -r requirements.txt
.venv\Scripts\pip install -q "faster-whisper>=1.0"

if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo Created .env file from .env.example
    echo IMPORTANT: Edit .env and add your HuggingFace token.
    echo   Get one at: https://huggingface.co/settings/tokens
) else (
    echo .env file already exists
)

echo.
echo === Setup complete! ===
echo.
echo To get started:
echo   1. Activate the environment:  .venv\Scripts\activate
echo   2. Run the GUI:               python gui.py
echo   3. Or use the CLI:            transcribe recording.mp3
echo.
echo If using speaker diarization, make sure to:
echo   - Add your HuggingFace token to .env
echo   - Accept model terms at:
echo     https://huggingface.co/pyannote/speaker-diarization-3.1
echo     https://huggingface.co/pyannote/segmentation-3.0
pause
