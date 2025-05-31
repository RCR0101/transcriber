# Transcriber GUI

A simple GUI wrapper for OpenAI's Whisper speech recognition model, allowing easy transcription of audio and video files to text.

## Requirements

- Python 3.12 or higher with tkinter installed
  - **Note**: tkinter must be installed with Python. On most systems:
    - Windows: Included by default
    - macOS: Included by default
    - Linux: Install via `sudo apt-get install python3-tk` (Ubuntu/Debian) or equivalent

- FFmpeg installed and available in PATH
  - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
  - macOS: Install via Homebrew: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or equivalent

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/transcriber.git
   cd transcriber
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Development Mode
Run directly with Python:
```bash
python gui.py
```

### Building Standalone Executable
Build a standalone executable using PyInstaller:
```bash
# Windows
pyinstaller gui.spec

# macOS/Linux
python -m PyInstaller gui.spec
```

The executable will be created in the `dist` directory.

## Usage

1. Launch the application
2. Click "Browse" to select an audio/video file (supported formats: mp3, mp4, wav, m4a, mov)
3. Optionally specify an output text file location (defaults to same directory as input)
4. Click "Transcribe" and wait for the process to complete
5. The transcribed text will be saved to the specified output file

## Notes

- First run will download the Whisper model (medium size, ~1.5GB)
- Transcription speed depends on your CPU/GPU capabilities
- The application uses the "medium" model by default for a good balance of accuracy and speed
- Transcription is done locally - no internet connection required after model download

## Troubleshooting

- If you get "FFmpeg not found" error, ensure FFmpeg is properly installed and in your system PATH
- If you get tkinter-related errors, ensure Python was installed with tkinter support
- For GPU support, ensure you have CUDA installed and the correct torch version
