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

## Model Selection and Performance

The application now uses the "small" model by default for optimal speed while maintaining good accuracy. You can modify the model size in `transcriber/engine.py` by changing the `model_size` parameter in the `WhisperEngine` class:

```python
def __init__(self, model_size: str = "small"):  # Change model size here
```

Available models and their characteristics:
- **tiny** (39M parameters): Fastest but lowest accuracy
- **base** (74M parameters): Fast with decent accuracy
- **small** (244M parameters): Default, good balance of speed and accuracy
- **medium** (769M parameters): Higher accuracy but ~2x slower than small
- **large** (1550M parameters): Highest accuracy but significantly slower

To change the model, you can either:
1. Modify the default in `transcriber/engine.py`
2. Pass the model size when creating the engine instance in `gui.py`

Performance Optimizations:
- **GPU Acceleration**: Automatically enabled if CUDA-capable GPU is available
- **Chunked Processing**: Long audio files are automatically processed in chunks
- **Memory Management**: Optimized for handling large files
- **Half Precision**: Automatically enabled on GPU for faster processing

## Notes

- First run will download the selected Whisper model:
  - tiny: ~50MB
  - base: ~150MB
  - small: ~500MB
  - medium: ~1.5GB
  - large: ~3GB
- Transcription speed depends on:
  - Selected model size
  - CPU/GPU capabilities
  - Audio file length
- All processing is done locally - no internet connection required after model download
- For best performance:
  - Use GPU if available (5-10x faster)
  - Use "small" model for good balance of speed/accuracy
  - For very long files, the chunking system prevents memory issues

## Troubleshooting

- If you get "FFmpeg not found" error, ensure FFmpeg is properly installed and in your system PATH
- If you get tkinter-related errors, ensure Python was installed with tkinter support
- For GPU support, ensure you have CUDA installed and the correct torch version
- On Windows:
  - If you get "system cannot find file specified" errors:
    - Try moving files out of the Downloads folder
    - Avoid paths with special characters or non-English characters
    - If using spaces in file paths, the application will handle them automatically
  - If you get "access is denied" errors:
    - Move the executable to a non-system folder (e.g., Documents)
    - Right-click the executable and select "Run as administrator"
    - Check Windows Defender or antivirus settings
    - Make sure you have write permissions in the folder
  - If you get permission errors, try running the application as administrator


## Fully Functional Commits
- Commit **fix: and more** , (9efd22a12baf1b8ca3e912150366652bcb227b31) (It is not very time-friendly)
- Commit **feat: might have made it more efficient**, (7030b9d42ef2fdd53fede5a3e90ce1fda29d10c4) (Faster but may be less accurate)
