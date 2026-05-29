# Transcriber

Transcribe audio and video files with speaker labels. Runs locally — nothing is sent to the cloud.

Works on **macOS** (Apple Silicon), **Windows**, and **Linux**. Automatically uses the fastest backend for your platform:
- **Apple Silicon Mac** — [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (optimised for M-series chips)
- **Windows / Linux / Intel Mac** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU or NVIDIA GPU)

Speaker identification powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

## What it does

- Transcribes audio/video files (mp3, mp4, wav, m4a, mov, flac, ogg, webm)
- Labels each speaker (Speaker 1, Speaker 2, etc.)
- Auto-detects language (English, Hindi, Hinglish, etc.)
- Can translate any language to English
- Two-stage noise reduction (DeepFilterNet + spectral gating)
- Custom vocabulary to improve recognition of names, jargon, and acronyms
- Outputs as JSON, plain text, or subtitles (SRT/VTT)
- Process one file or an entire folder at once
- Web-based GUI with live preview and synced audio playback

## Quick Start

### 1. Install prerequisites

You need **Python 3.11+** and **FFmpeg**.

<details>
<summary><b>macOS</b></summary>

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and FFmpeg
brew install python ffmpeg
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Download and install Python from [python.org](https://www.python.org/downloads/). **Check "Add Python to PATH"** during install.
2. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH, or install via:
   ```
   winget install FFmpeg
   ```
</details>

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt update
sudo apt install python3 python3-venv ffmpeg
```
</details>

### 2. Get a HuggingFace token (free)

This is needed for speaker identification (who said what). Skip this if you only need the text.

1. Sign up at [huggingface.co](https://huggingface.co) (it's free)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens) and create a token
3. Visit each link below and click **"Agree and access repository"**:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### 3. Set up the project

**macOS / Linux:**
```bash
git clone https://github.com/RCR0101/transcriber.git
cd transcriber
./setup.sh
```

**Windows:**
```
git clone https://github.com/RCR0101/transcriber.git
cd transcriber
setup.bat
```

The setup script creates a virtual environment, installs everything (including the right Whisper backend for your platform), and creates a `.env` file.

After it finishes, open `.env` in any text editor and paste your HuggingFace token:

```
HF_TOKEN=hf_paste_your_token_here
```

### 4. Run it

**macOS / Linux:**
```bash
source .venv/bin/activate
python gui.py
```

**Windows:**
```
.venv\Scripts\activate
python gui.py
```

This opens a web page at `http://localhost:7860`. Upload a file and click **Transcribe**.

> Don't have a HuggingFace token? Uncheck **Speaker Diarization** in the GUI — it will still transcribe, just without speaker labels.

## GUI

The web interface has:

- **Single File** tab — upload one file, see the transcript appear live as it processes
- **Batch** tab — upload multiple files, transcribe them all
- **Settings** panel — paste your HuggingFace token here (or set it in `.env`)
- Toggle speaker diarization, noise reduction, translation, and output format
- Custom vocabulary field for names, acronyms, and jargon
- Click any transcript line to jump to that point in the audio

## Command Line

For power users or scripting:

```bash
# Single file
transcribe recording.mp3

# Choose output format
transcribe recording.mp3 --format srt

# Translate to English
transcribe recording.mp3 --translate

# Skip speaker labels (no token needed)
transcribe recording.mp3 --no-diarize

# Clean up noisy audio before transcribing
transcribe recording.mp3 --denoise

# Help Whisper with specific terms
transcribe recording.mp3 -v "SARC, BITS Pilani, PyTorch"

# Multiple files
transcribe file1.mp3 file2.wav file3.m4a

# Entire folder
transcribe ./recordings/
```

**All options:**

| Flag | What it does |
|------|-------------|
| `-o, --output` | Where to save the output (single file only) |
| `-m, --model` | Which Whisper model to use (auto-detected per platform) |
| `--hf-token` | HuggingFace token (alternative to `.env` file) |
| `--no-diarize` | Skip speaker identification |
| `--translate` | Translate everything to English |
| `--denoise` | Two-stage noise reduction (DeepFilterNet + spectral gating) |
| `-v, --vocabulary` | Comma-separated terms to improve recognition accuracy |
| `--format` | `json`, `txt`, `srt`, or `vtt` (default: `json`) |
| `-q, --quiet` | Less terminal output |

## Output Formats

**JSON** — structured data with speaker labels, timestamps, and full text.

**TXT** — simple readable format:
```
[00:00:05] [SPEAKER_00] Hello, welcome to the interview.
[00:00:08] [SPEAKER_01] Thanks for having me.
```

**SRT / VTT** — subtitle files you can use with video players or upload to YouTube.

## Troubleshooting

**"FFmpeg is not installed"** — Install FFmpeg (see step 1 above) and try again.

**"HF_TOKEN required"** — You need a HuggingFace token for speaker labels. Either add it to `.env`, paste it in the GUI settings, or use `--no-diarize` to skip.

**"403 Forbidden" or "gated repo"** — You need to accept the model terms. Visit the links in step 2 above and click "Agree and access repository".

**First run is slow** — The first time downloads a ~3 GB model. After that it's cached and starts instantly.

**Transcription is wrong** — Try a different model with `-m`. The default is a good balance of speed and accuracy.

## Notes

- All processing happens on your machine. Nothing is uploaded anywhere.
- Works on macOS (Apple Silicon), Windows, and Linux.
- Apple Silicon Macs use mlx-whisper for best performance. All other platforms use faster-whisper with CPU or NVIDIA GPU.
- First run downloads the Whisper model (~3 GB). Runs offline after that.
