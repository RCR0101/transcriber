from __future__ import annotations
from pathlib import Path
import whisper, time, tqdm

class WhisperEngine:
    """
    Lazily loads a Whisper model once and re-uses it across files.
    """
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self._model = None  # will hold whisper.Whisper object

    def load(self):
        if self._model is None:
            t0 = time.time()
            self._model = whisper.load_model(self.model_size)
            print(f"🔹 Whisper-{self.model_size} loaded in {time.time()-t0:.1f}s")
        return self._model

    def transcribe_wav(self, wav_path: Path) -> str:
        model = self.load()
        result = model.transcribe(
            str(wav_path),
            verbose=False,
            fp16=False,          # CPU-friendly; GPU picked automatically if present
        )
        # Concatenate segments into plain text
        return "\n".join(seg["text"].strip() for seg in result["segments"])