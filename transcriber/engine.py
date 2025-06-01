from __future__ import annotations
from pathlib import Path
import whisper
import time
from typing import Optional, Dict, Union
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

class WhisperEngine:
    """Whisper engine that handles transcription and translation to English."""
    
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self._model = None
        self.chunk_size = 24 * 60  # 24 minutes chunks
        logger.info(f"Initializing WhisperEngine with model size: {model_size}")
        
    @property
    def model(self) -> whisper.Whisper:
        """Lazy load the Whisper model"""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper model on device: {device}")
            self._model = whisper.load_model(self.model_size).to(device)
            if device == "cuda":
                self._model = self._model.half()
        return self._model

    def get_transcription_options(self) -> dict:
        """Get optimized transcription options based on device"""
        device = next(self.model.parameters()).device
        return {
            'task': 'translate',
            'word_timestamps': True,
            'best_of': 3,
            'beam_size': 3,
            'temperature': [0.0, 0.2, 0.4],
            'condition_on_previous_text': True,
            'fp16': device.type == "cuda"
        }

    def transcribe(self, audio: Union[str, np.ndarray]) -> Dict:
        """Transcribe audio and translate to English."""
        try:
            # Load audio if path provided
            if isinstance(audio, (str, Path)):
                logger.info(f"Loading audio from: {audio}")
                audio = whisper.load_audio(str(audio))
                logger.debug(f"Audio loaded, length: {len(audio)/16000:.2f} seconds")

            # Process in chunks if audio is long
            if len(audio) > self.chunk_size * 16000:
                return self._process_long_audio(audio)
            
            # Process normally for shorter audio
            logger.info("Starting transcription")
            result = self.model.transcribe(audio, **self.get_transcription_options())
            logger.info("Transcription complete")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Transcription failed: {str(e)}") from e

    def _process_long_audio(self, audio: np.ndarray) -> Dict:
        """Process long audio in chunks"""
        logger.info("Processing long audio in chunks")
        chunk_length = self.chunk_size * 16000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
        
        results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            chunk_result = self.model.transcribe(chunk, **self.get_transcription_options())
            
            # Adjust timestamps for chunks after the first
            if i > 1:
                for segment in chunk_result["segments"]:
                    segment["start"] += (i-1) * self.chunk_size
                    segment["end"] += (i-1) * self.chunk_size
            
            results.append(chunk_result)
        
        # Merge results
        final_result = results[0]
        final_result["text"] = " ".join(r["text"] for r in results)
        final_result["segments"] = [s for r in results for s in r["segments"]]
        
        logger.info("Long audio processing complete")
        return final_result

    def transcribe_wav(self, wav_path: Path) -> str:
        """Transcribe audio file and return formatted text with timestamps."""
        result = self.transcribe(str(wav_path))
        return "\n".join(
            f"[{format_timestamp(segment['start'])}] {segment['text'].strip()}"
            for segment in result["segments"]
        )

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"