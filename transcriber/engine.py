from __future__ import annotations
from pathlib import Path
import whisper
import time
from typing import Optional, Dict, Union
import numpy as np

class WhisperEngine:
    """
    Whisper engine that always translates output to English.
    """
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self._model = None
    
    def load(self) -> whisper.Whisper:
        """Load or get the Whisper model"""
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio: Union[str, np.ndarray]) -> Dict:
        """
        Transcribe audio data or audio file and translate to English.
        
        Args:
            audio: Either a numpy array of audio data or a path to an audio file
            
        Returns:
            Dictionary containing transcription results
        """
        model = self.load()
        
        # Always translate to English
        options = {
            'verbose': False,  # Reduce terminal output
            'word_timestamps': True,
            'fp16': False,  # CPU-friendly
            'task': 'translate',  # Always translate to English
            'best_of': 5,
            'beam_size': 5,
            'patience': 1.0,
            'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'compression_ratio_threshold': 2.4,
            'condition_on_previous_text': True,
        }
        
        # Get English translation
        return model.transcribe(audio, **options)

    def transcribe_wav(self, wav_path: Path) -> str:
        """
        Transcribe audio file and translate to English, returning formatted text.
        
        Args:
            wav_path: Path to WAV file
        
        Returns:
            Transcribed and translated text with timestamps
        """
        result = self.transcribe(str(wav_path))
        
        # Format text with timestamps
        output_lines = []
        for segment in result["segments"]:
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()
            output_lines.append(f"[{timestamp}] {text}")
        
        return "\n".join(output_lines)

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"