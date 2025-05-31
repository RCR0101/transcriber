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
    """
    Whisper engine that always translates output to English.
    """
    def __init__(self, model_size: str = "small"):  # Changed default to small
        self.model_size = model_size
        self._model = None
        self.chunk_size = 24 * 60  # 24 minutes chunks for better memory management
    
    def load(self) -> whisper.Whisper:
        """Load or get the Whisper model"""
        if self._model is None:
            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = whisper.load_model(self.model_size).to(device)
            
            if device == "cuda":
                self._model = self._model.half()  # Use half precision for GPU
                
        return self._model

    def _chunk_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> list[np.ndarray]:
        """Split audio into chunks for more efficient processing"""
        chunk_length = self.chunk_size * sample_rate
        return [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

    def transcribe(self, audio: Union[str, np.ndarray]) -> Dict:
        """
        Transcribe audio data or audio file and translate to English.
        
        Args:
            audio: Either a numpy array of audio data or a path to an audio file
            
        Returns:
            Dictionary containing transcription results
        """
        model = self.load()
        device = next(model.parameters()).device
        
        # Optimize options based on device
        options = {
            'verbose': False,
            'word_timestamps': True,
            'task': 'translate',
            'best_of': 3,  # Reduced from 5
            'beam_size': 3,  # Reduced from 5
            'patience': 1.0,
            'temperature': [0.0, 0.2, 0.4, 0.6],  # Reduced temperature range
            'compression_ratio_threshold': 2.4,
            'condition_on_previous_text': True,
            'fp16': device.type == "cuda"  # Use fp16 only on GPU
        }
        
        try:
            # Load audio if path provided
            if isinstance(audio, (str, Path)):
                logger.debug(f"Loading audio from file: {audio}")
                audio = whisper.load_audio(str(audio))
                logger.debug("Audio loaded successfully")
            
            # Process in chunks for long audio
            if len(audio) > self.chunk_size * 16000:  # If longer than chunk size
                logger.debug("Processing long audio in chunks")
                chunks = self._chunk_audio(audio)
                results = []
                
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    chunk_result = model.transcribe(chunk, **options)
                    
                    # Adjust timestamps for chunks after the first
                    if i > 0:
                        for segment in chunk_result["segments"]:
                            segment["start"] += i * self.chunk_size
                            segment["end"] += i * self.chunk_size
                    
                    results.append(chunk_result)
                
                # Merge results
                final_result = results[0]
                for result in results[1:]:
                    final_result["segments"].extend(result["segments"])
                    final_result["text"] += " " + result["text"]
                
                return final_result
            
            # For shorter audio, process normally
            return model.transcribe(audio, **options)
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

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