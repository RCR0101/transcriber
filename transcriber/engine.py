from __future__ import annotations

import logging
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

_USE_MLX = sys.platform == "darwin" and platform.machine() == "arm64"

DEFAULT_MODEL_MLX = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL_FASTER = "large-v3-turbo"


def _default_model() -> str:
    return DEFAULT_MODEL_MLX if _USE_MLX else DEFAULT_MODEL_FASTER


class TranscriberEngine:
    def __init__(
        self,
        model_repo: str | None = None,
        hf_token: Optional[str] = None,
    ):
        self.model_repo = model_repo or _default_model()
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._diarization_pipeline = None
        self._faster_model = None

    @property
    def diarization_pipeline(self) -> Pipeline:
        if self._diarization_pipeline is None:
            if not self.hf_token:
                raise ValueError(
                    "HF_TOKEN required for speaker diarization. "
                    "Set the HF_TOKEN environment variable or pass --hf-token."
                )
            logger.info("Loading pyannote speaker-diarization-3.1 pipeline")
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
        return self._diarization_pipeline

    def transcribe_raw(
        self,
        audio_path: str | Path,
        *,
        translate: bool = False,
        vocabulary: str | None = None,
    ) -> dict:
        audio_path = str(Path(audio_path).resolve())
        task = "translate" if translate else "transcribe"
        prompt = vocabulary.strip() if vocabulary else None

        logger.info(f"Transcribing {audio_path} with model {self.model_repo} (task={task})")

        if _USE_MLX:
            return self._transcribe_mlx(audio_path, task, prompt)
        else:
            return self._transcribe_faster(audio_path, task, prompt)

    def _transcribe_mlx(self, audio_path: str, task: str, prompt: str | None = None) -> dict:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_repo,
            word_timestamps=True,
            task=task,
            hallucination_silence_threshold=2.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        logger.info("Transcription complete")
        return result

    def _transcribe_faster(self, audio_path: str, task: str, prompt: str | None = None) -> dict:
        from faster_whisper import WhisperModel

        if self._faster_model is None:
            logger.info(f"Loading faster-whisper model: {self.model_repo}")
            self._faster_model = WhisperModel(
                self.model_repo,
                device="auto",
                compute_type="auto",
            )

        segments_iter, info = self._faster_model.transcribe(
            audio_path,
            task=task,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )

        segments = []
        full_text_parts = []

        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            })
            full_text_parts.append(seg.text)

        logger.info("Transcription complete")
        return {
            "text": "".join(full_text_parts),
            "segments": segments,
            "language": info.language,
        }

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        diarize: bool = True,
        translate: bool = False,
        vocabulary: str | None = None,
        denoise: bool = False,
        on_segment: Callable[[dict], None] | None = None,
    ) -> dict:
        audio_path = str(Path(audio_path).resolve())

        if denoise:
            import tempfile
            from transcriber.denoise import denoise_file
            logger.info("Running two-stage noise reduction")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            denoise_file(audio_path, tmp.name)
            audio_path = tmp.name

        whisper_result = self.transcribe_raw(audio_path, translate=translate, vocabulary=vocabulary)

        if not diarize:
            return self._format_without_speakers(whisper_result, audio_path, on_segment=on_segment)

        return self.diarize(audio_path, whisper_result, on_segment=on_segment)

    def diarize(
        self,
        audio_path: str | Path,
        whisper_result: dict,
        on_segment: Callable[[dict], None] | None = None,
    ) -> dict:
        audio_path = str(Path(audio_path).resolve())

        logger.info("Running speaker diarization")
        diarization_output = self.diarization_pipeline(audio_path)
        if hasattr(diarization_output, "speaker_diarization"):
            diarization = diarization_output.speaker_diarization
        else:
            diarization = diarization_output
        logger.info("Diarization complete, merging results")

        return self._merge(whisper_result, diarization, audio_path, on_segment=on_segment)

    def _merge(self, whisper_result: dict, diarization, audio_path: str,
                on_segment: Callable[[dict], None] | None = None) -> dict:
        words = []
        for segment in whisper_result.get("segments", []):
            for w in segment.get("words", []):
                words.append(w)

        speaker_turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        for word in words:
            word["speaker"] = self._assign_speaker(word, speaker_turns)

        return self._group_into_segments(
            words,
            whisper_result.get("text", ""),
            audio_path,
            on_segment=on_segment,
        )

    def _assign_speaker(self, word: dict, speaker_turns: list) -> str:
        w_start = word.get("start", 0)
        w_end = word.get("end", w_start)

        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        nearest_dist = float("inf")
        nearest_speaker = "UNKNOWN"

        for turn in speaker_turns:
            overlap_start = max(w_start, turn["start"])
            overlap_end = min(w_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

            # track nearest turn as fallback
            dist = min(abs(w_start - turn["end"]), abs(turn["start"] - w_end))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_speaker = turn["speaker"]

        if best_overlap > 0:
            return best_speaker
        return nearest_speaker

    def _group_into_segments(
        self, words: list, full_text: str, audio_path: str,
        on_segment: Callable[[dict], None] | None = None,
    ) -> dict:
        if not words:
            return {
                "source_file": Path(audio_path).name,
                "text": full_text,
                "segments": [],
                "speakers": [],
                "duration": 0.0,
            }

        segments = []
        current_speaker = words[0].get("speaker", "UNKNOWN")
        current_words = [words[0]]

        for word in words[1:]:
            speaker = word.get("speaker", "UNKNOWN")
            if speaker == current_speaker:
                current_words.append(word)
            else:
                seg = self._build_segment(current_speaker, current_words)
                segments.append(seg)
                if on_segment:
                    on_segment(seg)
                current_speaker = speaker
                current_words = [word]

        seg = self._build_segment(current_speaker, current_words)
        segments.append(seg)
        if on_segment:
            on_segment(seg)

        speakers = sorted(set(seg["speaker"] for seg in segments))
        duration = max((w.get("end", 0) for w in words), default=0.0)

        return {
            "source_file": Path(audio_path).name,
            "text": full_text,
            "segments": segments,
            "speakers": speakers,
            "duration": round(duration, 2),
        }

    def _build_segment(self, speaker: str, words: list) -> dict:
        text = " ".join(w.get("word", "").strip() for w in words).strip()
        return {
            "speaker": speaker,
            "start": round(words[0].get("start", 0), 2),
            "end": round(words[-1].get("end", 0), 2),
            "text": text,
        }

    def _format_without_speakers(self, whisper_result: dict, audio_path: str,
                                 on_segment: Callable[[dict], None] | None = None) -> dict:
        segments = []
        for seg in whisper_result.get("segments", []):
            formatted = {
                "speaker": None,
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip(),
            }
            segments.append(formatted)
            if on_segment:
                on_segment(formatted)

        duration = 0.0
        if segments:
            duration = segments[-1]["end"]

        return {
            "source_file": Path(audio_path).name,
            "text": whisper_result.get("text", ""),
            "segments": segments,
            "speakers": [],
            "duration": round(duration, 2),
        }


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _vtt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_srt(result: dict) -> str:
    lines = []
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _srt_timestamp(seg.get("start", 0))
        end = _srt_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker")
        text = f"[{speaker}] {seg['text']}" if speaker else seg["text"]
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.get("segments", []):
        start = _vtt_timestamp(seg.get("start", 0))
        end = _vtt_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker")
        text = f"<v {speaker}>{seg['text']}" if speaker else seg["text"]
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)
