import json
import logging
import os
import pathlib
import shutil
import tempfile

import gradio as gr
from dotenv import load_dotenv

from transcriber.audio import extract_wav
from transcriber.engine import (
    TranscriberEngine,
    format_srt,
    format_timestamp,
    format_vtt,
)

load_dotenv()
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".flac", ".ogg", ".webm"}


def _check_ffmpeg() -> str | None:
    if shutil.which("ffmpeg"):
        return None
    return (
        "FFmpeg is not installed. It's required to process audio files.\n\n"
        "Install it with:\n"
        "  macOS:  brew install ffmpeg\n"
        "  Ubuntu: sudo apt install ffmpeg\n"
        "  Windows: download from https://ffmpeg.org/download.html"
    )


def _resolve_token(hf_token: str, diarize: bool) -> str | None:
    token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN", "")
    if diarize and not token:
        return None
    return token or None


def _format_segment_line(seg: dict) -> str:
    ts = format_timestamp(seg["start"])
    speaker = seg.get("speaker")
    prefix = f"[{speaker}] " if speaker else ""
    return f"[{ts}] {prefix}{seg['text']}"


def _format_segment_html(seg: dict, index: int) -> str:
    ts = format_timestamp(seg["start"])
    speaker = seg.get("speaker")
    speaker_html = f'<span style="color:#6366f1;font-weight:600">[{speaker}]</span> ' if speaker else ""
    seconds = seg["start"]
    return (
        f'<div class="seg" data-time="{seconds}" data-idx="{index}" '
        f'style="padding:6px 8px;margin:2px 0;border-radius:4px;cursor:pointer;'
        f'transition:background 0.15s" '
        f'onmouseenter="this.style.background=\'#f1f5f9\'" '
        f'onmouseleave="this.style.background=\'transparent\'" '
        f'onclick="seekAudio({seconds})">'
        f'<span style="color:#94a3b8;font-size:0.85em;margin-right:8px">[{ts}]</span>'
        f'{speaker_html}{seg["text"]}'
        f'</div>'
    )


def _write_output(result: dict, output_path: pathlib.Path, fmt: str) -> None:
    if fmt == "json":
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt == "srt":
        output_path.write_text(format_srt(result), encoding="utf-8")
    elif fmt == "vtt":
        output_path.write_text(format_vtt(result), encoding="utf-8")
    else:
        lines = [_format_segment_line(seg) for seg in result["segments"]]
        output_path.write_text("\n".join(lines), encoding="utf-8")


def _transcribe_file(engine, input_path: pathlib.Path, diarize: bool, translate: bool,
                     vocabulary: str | None = None, denoise: bool = False):
    ffmpeg_err = _check_ffmpeg()
    if ffmpeg_err and input_path.suffix.lower() != ".wav":
        raise gr.Error(ffmpeg_err)

    if input_path.suffix.lower() != ".wav":
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = pathlib.Path(tmpdir) / "audio.wav"
            extract_wav(input_path, wav_path)
            yield from _run_transcribe(engine, wav_path, input_path.name, diarize, translate, vocabulary, denoise)
    else:
        yield from _run_transcribe(engine, input_path, input_path.name, diarize, translate, vocabulary, denoise)


def _run_transcribe(engine, audio_path, source_name, diarize, translate,
                    vocabulary=None, denoise=False):
    result = engine.transcribe(audio_path, diarize=diarize, translate=translate,
                               vocabulary=vocabulary, denoise=denoise,
                               on_segment=lambda seg: None)
    result["source_file"] = source_name

    for seg in result.get("segments", []):
        yield seg, result


def transcribe_single(file, diarize, translate, fmt, hf_token, vocabulary, denoise):
    if file is None:
        gr.Warning("Please upload a file first.")
        return "<p>No file uploaded.</p>", None, None

    token = _resolve_token(hf_token, diarize)
    if diarize and token is None:
        gr.Warning("HuggingFace token is required for speaker diarization. "
                   "Enter your token in Settings, or uncheck Speaker Diarization.")
        return "<p>Missing HuggingFace token.</p>", None, None

    input_path = pathlib.Path(file)
    vocab = vocabulary.strip() if vocabulary else None
    engine = TranscriberEngine(hf_token=token)

    html_parts = []
    result = None
    seg_idx = 0

    for seg, res in _transcribe_file(engine, input_path, diarize, translate, vocab, denoise):
        result = res
        html_parts.append(_format_segment_html(seg, seg_idx))
        seg_idx += 1
        yield _wrap_transcript_html("\n".join(html_parts)), file, None

    if result is None:
        yield "<p>No segments found.</p>", file, None
        return

    output_path = input_path.with_suffix(f".{fmt}")
    _write_output(result, output_path, fmt)

    yield _wrap_transcript_html("\n".join(html_parts)), file, str(output_path)


def transcribe_batch(folder_files, diarize, translate, fmt, hf_token, vocabulary, denoise):
    if not folder_files:
        gr.Warning("Please upload files first.")
        return "No files uploaded.", None

    token = _resolve_token(hf_token, diarize)
    if diarize and token is None:
        gr.Warning("HuggingFace token is required for speaker diarization. "
                   "Enter your token in Settings, or uncheck Speaker Diarization.")
        return "Missing HuggingFace token.", None

    vocab = vocabulary.strip() if vocabulary else None
    engine = TranscriberEngine(hf_token=token)
    preview_lines = []
    output_paths = []
    failed = []

    files = [pathlib.Path(f) for f in folder_files]
    total = len(files)

    for i, input_path in enumerate(files, 1):
        preview_lines.append(f"\n--- [{i}/{total}] {input_path.name} ---")
        yield "\n".join(preview_lines), None

        try:
            result = None
            for seg, res in _transcribe_file(engine, input_path, diarize, translate, vocab, denoise):
                result = res
                preview_lines.append(_format_segment_line(seg))
                yield "\n".join(preview_lines), None

            if result:
                output_path = input_path.with_suffix(f".{fmt}")
                _write_output(result, output_path, fmt)
                output_paths.append(str(output_path))
        except Exception as e:
            logger.error(f"Failed: {input_path}", exc_info=True)
            failed.append(input_path.name)
            preview_lines.append(f"  FAILED: {e}")
            yield "\n".join(preview_lines), None

    succeeded = total - len(failed)
    preview_lines.append(f"\nDone: {succeeded}/{total} succeeded")
    output_summary = "\n".join(output_paths) if output_paths else None
    yield "\n".join(preview_lines), output_summary


def _wrap_transcript_html(inner: str) -> str:
    return (
        f'<div style="font-family:system-ui,-apple-system,sans-serif;font-size:14px;'
        f'max-height:400px;overflow-y:auto;padding:4px">{inner}</div>'
    )


SEEK_JS = """
function seekAudio(time) {
    const audioElements = document.querySelectorAll('audio');
    for (const audio of audioElements) {
        if (audio.src) {
            audio.currentTime = time;
            audio.play();
            break;
        }
    }
}
"""


def build_ui():
    with gr.Blocks(
        title="Audio Transcriber",
        theme=gr.themes.Soft(),
        head=f"<script>{SEEK_JS}</script>",
    ) as app:
        gr.Markdown("# Audio Transcriber")
        gr.Markdown("Transcribe audio/video with speaker diarization. Runs locally.")

        with gr.Accordion("Settings", open=False):
            hf_token = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_... (required for speaker diarization)",
                value=os.environ.get("HF_TOKEN", ""),
                type="password",
            )
            gr.Markdown(
                "Get a free token from [huggingface.co/settings/tokens]"
                "(https://huggingface.co/settings/tokens). "
                "Only needed if Speaker Diarization is enabled."
            )

        with gr.Row():
            diarize = gr.Checkbox(value=True, label="Speaker Diarization")
            translate = gr.Checkbox(value=False, label="Translate to English")
            denoise = gr.Checkbox(value=False, label="Noise Reduction")
            fmt = gr.Dropdown(choices=["json", "txt", "srt", "vtt"], value="json", label="Format")

        vocabulary = gr.Textbox(
            label="Custom Vocabulary",
            placeholder="e.g. SARC, BITS Pilani, Aryaman, PyTorch, MLX",
            info="Comma-separated names, acronyms, or jargon to help recognition accuracy",
        )

        with gr.Tabs():
            with gr.TabItem("Single File"):
                file_input = gr.File(label="Upload Audio/Video", file_types=[
                    ".mp3", ".mp4", ".wav", ".m4a", ".mov", ".flac", ".ogg", ".webm"
                ])
                single_btn = gr.Button("Transcribe", variant="primary")

                audio_player = gr.Audio(
                    label="Playback",
                    type="filepath",
                    interactive=False,
                )
                single_preview = gr.HTML(
                    label="Transcript",
                    value="<p style='color:#94a3b8'>Transcript will appear here. Click any line to jump to that point in the audio.</p>",
                )
                single_output = gr.Textbox(label="Output Path", interactive=False)

                single_btn.click(
                    fn=transcribe_single,
                    inputs=[file_input, diarize, translate, fmt, hf_token, vocabulary, denoise],
                    outputs=[single_preview, audio_player, single_output],
                )

            with gr.TabItem("Batch"):
                batch_input = gr.File(
                    label="Upload Multiple Audio Files",
                    file_count="multiple",
                    file_types=[".mp3", ".mp4", ".wav", ".m4a", ".mov", ".flac", ".ogg", ".webm"],
                )
                batch_btn = gr.Button("Transcribe All", variant="primary")
                batch_preview = gr.Textbox(label="Live Preview", lines=15, interactive=False)
                batch_output = gr.Textbox(label="Output Paths", interactive=False)

                batch_btn.click(
                    fn=transcribe_batch,
                    inputs=[batch_input, diarize, translate, fmt, hf_token, vocabulary, denoise],
                    outputs=[batch_preview, batch_output],
                )

    return app


if __name__ == "__main__":
    build_ui().launch()
