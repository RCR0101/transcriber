#!/usr/bin/env python3
import json
import pathlib
import tempfile

import click
from dotenv import load_dotenv

from transcriber.audio import extract_wav
from transcriber.engine import TranscriberEngine, format_timestamp, format_srt, format_vtt, _default_model

load_dotenv()

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".flac", ".ogg", ".webm"}


def _collect_files(paths: tuple[pathlib.Path, ...]) -> list[pathlib.Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
            )
        else:
            files.append(p)
    return files


def _transcribe_one(
    engine: TranscriberEngine,
    input_file: pathlib.Path,
    *,
    diarize: bool,
    translate: bool,
    vocabulary: str | None = None,
    denoise: bool = False,
    quiet: bool,
) -> dict:
    suffix = input_file.suffix.lower()
    if suffix != ".wav":
        if not quiet:
            click.echo("  Converting to WAV...")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = pathlib.Path(tmpdir) / "audio.wav"
            extract_wav(input_file, wav_path)
            if denoise and not quiet:
                click.echo("  Denoising audio...")
            result = engine.transcribe(wav_path, diarize=diarize, translate=translate,
                                       vocabulary=vocabulary, denoise=denoise)
    else:
        if denoise and not quiet:
            click.echo("  Denoising audio...")
        result = engine.transcribe(input_file, diarize=diarize, translate=translate,
                                   vocabulary=vocabulary, denoise=denoise)

    result["source_file"] = input_file.name
    return result


def _write_output(result: dict, output: pathlib.Path, fmt: str) -> None:
    if fmt == "json":
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt == "srt":
        output.write_text(format_srt(result), encoding="utf-8")
    elif fmt == "vtt":
        output.write_text(format_vtt(result), encoding="utf-8")
    else:
        lines = []
        for seg in result["segments"]:
            ts = format_timestamp(seg["start"])
            speaker = seg.get("speaker")
            prefix = f"[{speaker}] " if speaker else ""
            lines.append(f"[{ts}] {prefix}{seg['text']}")
        output.write_text("\n".join(lines), encoding="utf-8")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_files", nargs=-1, required=True,
                type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-o", "--output", type=click.Path(path_type=pathlib.Path),
              help="Output file path (only for single file; ignored in batch mode)")
@click.option("-m", "--model", default=None,
              help="Whisper model (default: auto-detected per platform)")
@click.option("--hf-token", envvar="HF_TOKEN",
              help="HuggingFace token for pyannote diarization (or set HF_TOKEN env var)")
@click.option("--no-diarize", is_flag=True,
              help="Skip speaker diarization")
@click.option("--translate", is_flag=True,
              help="Translate all speech to English (default: verbatim transcription)")
@click.option("--format", "fmt", type=click.Choice(["json", "txt", "srt", "vtt"]),
              default="json", show_default=True,
              help="Output format")
@click.option("-v", "--vocabulary", default=None,
              help="Comma-separated terms to bias recognition (names, jargon, acronyms)")
@click.option("--denoise", is_flag=True,
              help="Apply two-stage noise reduction before transcribing")
@click.option("-q", "--quiet", is_flag=True,
              help="Reduce terminal output")
def cli(
    input_files: tuple[pathlib.Path, ...],
    output: pathlib.Path | None,
    model: str,
    hf_token: str | None,
    no_diarize: bool,
    translate: bool,
    fmt: str,
    vocabulary: str | None,
    denoise: bool,
    quiet: bool,
):
    """Transcribe audio/video files with speaker diarization.

    Accepts one or more files, or a directory of audio files.
    Supports mp3, mp4, wav, m4a, mov, flac, ogg, webm.
    Auto-detects language. Use --translate to convert all speech to English.
    """
    files = _collect_files(input_files)

    if not files:
        raise click.ClickException("No audio files found in the given paths.")

    batch = len(files) > 1
    if batch and output is not None:
        raise click.ClickException("-o/--output cannot be used with multiple input files.")

    if not quiet:
        click.echo(f"Model: {model or _default_model()}")
        if no_diarize:
            click.echo("Diarization: disabled")
        if translate:
            click.echo("Mode: translate to English")
        if denoise:
            click.echo("Noise reduction: enabled")
        if batch:
            click.echo(f"Batch: {len(files)} files")

    engine = TranscriberEngine(model_repo=model, hf_token=hf_token)

    failed = []
    for i, input_file in enumerate(files, 1):
        if not quiet:
            label = f"[{i}/{len(files)}] " if batch else ""
            click.echo(f"{label}Processing: {input_file}")

        try:
            result = _transcribe_one(
                engine, input_file,
                diarize=not no_diarize, translate=translate,
                vocabulary=vocabulary, denoise=denoise, quiet=quiet,
            )

            out_path = output if output else input_file.with_suffix(f".{fmt}")
            _write_output(result, out_path, fmt)

            if not quiet:
                click.echo(f"  Saved to {out_path}")
        except (RuntimeError, ValueError) as e:
            if batch:
                failed.append((input_file, e))
                if not quiet:
                    click.echo(f"  FAILED: {e}", err=True)
            else:
                raise click.ClickException(str(e))
        except Exception as e:
            if batch:
                failed.append((input_file, e))
                if not quiet:
                    click.echo(f"  FAILED: {e}", err=True)
            else:
                raise

    if batch and not quiet:
        succeeded = len(files) - len(failed)
        click.echo(f"\nDone: {succeeded}/{len(files)} succeeded")
        if failed:
            click.echo("Failed files:")
            for f, err in failed:
                click.echo(f"  {f}: {err}")


if __name__ == "__main__":
    cli()
