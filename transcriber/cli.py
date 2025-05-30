#!/usr/bin/env python3
import tempfile, pathlib
import click
from transcriber.audio import extract_wav
from transcriber.engine import WhisperEngine

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_file", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-m", "--model", default="small", show_default=True,
              help="Whisper model size (tiny|base|small|medium|large)")
@click.option("-o", "--output", type=click.Path(path_type=pathlib.Path),
              help="Output TXT file (default: same name).txt")
def cli(input_file: pathlib.Path, model: str, output: pathlib.Path | None):
    """
    Transcribe a podcast MP4 → plain-text using Whisper offline.
    """
    if output is None:
        output = input_file.with_suffix(".txt")

    engine = WhisperEngine(model_size=model)

    print("⏩  Extracting audio…")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav = pathlib.Path(tmpdir) / "audio.wav"
        extract_wav(input_file, wav)

        print("📝  Transcribing…")
        text = engine.transcribe_wav(wav)

    output.write_text(text, encoding="utf-8")
    print("✅  Saved transcript to", output)

if __name__ == "__main__":
    cli()