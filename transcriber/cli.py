#!/usr/bin/env python3
import pathlib
import tempfile
import click
from transcriber.audio import extract_wav
from transcriber.engine import WhisperEngine

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_file", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("-m", "--model", default="medium", show_default=True,
              help="Whisper model size (tiny|base|small|medium|large)")
@click.option("-o", "--output", type=click.Path(path_type=pathlib.Path),
              help="Output TXT file (default: same name).txt")
@click.option("-q", "--quiet", is_flag=True,
              help="Reduce terminal output")
def cli(
    input_file: pathlib.Path,
    model: str,
    output: pathlib.Path | None,
    quiet: bool,
):
    """
    Transcribe audio/video files with automatic translation to English.
    
    All content is automatically translated to English with timestamps.
    """
    if output is None:
        output = input_file.with_suffix(".txt")

    if not quiet:
        click.echo(f"Processing: {input_file}")
        click.echo(f"Using model: {model}")
        click.echo(f"Output will be saved to: {output}")

    # Initialize engine
    engine = WhisperEngine(model_size=model)

    if not quiet:
        click.echo("⏩  Extracting audio…")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = pathlib.Path(tmpdir) / "audio.wav"
        extract_wav(input_file, wav_path)
        
        if not quiet:
            click.echo("📝  Transcribing and translating to English... (this may take a while)")
        
        result = engine.transcribe_wav(wav_path)
        
        # Write to file
        output.write_text(result, encoding="utf-8")
        
        if not quiet:
            click.echo(f"✅  Saved transcript to {output}")

if __name__ == "__main__":
    cli()