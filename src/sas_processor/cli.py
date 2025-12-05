"""Command-line interface for SAS Audio Processor."""

import json
import sys
from pathlib import Path

import click

from sas_processor.processor import process_audio


def emit_json(data: dict) -> None:
    """Print JSON to stdout (line-delimited for streaming)."""
    print(json.dumps(data), flush=True)


def emit_error(code: str, message: str, severity: str = "fatal") -> None:
    """Print error to stderr in JSON format."""
    print(json.dumps({
        "type": "error",
        "code": code,
        "message": message,
        "severity": severity
    }), file=sys.stderr, flush=True)


@click.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False),
              help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(),
              help='Output WAV file path')
@click.option('--bpm', required=True, type=float,
              help='BPM (beats per minute) of the audio')
@click.option('--bars', required=True, type=int,
              help='Number of bars to extract')
@click.option('--meter', default=4, type=int,
              help='Beats per bar (default: 4 for 4/4 time)')
@click.option('--key', default=None, type=str,
              help='Musical key (optional, for future use)')
@click.option('--json', 'json_output', is_flag=True,
              help='Output progress and results as JSON')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def main(input_path: str, output_path: str, bpm: float, bars: int,
         meter: int, key: str, json_output: bool, verbose: bool) -> None:
    """
    SAS Audio Processor - Detect downbeat and trim audio to specified bars.

    Takes a WAV file, detects the downbeat using the provided BPM,
    and outputs a trimmed WAV file starting at the downbeat for
    the specified number of bars.
    """
    try:
        _run_processing(input_path, output_path, bpm, bars, meter, json_output, verbose)
    except SystemExit:
        # Allow sys.exit() calls to pass through
        raise
    except Exception as e:
        # Catch-all for any unexpected errors - ALWAYS return proper output
        if json_output:
            emit_error("UNEXPECTED_ERROR", f"Unexpected error: {str(e)}")
        else:
            click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


def _run_processing(input_path: str, output_path: str, bpm: float, bars: int,
                    meter: int, json_output: bool, verbose: bool) -> None:
    """Internal processing logic, wrapped by main() for error handling."""
    # Validate inputs
    if bpm <= 0 or bpm > 999:
        if json_output:
            emit_error("INVALID_BPM", "BPM must be between 1 and 999")
        else:
            click.echo("Error: BPM must be between 1 and 999", err=True)
        sys.exit(1)

    if bars <= 0 or bars > 999:
        if json_output:
            emit_error("INVALID_BARS", "Bars must be between 1 and 999")
        else:
            click.echo("Error: Bars must be between 1 and 999", err=True)
        sys.exit(1)

    if meter < 1 or meter > 16:
        if json_output:
            emit_error("INVALID_METER", "Meter must be between 1 and 16")
        else:
            click.echo("Error: Meter must be between 1 and 16", err=True)
        sys.exit(1)

    # Check input file exists
    if not Path(input_path).exists():
        if json_output:
            emit_error("FILE_NOT_FOUND", f"Input file not found: {input_path}")
        else:
            click.echo(f"Error: Input file not found: {input_path}", err=True)
        sys.exit(1)

    # Progress callback for JSON output
    def progress_callback(stage: str, percent: int) -> None:
        if json_output:
            emit_json({
                "type": "progress",
                "stage": stage,
                "percent": percent
            })
        elif verbose:
            click.echo(f"[{stage}] {percent}%")

    # Process the audio
    result = process_audio(
        input_path=input_path,
        output_path=output_path,
        bpm=bpm,
        bars=bars,
        meter=meter,
        progress_callback=progress_callback
    )

    if result.success:
        if json_output:
            emit_json({
                "type": "complete",
                "success": True,
                "output": result.output_path,
                "downbeat_time": round(result.downbeat_time, 4),
                "original_duration": round(result.original_duration, 4),
                "output_duration": round(result.output_duration, 4),
                "sample_rate": result.sample_rate
            })
        else:
            click.echo(f"Success! Output saved to: {result.output_path}")
            if verbose:
                click.echo(f"  Downbeat at: {result.downbeat_time:.3f}s")
                click.echo(f"  Original duration: {result.original_duration:.3f}s")
                click.echo(f"  Output duration: {result.output_duration:.3f}s")
                click.echo(f"  Sample rate: {result.sample_rate}Hz")
        sys.exit(0)
    else:
        error_code = result.error_code or "PROCESSING_ERROR"
        if json_output:
            emit_error(error_code, result.error or "Unknown error")
        else:
            click.echo(f"Error [{error_code}]: {result.error}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
