"""Command-line interface for SAS Audio Processor."""

import json
import sys
from pathlib import Path

import click

from sas_processor import __version__ as PROCESSOR_VERSION


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


def _validate_input_file(input_path: str) -> None:
    """Validate input file exists. Exits with error if not."""
    if not Path(input_path).exists():
        emit_error("FILE_NOT_FOUND", f"Input file not found: {input_path}")
        sys.exit(1)


@click.group()
@click.version_option(version=PROCESSOR_VERSION)
def main():
    """SAS Audio Processor - Audio processing tools for music production."""
    pass


@main.command()
def ping():
    """Health check - returns version and status."""
    emit_json({
        "type": "ping",
        "success": True,
        "version": PROCESSOR_VERSION,
        "status": "ok"
    })


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--json', 'json_output', is_flag=True, default=True, hidden=True,
              help='Output as JSON (always on)')
def analyze(input_path: str, json_output: bool) -> None:
    """Analyze audio file: detect BPM and extract features."""
    _validate_input_file(input_path)

    try:
        from sas_processor.feature_extraction import extract_features
        features = extract_features(input_path)

        emit_json({
            "type": "analyze",
            "success": True,
            "bpm": features.bpm,
            "duration_seconds": features.duration_seconds,
            "sample_rate": features.sample_rate,
            "channels": features.channels,
        })
    except Exception as e:
        emit_error("ANALYSIS_ERROR", f"Analysis failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--bpm', required=True, type=float,
              help='BPM (beats per minute) of the audio')
@click.option('--bars', required=True, type=int,
              help='Number of bars to extract')
@click.option('--meter', default=4, type=int,
              help='Beats per bar (default: 4 for 4/4 time)')
@click.option('--json', 'json_output', is_flag=True, default=True, hidden=True,
              help='Output as JSON (always on)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def trim(input_path: str, output_path: str, bpm: float, bars: int,
         meter: int, json_output: bool, verbose: bool) -> None:
    """Detect downbeat and trim audio to specified bars."""
    _validate_input_file(input_path)

    if bpm <= 0 or bpm > 999:
        emit_error("INVALID_BPM", "BPM must be between 1 and 999")
        sys.exit(1)

    if bars <= 0 or bars > 999:
        emit_error("INVALID_BARS", "Bars must be between 1 and 999")
        sys.exit(1)

    if meter < 1 or meter > 16:
        emit_error("INVALID_METER", "Meter must be between 1 and 16")
        sys.exit(1)

    try:
        from sas_processor.processor import process_audio

        def progress_callback(stage: str, percent: int) -> None:
            emit_json({"type": "progress", "stage": stage, "percent": percent})

        result = process_audio(
            input_path=input_path,
            output_path=output_path,
            bpm=bpm,
            bars=bars,
            meter=meter,
            progress_callback=progress_callback
        )

        if result.success:
            emit_json({
                "type": "trim",
                "success": True,
                "output": result.output_path,
                "downbeat_time": round(result.downbeat_time, 4),
                "original_duration": round(result.original_duration, 4),
                "output_duration": round(result.output_duration, 4),
                "sample_rate": result.sample_rate
            })
        else:
            emit_error(result.error_code or "PROCESSING_ERROR",
                       result.error or "Unknown error")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        emit_error("UNEXPECTED_ERROR", f"Unexpected error: {str(e)}")
        sys.exit(1)


@main.command('time-stretch')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--source-bpm', required=True, type=float,
              help='Source BPM of the audio')
@click.option('--target-bpm', required=True, type=float,
              help='Target BPM after stretching')
@click.option('--json', 'json_output', is_flag=True, default=True, hidden=True,
              help='Output as JSON (always on)')
def time_stretch(input_path: str, output_path: str, source_bpm: float,
                 target_bpm: float, json_output: bool) -> None:
    """Time-stretch audio to change tempo while preserving pitch."""
    _validate_input_file(input_path)

    try:
        from sas_processor.time_stretch import time_stretch_audio
        result = time_stretch_audio(input_path, output_path, source_bpm, target_bpm)

        if result.success:
            emit_json({
                "type": "time-stretch",
                "success": True,
                "output": result.output_path,
                "source_bpm": result.source_bpm,
                "target_bpm": result.target_bpm,
                "rate": round(result.rate, 6),
                "duration_seconds": round(result.duration_seconds, 4),
            })
        else:
            emit_error(result.error_code or "STRETCH_ERROR",
                       result.error or "Unknown error")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        emit_error("STRETCH_ERROR", f"Time stretch failed: {str(e)}")
        sys.exit(1)


# --- Phase 2: Core Effects ---

@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--mode', type=click.Choice(['peak', 'lufs']), default='lufs',
              help='Normalization mode (default: lufs)')
@click.option('--target-lufs', type=float, default=-14.0,
              help='Target loudness in LUFS (default: -14)')
@click.option('--target-peak', type=float, default=-1.0,
              help='Target peak in dB (default: -1.0)')
def normalize(input_path: str, output_path: str, mode: str,
              target_lufs: float, target_peak: float) -> None:
    """Normalize audio to target loudness level."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import normalize_audio
        result = normalize_audio(input_path, output_path, mode, target_lufs, target_peak)
        emit_json({"type": "normalize", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("NORMALIZE_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--db', required=True, type=float,
              help='Gain in decibels (positive = louder, negative = quieter)')
def gain(input_path: str, output_path: str, db: float) -> None:
    """Apply gain (volume change) in decibels."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import apply_gain
        result = apply_gain(input_path, output_path, db)
        emit_json({"type": "gain", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("GAIN_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
def mono(input_path: str, output_path: str) -> None:
    """Convert stereo audio to mono."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import to_mono
        result = to_mono(input_path, output_path)
        emit_json({"type": "mono", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("MONO_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--sample-rate', type=int, default=None,
              help='Target sample rate (e.g. 44100, 48000)')
@click.option('--bit-depth', type=click.Choice(['16', '24', '32']), default=None,
              help='Target bit depth')
def convert(input_path: str, output_path: str, sample_rate: int,
            bit_depth: str) -> None:
    """Convert audio sample rate and/or bit depth."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import convert_audio
        result = convert_audio(input_path, output_path, sample_rate, bit_depth)
        emit_json({"type": "convert", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("CONVERT_ERROR", str(e))
        sys.exit(1)


@main.command('silence-remove')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--top-db', type=float, default=30.0,
              help='Threshold in dB below peak to consider as silence (default: 30)')
def silence_remove(input_path: str, output_path: str, top_db: float) -> None:
    """Trim silence from start and end of audio."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import remove_silence
        result = remove_silence(input_path, output_path, top_db)
        emit_json({"type": "silence-remove", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("SILENCE_REMOVE_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--threshold', type=float, default=-20.0,
              help='Threshold in dB (default: -20)')
@click.option('--ratio', type=float, default=4.0,
              help='Compression ratio (default: 4.0)')
@click.option('--attack', type=float, default=1.0,
              help='Attack time in ms (default: 1.0)')
@click.option('--release', type=float, default=100.0,
              help='Release time in ms (default: 100.0)')
def compress(input_path: str, output_path: str, threshold: float,
             ratio: float, attack: float, release: float) -> None:
    """Apply dynamics compression."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import compress_audio
        result = compress_audio(input_path, output_path, threshold, ratio, attack, release)
        emit_json({"type": "compress", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("COMPRESS_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--freq', required=True, type=float,
              help='Center frequency in Hz')
@click.option('--gain-db', required=True, type=float,
              help='Gain in dB (positive = boost, negative = cut)')
@click.option('--q', type=float, default=1.0,
              help='Q factor / bandwidth (default: 1.0)')
def eq(input_path: str, output_path: str, freq: float,
       gain_db: float, q: float) -> None:
    """Apply parametric EQ band."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import apply_eq
        result = apply_eq(input_path, output_path, freq, gain_db, q)
        emit_json({"type": "eq", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("EQ_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--room-size', type=float, default=0.5,
              help='Room size 0.0-1.0 (default: 0.5)')
@click.option('--damping', type=float, default=0.5,
              help='Damping 0.0-1.0 (default: 0.5)')
@click.option('--wet-level', type=float, default=0.33,
              help='Wet/dry mix 0.0-1.0 (default: 0.33)')
def reverb(input_path: str, output_path: str, room_size: float,
           damping: float, wet_level: float) -> None:
    """Apply algorithmic reverb."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import apply_reverb
        result = apply_reverb(input_path, output_path, room_size, damping, wet_level)
        emit_json({"type": "reverb", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("REVERB_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--threshold', type=float, default=-1.0,
              help='Limiter threshold in dB (default: -1.0)')
def limit(input_path: str, output_path: str, threshold: float) -> None:
    """Apply brick-wall limiter."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import apply_limiter
        result = apply_limiter(input_path, output_path, threshold)
        emit_json({"type": "limit", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("LIMIT_ERROR", str(e))
        sys.exit(1)


@main.command('filter')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--type', 'filter_type', required=True,
              type=click.Choice(['highpass', 'lowpass']),
              help='Filter type')
@click.option('--cutoff-hz', required=True, type=float,
              help='Cutoff frequency in Hz')
def filter_cmd(input_path: str, output_path: str, filter_type: str,
               cutoff_hz: float) -> None:
    """Apply high-pass or low-pass filter."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import apply_filter
        result = apply_filter(input_path, output_path, filter_type, cutoff_hz)
        emit_json({"type": "filter", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("FILTER_ERROR", str(e))
        sys.exit(1)


@main.command('pitch-shift')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output WAV file path')
@click.option('--semitones', required=True, type=float,
              help='Number of semitones to shift (positive = up, negative = down)')
def pitch_shift(input_path: str, output_path: str, semitones: float) -> None:
    """Pitch-shift audio by semitones."""
    _validate_input_file(input_path)
    try:
        from sas_processor.effects import pitch_shift_audio
        result = pitch_shift_audio(input_path, output_path, semitones)
        emit_json({"type": "pitch-shift", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("PITCH_SHIFT_ERROR", str(e))
        sys.exit(1)


# --- Phase 3: Analysis & Segmentation ---

@main.command('detect-key')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
def detect_key(input_path: str) -> None:
    """Detect the musical key of audio."""
    _validate_input_file(input_path)
    try:
        from sas_processor.analysis import detect_musical_key
        result = detect_musical_key(input_path)
        emit_json({"type": "detect-key", "success": True, **result})
    except Exception as e:
        emit_error("KEY_DETECTION_ERROR", str(e))
        sys.exit(1)


@main.command()
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
def loudness(input_path: str) -> None:
    """Measure integrated LUFS loudness."""
    _validate_input_file(input_path)
    try:
        from sas_processor.analysis import measure_loudness
        result = measure_loudness(input_path)
        emit_json({"type": "loudness", "success": True, **result})
    except Exception as e:
        emit_error("LOUDNESS_ERROR", str(e))
        sys.exit(1)


@main.command('onset-detect')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
def onset_detect(input_path: str) -> None:
    """Detect onset times in audio."""
    _validate_input_file(input_path)
    try:
        from sas_processor.analysis import detect_onsets
        result = detect_onsets(input_path)
        emit_json({"type": "onset-detect", "success": True, **result})
    except Exception as e:
        emit_error("ONSET_DETECT_ERROR", str(e))
        sys.exit(1)


@main.command('split-bars')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output-dir', required=True,
              type=click.Path(), help='Output directory for bar chunks')
@click.option('--bpm', required=True, type=float, help='BPM of the audio')
@click.option('--bars-per-chunk', type=int, default=1,
              help='Number of bars per output chunk (default: 1)')
@click.option('--meter', default=4, type=int,
              help='Beats per bar (default: 4)')
def split_bars(input_path: str, output_dir: str, bpm: float,
               bars_per_chunk: int, meter: int) -> None:
    """Split audio into N-bar chunks."""
    _validate_input_file(input_path)
    try:
        from sas_processor.analysis import split_audio_bars
        result = split_audio_bars(input_path, output_dir, bpm, bars_per_chunk, meter)
        emit_json({"type": "split-bars", "success": True, **result})
    except Exception as e:
        emit_error("SPLIT_BARS_ERROR", str(e))
        sys.exit(1)


# --- Phase 4: MIDI Extraction ---

@main.command('melody-to-midi')
@click.option('--input', '-i', 'input_path', required=True,
              type=click.Path(exists=False), help='Input WAV file path')
@click.option('--output', '-o', 'output_path', required=True,
              type=click.Path(), help='Output MIDI file path')
def melody_to_midi(input_path: str, output_path: str) -> None:
    """Extract melody from audio and save as MIDI."""
    _validate_input_file(input_path)
    try:
        from sas_processor.midi_extraction import extract_melody_to_midi
        result = extract_melody_to_midi(input_path, output_path)
        emit_json({"type": "melody-to-midi", "success": True, "output": output_path, **result})
    except Exception as e:
        emit_error("MIDI_EXTRACTION_ERROR", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
