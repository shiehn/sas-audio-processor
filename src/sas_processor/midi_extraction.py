"""MIDI extraction from audio using basic-pitch."""

from sas_processor.io_utils import atomic_midi_write


def extract_melody_to_midi(input_path: str, output_path: str) -> dict:
    """Extract melody from audio and save as MIDI file.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output MIDI file

    Returns:
        Dict with extraction details
    """
    from basic_pitch.inference import predict_and_save, predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(input_path)

    # Atomic write — temp + replace prevents half-written .mid files on crash.
    atomic_midi_write(output_path, midi_data)

    return {
        "note_count": len(note_events),
        "duration_seconds": round(float(midi_data.get_end_time()), 4),
    }
