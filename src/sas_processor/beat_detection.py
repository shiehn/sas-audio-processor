"""Beat and downbeat detection using librosa."""

import numpy as np
import librosa


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono if stereo."""
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def detect_beats(audio: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """
    Detect beat positions in audio using known BPM.

    Args:
        audio: Audio samples (mono or will be converted to mono)
        sr: Sample rate
        bpm: Known BPM of the audio

    Returns:
        Array of beat positions in samples (at original sample rate)
    """
    audio_mono = _to_mono(audio)

    # Librosa works best at 22050Hz, but we want beat positions at original SR
    # Use beat_track with known BPM to skip tempo estimation
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio_mono,
        sr=sr,
        bpm=bpm,
        units='frames'
    )

    # Convert frames to samples at original sample rate
    # Default hop_length in librosa is 512
    hop_length = 512
    beat_samples = librosa.frames_to_samples(beat_frames, hop_length=hop_length)

    return beat_samples


def find_downbeat(audio_mono: np.ndarray, sr: int, beat_samples: np.ndarray, meter: int = 4) -> int:
    """
    Find the downbeat (beat 1) position using energy-based heuristic.

    This works by assuming 4/4 time and finding which beat position
    (0, 1, 2, or 3) has the highest average onset strength across all bars.

    Args:
        audio_mono: Audio samples (mono)
        sr: Sample rate
        beat_samples: Array of beat positions in samples
        meter: Beats per bar (default 4 for 4/4 time)

    Returns:
        Index into beat_samples array that represents the first downbeat
    """
    if len(beat_samples) < meter:
        # Not enough beats, return first beat
        return 0

    # Get onset strength envelope
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr, hop_length=hop_length)

    # Get onset strength at each beat position
    beat_frames = librosa.samples_to_frames(beat_samples, hop_length=hop_length)

    # Clip to valid range
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)
    beat_strengths = onset_env[beat_frames]

    # Group beats into measures and find which position (0-3) is strongest
    n_complete_bars = len(beat_samples) // meter
    if n_complete_bars < 2:
        return 0

    # Try each possible downbeat offset (0, 1, 2, 3)
    best_offset = 0
    best_strength = -np.inf

    for offset in range(meter):
        # Get all beats at this position in the bar
        positions = list(range(offset, len(beat_samples), meter))
        if len(positions) < 2:
            continue

        # Average strength at this beat position
        strengths = beat_strengths[positions]
        avg_strength = np.mean(strengths)

        if avg_strength > best_strength:
            best_strength = avg_strength
            best_offset = offset

    return best_offset


def get_downbeat_sample(audio: np.ndarray, sr: int, bpm: float, meter: int = 4) -> tuple[int, np.ndarray]:
    """
    Find the first downbeat position in samples.

    Args:
        audio: Audio samples
        sr: Sample rate
        bpm: Known BPM
        meter: Beats per bar (default 4)

    Returns:
        Tuple of (downbeat_sample_position, all_beat_samples)
    """
    audio_mono = _to_mono(audio)
    beat_samples = detect_beats(audio_mono, sr, bpm)

    if len(beat_samples) == 0:
        return 0, np.array([])

    downbeat_idx = find_downbeat(audio_mono, sr, beat_samples, meter)
    downbeat_sample = beat_samples[downbeat_idx]

    return int(downbeat_sample), beat_samples
