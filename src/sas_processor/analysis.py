"""Audio analysis: key detection, loudness measurement, onset detection, bar splitting."""

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _load_mono(input_path: str):
    """Load audio as mono float64. Returns (audio_mono, sr, original_subtype)."""
    info = sf.info(input_path)
    audio, sr = sf.read(input_path, dtype='float64')
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    return mono, sr, info.subtype, audio


# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_musical_key(input_path: str) -> dict:
    """Detect musical key using Krumhansl-Schmuckler algorithm."""
    mono, sr, _, _ = _load_mono(input_path)

    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)  # 12-element vector

    best_corr = -np.inf
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        major_shifted = np.roll(MAJOR_PROFILE, i)
        minor_shifted = np.roll(MINOR_PROFILE, i)

        corr_major = np.corrcoef(chroma_avg, major_shifted)[0, 1]
        corr_minor = np.corrcoef(chroma_avg, minor_shifted)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = KEY_NAMES[i]
            best_mode = "major"

        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = KEY_NAMES[i]
            best_mode = "minor"

    return {
        "key": best_key,
        "mode": best_mode,
        "key_label": f"{best_key} {best_mode}",
        "confidence": round(float(best_corr), 4),
    }


def measure_loudness(input_path: str) -> dict:
    """Measure integrated LUFS loudness."""
    import pyloudnorm as pyln

    audio, sr = sf.read(input_path, dtype='float64')
    meter = pyln.Meter(sr)

    if audio.ndim == 1:
        lufs = meter.integrated_loudness(audio.reshape(-1, 1))
    else:
        lufs = meter.integrated_loudness(audio)

    peak = float(np.max(np.abs(audio)))
    peak_db = 20.0 * np.log10(peak) if peak > 0 else -np.inf

    return {
        "lufs": round(float(lufs), 2),
        "peak_db": round(float(peak_db), 2),
        "peak_linear": round(peak, 6),
    }


def detect_onsets(input_path: str) -> dict:
    """Detect onset times in audio."""
    mono, sr, _, _ = _load_mono(input_path)

    onset_frames = librosa.onset.onset_detect(y=mono, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return {
        "onset_count": len(onset_times),
        "onset_times": [round(float(t), 4) for t in onset_times],
    }


def split_audio_bars(input_path: str, output_dir: str, bpm: float,
                     bars_per_chunk: int, meter: int) -> dict:
    """Split audio into bar-aligned chunks."""
    audio, sr = sf.read(input_path, dtype='float64')
    info = sf.info(input_path)
    subtype = info.subtype

    os.makedirs(output_dir, exist_ok=True)

    samples_per_beat = (60.0 / bpm) * sr
    samples_per_bar = samples_per_beat * meter
    samples_per_chunk = int(samples_per_bar * bars_per_chunk)

    total_samples = audio.shape[0] if audio.ndim > 1 else len(audio)
    n_chunks = total_samples // samples_per_chunk

    output_files = []
    for i in range(n_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        if audio.ndim > 1:
            chunk = audio[start:end]
        else:
            chunk = audio[start:end]

        filename = f"bar_{i+1:04d}.wav"
        out_path = str(Path(output_dir) / filename)
        sf.write(out_path, chunk, sr, subtype=subtype)
        output_files.append(out_path)

    return {
        "chunks": len(output_files),
        "bars_per_chunk": bars_per_chunk,
        "output_dir": output_dir,
        "files": output_files,
    }
