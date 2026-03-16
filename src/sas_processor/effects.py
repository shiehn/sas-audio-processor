"""Audio effects processing using pedalboard, pyloudnorm, and numpy."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import soundfile as sf
import librosa


AudioData = npt.NDArray[np.float64]


def _load(input_path: str) -> Tuple[AudioData, int, str]:
    """Load audio preserving format info. Returns (audio, sr, subtype)."""
    info = sf.info(input_path)
    audio, sr = sf.read(input_path, dtype='float64')
    return audio, sr, info.subtype


def _save(audio: AudioData, output_path: str, sr: int, subtype: str) -> None:
    """Save audio preserving format."""
    sf.write(output_path, audio, sr, subtype=subtype)


def _apply_pedalboard_effect(effects: List[Any], input_path: str,
                             output_path: str) -> None:
    """Load audio, apply pedalboard effects, and save.

    Handles float32 conversion and channel layout (pedalboard expects
    channels-first float32; soundfile uses channels-last float64).
    """
    from pedalboard import Pedalboard
    audio, sr, subtype = _load(input_path)

    board = Pedalboard(effects)

    audio_f32 = audio.astype(np.float32)
    if audio_f32.ndim == 1:
        audio_f32 = audio_f32.reshape(1, -1)
    else:
        audio_f32 = audio_f32.T

    processed = board(audio_f32, sr)

    if audio.ndim == 1:
        processed = processed.flatten()
    else:
        processed = processed.T

    _save(processed.astype(np.float64), output_path, sr, subtype)


def normalize_audio(input_path: str, output_path: str, mode: str,
                    target_lufs: float, target_peak: float) -> Dict[str, Any]:
    audio, sr, subtype = _load(input_path)

    if mode == 'lufs':
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyloudnorm expects (samples, channels) for stereo
        if audio.ndim == 1:
            current_lufs = meter.integrated_loudness(audio.reshape(-1, 1))
            gain_db = target_lufs - current_lufs
            audio = audio * (10 ** (gain_db / 20.0))
        else:
            current_lufs = meter.integrated_loudness(audio)
            gain_db = target_lufs - current_lufs
            audio = audio * (10 ** (gain_db / 20.0))
    else:  # peak
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_linear = 10 ** (target_peak / 20.0)
            audio = audio * (target_linear / peak)
            gain_db = 20.0 * np.log10(target_linear / peak)
        else:
            gain_db = 0.0

    _save(audio, output_path, sr, subtype)
    return {"mode": mode, "gain_db": round(float(gain_db), 2)}


def apply_gain(input_path: str, output_path: str, db: float) -> Dict[str, Any]:
    audio, sr, subtype = _load(input_path)
    factor = 10 ** (db / 20.0)
    audio = audio * factor
    _save(audio, output_path, sr, subtype)
    return {"gain_db": db}


def to_mono(input_path: str, output_path: str) -> Dict[str, int]:
    audio, sr, subtype = _load(input_path)
    if audio.ndim > 1:
        channels_in = audio.shape[1]
        audio = np.mean(audio, axis=1)
    else:
        channels_in = 1
    _save(audio, output_path, sr, subtype)
    return {"channels_in": channels_in, "channels_out": 1}


def convert_audio(input_path: str, output_path: str,
                  target_sr: Optional[int], bit_depth: Optional[str]) -> Dict[str, Any]:
    audio, sr, subtype = _load(input_path)
    result = {"original_sample_rate": sr, "original_bit_depth": subtype}

    if target_sr and target_sr != sr:
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            channels = []
            for ch in range(audio.shape[1]):
                channels.append(librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr))
            audio = np.column_stack(channels)
        sr = target_sr

    if bit_depth:
        subtype = {'16': 'PCM_16', '24': 'PCM_24', '32': 'FLOAT'}[bit_depth]

    _save(audio, output_path, sr, subtype)
    result.update({"sample_rate": sr, "bit_depth": subtype})
    return result


def remove_silence(input_path: str, output_path: str, top_db: float) -> Dict[str, Any]:
    audio, sr, subtype = _load(input_path)
    original_len = audio.shape[0] if audio.ndim > 1 else len(audio)

    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    intervals = librosa.effects.split(mono, top_db=top_db)
    if len(intervals) == 0:
        _save(audio, output_path, sr, subtype)
        return {"trimmed_start": 0.0, "trimmed_end": 0.0}

    start = intervals[0][0]
    end = intervals[-1][1]

    if audio.ndim > 1:
        audio = audio[start:end]
    else:
        audio = audio[start:end]

    _save(audio, output_path, sr, subtype)
    return {
        "trimmed_start": round(start / sr, 4),
        "trimmed_end": round((original_len - end) / sr, 4),
        "duration_seconds": round((end - start) / sr, 4),
    }


def compress_audio(input_path: str, output_path: str, threshold_db: float,
                   ratio: float, attack_ms: float, release_ms: float) -> Dict[str, float]:
    from pedalboard import Compressor
    _apply_pedalboard_effect(
        [Compressor(threshold_db=threshold_db, ratio=ratio,
                    attack_ms=attack_ms, release_ms=release_ms)],
        input_path, output_path,
    )
    return {"threshold_db": threshold_db, "ratio": ratio,
            "attack_ms": attack_ms, "release_ms": release_ms}


def apply_eq(input_path: str, output_path: str, freq: float,
             gain_db: float, q: float) -> Dict[str, float]:
    from pedalboard import PeakFilter
    _apply_pedalboard_effect(
        [PeakFilter(cutoff_frequency_hz=freq, gain_db=gain_db, q=q)],
        input_path, output_path,
    )
    return {"freq": freq, "gain_db": gain_db, "q": q}


def apply_reverb(input_path: str, output_path: str, room_size: float,
                 damping: float, wet_level: float) -> Dict[str, float]:
    from pedalboard import Reverb
    _apply_pedalboard_effect(
        [Reverb(room_size=room_size, damping=damping, wet_level=wet_level,
                dry_level=1.0 - wet_level)],
        input_path, output_path,
    )
    return {"room_size": room_size, "damping": damping, "wet_level": wet_level}


def apply_limiter(input_path: str, output_path: str, threshold_db: float) -> Dict[str, float]:
    from pedalboard import Limiter
    _apply_pedalboard_effect(
        [Limiter(threshold_db=threshold_db)],
        input_path, output_path,
    )
    return {"threshold_db": threshold_db}


def apply_filter(input_path: str, output_path: str, filter_type: str,
                 cutoff_hz: float) -> Dict[str, Any]:
    from pedalboard import HighpassFilter, LowpassFilter
    filt = HighpassFilter(cutoff_frequency_hz=cutoff_hz) if filter_type == 'highpass' \
        else LowpassFilter(cutoff_frequency_hz=cutoff_hz)
    _apply_pedalboard_effect([filt], input_path, output_path)
    return {"filter_type": filter_type, "cutoff_hz": cutoff_hz}


def pitch_shift_audio(input_path: str, output_path: str, semitones: float) -> Dict[str, float]:
    audio, sr, subtype = _load(input_path)

    if audio.ndim == 1:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    else:
        channels = []
        for ch in range(audio.shape[1]):
            shifted_ch = librosa.effects.pitch_shift(audio[:, ch], sr=sr, n_steps=semitones)
            channels.append(shifted_ch)
        shifted = np.column_stack(channels)

    _save(shifted, output_path, sr, subtype)
    return {"semitones": semitones}
