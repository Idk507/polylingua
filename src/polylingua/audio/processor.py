"""Audio normalization helpers used across PolyLingua."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass(slots=True)
class AudioFormat:
    """Represents audio characteristics used by downstream services."""

    sample_rate: int
    channels: int
    sample_width: int


class AudioProcessor:
    """Prepare raw audio bytes so STT engines receive consistent input."""

    def __init__(self, target_sample_rate: int = 16_000, target_peak_db: float = -20.0) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_peak_db = target_peak_db

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        buffer = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(buffer, always_2d=False)
        if data.ndim == 1:
            data = data.astype(np.float32)
        else:
            data = data.astype(np.float32)
        return data, sample_rate

    def resample(self, audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio
        return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate)

    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        return audio.mean(axis=1)

    def normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio)) or 1.0
        normalized = audio / peak
        target_linear = 10 ** (self.target_peak_db / 20)
        return normalized * target_linear

    def export_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, subtype="PCM_16")
        return buffer.getvalue()

    def validate_audio_duration(self, audio: np.ndarray, sample_rate: int, max_seconds: float = 30.0) -> None:
        duration = audio.shape[0] / max(sample_rate, 1)
        if duration <= 0.0:
            raise ValueError("Audio clip must have a non-zero duration")
        if duration > max_seconds:
            raise ValueError("Audio clip exceeds maximum duration allowed")

    def prepare_for_stt(self, audio_bytes: bytes) -> Tuple[bytes, Dict[str, str]]:
        audio, sample_rate = self.load_audio_from_bytes(audio_bytes)
        mono = self.convert_to_mono(audio)
        resampled = self.resample(mono, source_rate=sample_rate, target_rate=self.target_sample_rate)
        normalized = self.normalize_volume(resampled)
        self.validate_audio_duration(normalized, self.target_sample_rate)
        wav_bytes = self.export_to_wav_bytes(normalized, self.target_sample_rate)
        metadata = {
            "sample_rate": str(self.target_sample_rate),
            "channels": "1",
            "sample_width": "2",
        }
        return wav_bytes, metadata
