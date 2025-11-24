"""Speech-to-text engine registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from abc import ABC, abstractmethod

try:  # pragma: no cover - optional heavy dependency
    import whisper
except Exception:  # pragma: no cover - optional dependency not installed
    whisper = None  # type: ignore


@dataclass(slots=True)
class SttResult:
    """Represents the output of a speech-to-text transcription."""

    text: str
    language: Optional[str]
    confidence: Optional[float] = None


class BaseSttEngine(ABC):
    """Abstract interface for STT providers."""

    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> SttResult:
        raise NotImplementedError


class WhisperSttEngine(BaseSttEngine):
    """Wrapper around openai-whisper models."""

    name = "whisper"

    def __init__(self, model_name: str = "base", device: Optional[str] = None) -> None:
        if whisper is None:
            raise RuntimeError("openai-whisper is not installed")
        self._model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_bytes: bytes, language_hint: Optional[str] = None) -> SttResult:
        import numpy as np  # local import to avoid hard dependency unless used
        import soundfile as sf
        import io

        buffer = io.BytesIO(audio_bytes)
        audio_array, _ = sf.read(buffer)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        result = self._model.transcribe(audio_array, language=language_hint)
        text = result.get("text", "").strip()
        language = result.get("language", language_hint)
        confidence = result.get("confidence")
        return SttResult(text=text, language=language, confidence=confidence)


class SttRouterService:
    """Route STT requests across registered engines."""

    def __init__(self, engines: Optional[Dict[str, BaseSttEngine]] = None, default_engine: Optional[str] = None) -> None:
        self.engines: Dict[str, BaseSttEngine] = engines or {}
        self.default_engine = default_engine or next(iter(self.engines.keys()), None)

    def register_engine(self, engine: BaseSttEngine) -> None:
        self.engines[engine.name] = engine
        if not self.default_engine:
            self.default_engine = engine.name

    def transcribe_with_best_engine(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> SttResult:
        key = engine_name or self.default_engine
        if key is None:
            raise RuntimeError("No STT engine registered")
        if key not in self.engines:
            raise ValueError(f"Unknown STT engine: {key}")
        engine = self.engines[key]
        return engine.transcribe(audio_bytes, language_hint=language_hint)

```