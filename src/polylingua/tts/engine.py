"""Text-to-speech engines and routing."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional heavy dependencies
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
except Exception:  # pragma: no cover - allow tests without heavy deps
    torch = None  # type: ignore
    hf_hub_download = None  # type: ignore
    SpeechT5ForTextToSpeech = None  # type: ignore
    SpeechT5HifiGan = None  # type: ignore
    SpeechT5Processor = None  # type: ignore


@dataclass(slots=True)
class TtsRequest:
    """Represents the payload forwarded to a TTS engine."""

    text: str
    language: str = "en"
    voice_id: Optional[str] = None
    speed: float = 1.0

    def __post_init__(self) -> None:
        self.text = (self.text or "").strip()
        if not self.text:
            raise ValueError("TtsRequest.text must be a non-empty string")
        if self.speed <= 0:
            raise ValueError("TtsRequest.speed must be greater than zero")
        self.language = (self.language or "en").lower()
        if self.voice_id:
            self.voice_id = self.voice_id.lower()


@dataclass(slots=True)
class TtsResult:
    """Payload returned by text-to-speech engines."""

    audio_bytes: bytes
    audio_format: str
    duration: float

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(self.audio_bytes)


class BaseTtsEngine(ABC):
    """Abstract base class for text-to-speech engines."""

    name: str = "base"

    @abstractmethod
    def synthesize(self, tts_request: TtsRequest) -> TtsResult:
        raise NotImplementedError

    def supports_language(self, language: str) -> bool:
        return True


class SpeechT5Engine(BaseTtsEngine):
    """Hugging Face SpeechT5 implementation supporting English voices."""

    name = "speecht5"
    _AVAILABLE_VOICES = {"awb", "clb", "rms", "slt", "bdl"}

    def __init__(self, default_voice: str = "slt", device: Optional[str] = None) -> None:
        if torch is None or SpeechT5Processor is None:
            raise RuntimeError("transformers SpeechT5 dependencies are not installed")
        self.default_voice = default_voice if default_voice in self._AVAILABLE_VOICES else "slt"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_models(device: str):  # pragma: no cover - heavy load
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        return processor, model, vocoder

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_speaker_embeddings() -> Dict[str, np.ndarray]:  # pragma: no cover - heavy load
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is not installed")
        archive_path = hf_hub_download(repo_id="Matthijs/cmu-arctic-xvectors", filename="spkrec-xvect.zip")
        import zipfile

        speaker_vectors: Dict[str, list[np.ndarray]] = {voice: [] for voice in SpeechT5Engine._AVAILABLE_VOICES}
        with zipfile.ZipFile(archive_path, "r") as archive:
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    continue
                voice = name.split("/")[-1].split("-")[0]
                if voice in speaker_vectors:
                    with archive.open(name) as handle:
                        data = np.load(handle)
                        if isinstance(data, np.ndarray):
                            speaker_vectors[voice].append(data.astype(np.float32))
        averaged: Dict[str, np.ndarray] = {}
        for voice, vectors in speaker_vectors.items():
            if not vectors:
                continue
            stacked = np.stack(vectors, axis=0)
            averaged[voice] = stacked.mean(axis=0, keepdims=True)
        if not averaged:
            raise RuntimeError("No speaker embeddings extracted from archive")
        return averaged

    def synthesize(self, tts_request: TtsRequest) -> TtsResult:  # pragma: no cover - heavy load
        processor, model, vocoder = self._load_models(self.device)
        embeddings = self._load_speaker_embeddings()
        voice = tts_request.voice_id or self.default_voice
        if voice not in embeddings:
            raise ValueError(f"Voice '{voice}' is not available")
        speaker_embedding = torch.from_numpy(embeddings[voice]).to(model.device)
        inputs = processor(text=tts_request.text, return_tensors="pt")
        generated_speech = model.generate_speech(
            inputs["input_ids"].to(model.device),
            speaker_embedding,
            vocoder=vocoder,
        )
        audio = generated_speech.cpu().numpy().astype(np.float32)
        duration = audio.shape[-1] / self.sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, subtype="PCM_16")
        return TtsResult(audio_bytes=buffer.getvalue(), audio_format="wav", duration=duration)


class TtsRouterService:
    """Route TTS requests across registered engines."""

    def __init__(self, engines: Optional[Dict[str, BaseTtsEngine]] = None, default_engine: Optional[str] = None) -> None:
        self.engines: Dict[str, BaseTtsEngine] = engines or {}
        self.default_engine = default_engine or next(iter(self.engines.keys()), None)

    def register_engine(self, engine: BaseTtsEngine) -> None:
        self.engines[engine.name] = engine
        if not self.default_engine:
            self.default_engine = engine.name

    def synthesize_with_best_engine(
        self,
        request: TtsRequest,
        engine_name: Optional[str] = None,
    ) -> TtsResult:
        key = engine_name or self.default_engine
        if key is None:
            raise RuntimeError("No TTS engine registered")
        if key not in self.engines:
            raise ValueError(f"Unknown TTS engine: {key}")
        engine = self.engines[key]
        return engine.synthesize(request)

```