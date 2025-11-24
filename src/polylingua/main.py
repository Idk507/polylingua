"""FastAPI application wiring PolyLingua services together."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .audio import AudioProcessor
from .dialogue import ConversationManager, DialogueEngine
from .language import LanguageDetector, LanguagePreferenceManager
from .translation import (
    BaseTranslator,
    TranslationResult,
    TranslationService,
    HuggingFaceTranslator,
)
from .tts import BaseTtsEngine, TtsRequest, TtsResult, TtsRouterService
from .stt import BaseSttEngine, SttResult, SttRouterService


class _PassthroughTranslator(BaseTranslator):
    """Fallback translator used when transformers is unavailable."""

    name = "noop"

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        cleaned = (text or "").strip()
        return TranslationResult(
            source_text=text,
            translated_text=cleaned,
            source_lang=source_lang.lower(),
            target_lang=target_lang.lower(),
            provider="noop",
        )


class _EchoSttEngine(BaseSttEngine):
    """Simple STT fallback that returns a placeholder transcription."""

    name = "echo"

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> SttResult:
        del audio_bytes
        return SttResult(
            text="[stt unavailable]",
            language=language_hint,
            confidence=None,
        )


class _SilenceTtsEngine(BaseTtsEngine):
    """Lightweight TTS fallback that returns silence."""

    name = "silence"

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, tts_request: TtsRequest) -> TtsResult:
        duration = max(
            len(tts_request.text) * 0.05 / max(tts_request.speed, 1e-6),
            0.5,
        )
        samples = int(duration * self.sample_rate)
        waveform = np.zeros(samples, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.sample_rate, subtype="PCM_16")
        return TtsResult(
            audio_bytes=buffer.getvalue(),
            audio_format="wav",
            duration=duration,
        )

    def supports_language(self, language: str) -> bool:
        return language.lower() == "en"


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DetectLanguageResponse(BaseModel):
    language_code: Optional[str]
    confidence: float
    source: str


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: str = Field(..., min_length=2)
    target_lang: str = Field(..., min_length=2)
    provider: Optional[str] = None


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str


class TextTurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source_language: Optional[str] = None
    target_language: Optional[str] = None


class TextTurnResponse(BaseModel):
    turn: dict
    tts_audio_base64: Optional[str]


class AudioTurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    audio_base64: str = Field(..., min_length=1)
    language_hint: Optional[str] = None
    target_language: Optional[str] = None


class AudioTurnResponse(BaseModel):
    turn: dict
    tts_audio_base64: Optional[str]


class TtsSynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("en", min_length=2)
    voice_id: Optional[str] = None
    speed: float = Field(1.0, gt=0)


class TtsSynthesisResponse(BaseModel):
    audio_base64: str
    audio_format: str
    duration: float


class SttRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    language_hint: Optional[str] = None


class SttResponse(BaseModel):
    text: str
    language: Optional[str]
    confidence: Optional[float]


def _build_translation_service(
    detector: LanguageDetector,
) -> TranslationService:
    translators: dict[str, BaseTranslator] = {"noop": _PassthroughTranslator()}
    default_provider = "noop"
    try:
        translators["huggingface"] = HuggingFaceTranslator()
        default_provider = "huggingface"
    except Exception:
        pass
    return TranslationService(
        translators=translators,
        default_provider=default_provider,
        language_detector=detector,
        internal_language="en",
    )


def _build_stt_router() -> SttRouterService:
    router = SttRouterService()
    router.register_engine(_EchoSttEngine())
    return router


def _build_tts_router() -> TtsRouterService:
    router = TtsRouterService()
    router.register_engine(_SilenceTtsEngine())
    return router


def create_app() -> FastAPI:
    app = FastAPI(
        title="PolyLingua",
        description="A Multilingual Intelligent Speech Assistant",
    )

    detector = LanguageDetector(use_lingua=False)
    preferences_path = (
        Path(__file__).resolve().parents[2] / "language_preferences.json"
    )
    preference_manager = LanguagePreferenceManager(preferences_path)
    translation_service = _build_translation_service(detector)
    stt_router = _build_stt_router()
    tts_router = _build_tts_router()
    audio_processor = AudioProcessor()
    dialogue_engine = DialogueEngine()
    conversation_manager = ConversationManager(
        translation_service=translation_service,
        dialogue_engine=dialogue_engine,
        stt_service=stt_router,
        tts_service=tts_router,
        audio_processor=audio_processor,
        internal_language="en",
    )

    app.state.language_detector = detector
    app.state.translation_service = translation_service
    app.state.language_preferences = preference_manager
    app.state.audio_processor = audio_processor
    app.state.stt_router = stt_router
    app.state.tts_router = tts_router
    app.state.conversation_manager = conversation_manager

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/detect-language", response_model=DetectLanguageResponse)
    async def detect_language(
        payload: DetectLanguageRequest,
    ) -> DetectLanguageResponse:
        result = detector.detect_from_text(payload.text)
        return DetectLanguageResponse(
            language_code=result.language_code,
            confidence=result.confidence,
            source=result.source,
        )

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(payload: TranslateRequest) -> TranslateResponse:
        try:
            result = translation_service.translate(
                payload.text,
                payload.source_lang,
                payload.target_lang,
                provider=payload.provider,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return TranslateResponse(
            source_text=result.source_text,
            translated_text=result.translated_text,
            source_lang=result.source_lang,
            target_lang=result.target_lang,
            provider=result.provider,
        )

    @app.post("/tts", response_model=TtsSynthesisResponse)
    async def synthesize_tts(
        payload: TtsSynthesisRequest,
    ) -> TtsSynthesisResponse:
        try:
            request = TtsRequest(
                text=payload.text,
                language=payload.language,
                voice_id=payload.voice_id,
                speed=payload.speed,
            )
            result = tts_router.synthesize_with_best_engine(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        audio_base64 = base64.b64encode(result.audio_bytes).decode("utf-8")
        return TtsSynthesisResponse(
            audio_base64=audio_base64,
            audio_format=result.audio_format,
            duration=result.duration,
        )

    @app.post("/stt", response_model=SttResponse)
    async def transcribe(payload: SttRequest) -> SttResponse:
        try:
            audio_bytes = base64.b64decode(payload.audio_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio_base64 data",
            ) from exc
        result = stt_router.transcribe_with_best_engine(
            audio_bytes,
            language_hint=payload.language_hint,
        )
        return SttResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
        )

    @app.post("/turn/text", response_model=TextTurnResponse)
    async def handle_text_turn(payload: TextTurnRequest) -> TextTurnResponse:
        result = conversation_manager.handle_text_turn(
            session_id=payload.session_id,
            user_text=payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
        user_lang = result.turn.languages.get("user")
        if user_lang and preference_manager:
            preference_manager.update_preference(
                payload.session_id,
                user_lang,
                confidence=1.0,
            )
        audio_payload = None
        if result.tts:
            audio_payload = base64.b64encode(
                result.tts.audio_bytes
            ).decode("utf-8")
        return TextTurnResponse(
            turn=result.turn.to_dict(),
            tts_audio_base64=audio_payload,
        )

    @app.post("/turn/audio", response_model=AudioTurnResponse)
    async def handle_audio_turn(
        payload: AudioTurnRequest,
    ) -> AudioTurnResponse:
        try:
            audio_bytes = base64.b64decode(payload.audio_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid audio_base64 data",
            ) from exc
        result = conversation_manager.handle_audio_turn(
            session_id=payload.session_id,
            audio_bytes=audio_bytes,
            language_hint=payload.language_hint,
            target_language=payload.target_language,
        )
        user_lang = result.turn.languages.get("user")
        if user_lang and preference_manager:
            confidence_str = result.turn.translations.get(
                "stt_meta:confidence"
            )
            confidence_value = float(confidence_str) if confidence_str else 1.0
            preference_manager.update_preference(
                payload.session_id,
                user_lang,
                confidence=confidence_value,
            )
        audio_payload = None
        if result.tts:
            audio_payload = base64.b64encode(
                result.tts.audio_bytes
            ).decode("utf-8")
        return AudioTurnResponse(
            turn=result.turn.to_dict(),
            tts_audio_base64=audio_payload,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

