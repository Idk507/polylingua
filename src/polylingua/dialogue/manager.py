"""High-level conversation management orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Protocol, Tuple

from .engine import DialogueEngine
from .state import ConversationState
from .turn import ConversationTurn


@dataclass(slots=True)
class SttOutput:
    """Simple container representing a speech-to-text transcription."""

    text: str
    language: Optional[str]
    confidence: Optional[float] = None


@dataclass(slots=True)
class TranslationOutput:
    """Simple container for translation responses."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str = "unknown"


@dataclass(slots=True)
class TtsRequest:
    """Minimal request payload shared with TTS services."""

    text: str
    language: str
    voice_id: Optional[str] = None


@dataclass(slots=True)
class TtsOutput:
    """Audio payload returned by text-to-speech engines."""

    audio_bytes: bytes
    audio_format: str
    duration: float


class SupportsStt(Protocol):
    """Protocol describing the STT service we rely on."""

    def transcribe_with_best_engine(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> SttOutput: ...


class SupportsTranslation(Protocol):
    """Protocol describing the translation service dependency."""

    def translate_to_internal_language(self, text: str, detected_lang: str, internal_lang: Optional[str] = None) -> TranslationOutput: ...

    def translate_to_user_language(self, text: str, user_lang: str, internal_lang: Optional[str] = None) -> TranslationOutput: ...


class SupportsTts(Protocol):
    """Protocol describing the TTS service dependency."""

    def synthesize_with_best_engine(self, request: TtsRequest) -> TtsOutput: ...


class SupportsAudioProcessor(Protocol):
    """Protocol for preprocessing raw audio prior to STT."""

    def prepare_for_stt(self, audio_bytes: bytes) -> Tuple[bytes, Mapping[str, str]]: ...


@dataclass(slots=True)
class TurnResult:
    """Bundle returned by :class:`ConversationManager` operations."""

    turn: ConversationTurn
    tts: Optional[TtsOutput] = None


class ConversationManager:
    """Coordinate STT, translation, dialogue, and TTS for conversations."""

    def __init__(
        self,
        translation_service: SupportsTranslation,
        dialogue_engine: DialogueEngine,
        stt_service: Optional[SupportsStt] = None,
        tts_service: Optional[SupportsTts] = None,
        audio_processor: Optional[SupportsAudioProcessor] = None,
        internal_language: str = "en",
    ) -> None:
        self.translation_service = translation_service
        self.dialogue_engine = dialogue_engine
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.audio_processor = audio_processor
        self.internal_language = internal_language.lower()
        self._conversations: Dict[str, ConversationState] = {}

    def get_or_create_state(self, session_id: str) -> ConversationState:
        session_id = str(session_id)
        if session_id not in self._conversations:
            self._conversations[session_id] = ConversationState(
                session_id=session_id,
                internal_language=self.internal_language,
            )
        return self._conversations[session_id]

    def handle_audio_turn(
        self,
        session_id: str,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> TurnResult:
        if not self.stt_service:
            raise RuntimeError("No STT service configured")
        prepared_audio = audio_bytes
        if self.audio_processor:
            prepared_audio, _ = self.audio_processor.prepare_for_stt(audio_bytes)
        stt_output = self.stt_service.transcribe_with_best_engine(
            prepared_audio,
            language_hint=language_hint,
        )
        detected_language = (stt_output.language or language_hint or self.internal_language).lower()
        return self.handle_text_turn(
            session_id=session_id,
            user_text=stt_output.text,
            source_language=detected_language,
            target_language=target_language,
            stt_metadata={"confidence": stt_output.confidence},
        )

    def handle_text_turn(
        self,
        session_id: str,
        user_text: str,
        source_language: Optional[str],
        target_language: Optional[str] = None,
        stt_metadata: Optional[Mapping[str, object]] = None,
    ) -> TurnResult:
        state = self.get_or_create_state(session_id)
        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("user_text must be a non-empty string")
        source_language = (source_language or state.user_language_preference or self.internal_language).lower()
        target_language = (target_language or state.user_language_preference or source_language).lower()
        translation_to_internal: Optional[TranslationOutput] = None
        normalized_text = user_text
        if source_language != state.internal_language:
            translation_to_internal = self.translation_service.translate_to_internal_language(
                user_text,
                detected_lang=source_language,
                internal_lang=state.internal_language,
            )
            normalized_text = translation_to_internal.translated_text
        assistant_internal = self.dialogue_engine.generate_response(
            state,
            normalized_text,
            user_language=source_language,
        )
        assistant_user = assistant_internal
        translation_back: Optional[TranslationOutput] = None
        if target_language != state.internal_language:
            translation_back = self.translation_service.translate_to_user_language(
                assistant_internal,
                user_lang=target_language,
                internal_lang=state.internal_language,
            )
            assistant_user = translation_back.translated_text
        languages = {
            "user": source_language,
            "assistant": target_language,
            "internal": state.internal_language,
        }
        translations = {}
        if translation_to_internal:
            translations[f"internal:{state.internal_language}"] = translation_to_internal.translated_text
        if translation_back:
            translations[f"assistant:{target_language}"] = translation_back.translated_text
        turn = ConversationTurn(
            turn_id=f"{session_id}-{state.turn_count + 1}",
            user_utterance_text=user_text,
            languages=languages,
            translations=translations,
            assistant_response=assistant_user,
        )
        if stt_metadata:
            turn.translations.update({f"stt_meta:{key}": str(value) for key, value in stt_metadata.items() if value is not None})
        state.add_turn(turn)
        if target_language and target_language != state.user_language_preference:
            state.user_language_preference = target_language
        tts_output: Optional[TtsOutput] = None
        if self.tts_service:
            tts_output = self.tts_service.synthesize_with_best_engine(
                TtsRequest(text=assistant_user, language=target_language, voice_id=None)
            )
        return TurnResult(turn=turn, tts=tts_output)

    def clear(self, session_id: str) -> None:
        """Remove cached state for a session."""
        self._conversations.pop(str(session_id), None)

    def list_active_sessions(self) -> Iterable[str]:
        return tuple(self._conversations.keys())

