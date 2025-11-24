from __future__ import annotations

from typing import Optional

from polylingua.dialogue.engine import DialogueEngine
from polylingua.dialogue.manager import (
    ConversationManager,
    SttOutput,
    TtsOutput,
    TtsRequest,
    TranslationOutput,
)


class DummyTranslationService:
    def __init__(self) -> None:
        self.internal_calls: list[tuple[str, str, str]] = []
        self.user_calls: list[tuple[str, str, str]] = []

    def translate_to_internal_language(
        self,
        text: str,
        detected_lang: str,
        internal_lang: Optional[str] = None,
    ) -> TranslationOutput:
        self.internal_calls.append((text, detected_lang, internal_lang or ""))
        translated = f"{text} in {internal_lang}" if internal_lang else text
        return TranslationOutput(
            source_text=text,
            translated_text=translated,
            source_lang=detected_lang,
            target_lang=internal_lang or detected_lang,
            provider="dummy",
        )

    def translate_to_user_language(
        self,
        text: str,
        user_lang: str,
        internal_lang: Optional[str] = None,
    ) -> TranslationOutput:
        self.user_calls.append((text, user_lang, internal_lang or ""))
        translated = f"{text} in {user_lang}"
        return TranslationOutput(
            source_text=text,
            translated_text=translated,
            source_lang=internal_lang or "",
            target_lang=user_lang,
            provider="dummy",
        )


class DummyTtsService:
    def __init__(self) -> None:
        self.requests: list[TtsRequest] = []

    def synthesize_with_best_engine(self, request: TtsRequest) -> TtsOutput:
        self.requests.append(request)
        return TtsOutput(audio_bytes=b"data", audio_format="wav", duration=1.0)


class DummySttService:
    def __init__(self, result: SttOutput) -> None:
        self.result = result
        self.calls = 0

    def transcribe_with_best_engine(
        self,
        _audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> SttOutput:
        self.calls += 1
        _ = language_hint
        return self.result


class DummyAudioProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def prepare_for_stt(self, audio_bytes: bytes):
        self.calls += 1
        return b"processed" + audio_bytes, {"sample_rate": "16000"}


def test_handle_text_turn_translates_and_tracks_state() -> None:
    translation = DummyTranslationService()
    tts = DummyTtsService()
    engine = DialogueEngine()
    manager = ConversationManager(
        translation_service=translation,
        dialogue_engine=engine,
        tts_service=tts,
        internal_language="en",
    )
    result = manager.handle_text_turn(
        session_id="abc",
        user_text="hola",
        source_language="es",
        target_language="es",
    )
    turn = result.turn
    assert turn.assistant_response.startswith("Acknowledged")
    assert translation.internal_calls
    assert translation.user_calls
    assert tts.requests and tts.requests[0].text == turn.assistant_response
    state = manager.get_or_create_state("abc")
    assert state.turn_count == 1
    assert state.user_language_preference == "es"


def test_handle_audio_turn_uses_stt_and_audio_processor() -> None:
    translation = DummyTranslationService()
    stt = DummySttService(
        SttOutput(text="bonjour", language="fr", confidence=0.9)
    )
    audio_processor = DummyAudioProcessor()
    engine = DialogueEngine()
    manager = ConversationManager(
        translation_service=translation,
        dialogue_engine=engine,
        stt_service=stt,
        audio_processor=audio_processor,
        internal_language="en",
    )
    result = manager.handle_audio_turn(
        session_id="session",
        audio_bytes=b"raw",
        language_hint="fr",
        target_language="en",
    )
    turn = result.turn
    assert stt.calls == 1
    assert audio_processor.calls == 1
    # translation to internal invoked for non-en source
    assert translation.internal_calls
    assert "stt_meta:confidence" in turn.translations
    assert turn.languages["user"] == "fr"
    assert turn.languages["assistant"] == "en"

