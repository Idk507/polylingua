"""Translation primitives with pluggable providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from abc import ABC, abstractmethod
import logging

try:  # pragma: no cover - allow running without transformers installed
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - transformers optional for tests
    hf_pipeline = None


@dataclass(slots=True)
class TranslationResult:
    """Represents the payload returned by translation engines."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "provider": self.provider,
        }


class BaseTranslator(ABC):
    """Abstract base interface for concrete translation providers."""

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        raise NotImplementedError


class HuggingFaceTranslator(BaseTranslator):
    """Helsinki-NLP opus-mt powered translator from Hugging Face."""

    def __init__(self, model_template: str = "Helsinki-NLP/opus-mt-{src}-{tgt}", max_length: int = 400) -> None:
        self.model_template = model_template
        self.max_length = max_length
        self._pipelines: Dict[str, object] = {}

    def _normalize(self, lang: str) -> str:
        if not lang:
            raise ValueError("Language codes must be provided")
        return lang.lower()

    def _model_name_for(self, source_lang: str, target_lang: str) -> str:
        return self.model_template.format(src=self._normalize(source_lang), tgt=self._normalize(target_lang))

    def _get_pipeline(self, source_lang: str, target_lang: str):
        key = f"{self._normalize(source_lang)}-{self._normalize(target_lang)}"
        if key not in self._pipelines:
            if hf_pipeline is None:
                raise RuntimeError("transformers is not installed")
            model_name = self._model_name_for(source_lang, target_lang)
            logging.info("Loading translation model %s", model_name)
            self._pipelines[key] = hf_pipeline("translation", model=model_name, max_length=self.max_length)
        return self._pipelines[key]

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        if not text or not text.strip():
            raise ValueError("Text to translate must be a non-empty string")
        normalized_source = self._normalize(source_lang)
        normalized_target = self._normalize(target_lang)
        if normalized_source == normalized_target:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=normalized_source,
                target_lang=normalized_target,
                provider="noop",
            )
        translator_pipeline = self._get_pipeline(normalized_source, normalized_target)
        result = translator_pipeline(text)[0]
        translated_text = result.get("translation_text", text)
        return TranslationResult(
            source_text=text,
            translated_text=translated_text,
            source_lang=normalized_source,
            target_lang=normalized_target,
            provider="huggingface",
        )


class TranslationService:
    """High-level translation orchestrator supporting multiple providers."""

    def __init__(
        self,
        translators: Optional[Dict[str, BaseTranslator]] = None,
        default_provider: str = "huggingface",
        language_detector: Optional[object] = None,
        internal_language: str = "en",
    ) -> None:
        self.translators = translators or {"huggingface": HuggingFaceTranslator()}
        self.default_provider = default_provider
        self.language_detector = language_detector
        self.internal_language = internal_language.lower()

    def translate(
        self, text: str, source_lang: str, target_lang: str, provider: Optional[str] = None
    ) -> TranslationResult:
        provider_key = provider or self.default_provider
        if provider_key not in self.translators:
            raise ValueError(f"Unknown translation provider: {provider_key}")
        translator = self.translators[provider_key]
        return translator.translate(text, source_lang, target_lang)

    def translate_to_internal_language(
        self, text: str, detected_lang: str, internal_lang: Optional[str] = None
    ) -> TranslationResult:
        internal_target = (internal_lang or self.internal_language).lower()
        detected_lang = detected_lang.lower()
        if detected_lang == internal_target:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=detected_lang,
                target_lang=internal_target,
                provider="noop",
            )
        return self.translate(text, detected_lang, internal_target)

    def translate_to_user_language(
        self, text: str, user_lang: str, internal_lang: Optional[str] = None
    ) -> TranslationResult:
        source_lang = (internal_lang or self.internal_language).lower()
        target_lang = user_lang.lower()
        if source_lang == target_lang:
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                provider="noop",
            )
        return self.translate(text, source_lang, target_lang)

    def detect_and_translate_auto(self, text: str, target_lang: str) -> TranslationResult:
        if not self.language_detector:
            raise RuntimeError("No language detector configured")
        detection = self.language_detector.detect_from_text(text)
        source_lang = detection.language_code or self.internal_language
        return self.translate(text, source_lang, target_lang)

```