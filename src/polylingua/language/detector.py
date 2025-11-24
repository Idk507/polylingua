"""Language detection helpers relying on langdetect/lingua."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs

try:  # pragma: no cover - optional dependency
    from lingua import Language, LanguageDetectorBuilder
except Exception:  # pragma: no cover - lingua may be unavailable
    Language = None  # type: ignore
    LanguageDetectorBuilder = None  # type: ignore

DetectorFactory.seed = 0


def normalize_language_code(raw_code: Optional[str]) -> Optional[str]:
    if not raw_code:
        return None
    normalized = raw_code.strip().lower()
    if not normalized:
        return None
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    return normalized


@dataclass(slots=True)
class LanguageDetectionResult:
    """Represents the outcome of a language detection attempt."""

    language_code: Optional[str]
    confidence: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "language_code": self.language_code,
            "confidence": self.confidence,
            "source": self.source,
        }


class LanguageDetector:
    """Detect languages from text using lingua when available, otherwise langdetect."""

    def __init__(self, use_lingua: bool = True) -> None:
        self._lingua_detector = None
        if use_lingua and LanguageDetectorBuilder is not None:
            languages = list(Language) if Language else []
            if languages:
                self._lingua_detector = (
                    LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().build()
                )

    def detect_from_text(self, text: str) -> LanguageDetectionResult:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Text input for language detection must be non-empty")
        if self._lingua_detector is not None:
            language = self._lingua_detector.detect_language_of(cleaned)
            confidence = self._lingua_detector.compute_language_confidence(language, cleaned)
            language_code = normalize_language_code(language.iso_code_639_1.name if language else None)
            return LanguageDetectionResult(language_code=language_code, confidence=confidence, source="lingua")
        try:
            candidates = detect_langs(cleaned)
        except LangDetectException as exc:  # pragma: no cover - rare edge cases
            raise ValueError("Unable to detect language") from exc
        best = candidates[0]
        language_code = normalize_language_code(best.lang)
        return LanguageDetectionResult(language_code=language_code, confidence=best.prob, source="langdetect")

