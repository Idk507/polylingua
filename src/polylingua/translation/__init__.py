"""Translation service abstractions."""

from .service import BaseTranslator, HuggingFaceTranslator, TranslationResult, TranslationService

__all__ = [
    "TranslationResult",
    "BaseTranslator",
    "HuggingFaceTranslator",
    "TranslationService",
]
