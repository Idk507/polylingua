"""Language detection utilities."""

from .detector import LanguageDetectionResult, LanguageDetector, normalize_language_code
from .preferences import LanguagePreferenceManager

__all__ = [
    "LanguageDetectionResult",
    "LanguageDetector",
    "LanguagePreferenceManager",
    "normalize_language_code",
]
