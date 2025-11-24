"""Conversation turn data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def _coerce_timestamp(value: Optional[Any]) -> datetime:
    """Convert supported timestamp inputs to a timezone-aware ``datetime``."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        parsed = value.strip()
        if not parsed:
            return datetime.now(timezone.utc)
        parsed = parsed.replace("Z", "+00:00")
        dt = datetime.fromisoformat(parsed)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    raise TypeError("timestamp must be datetime, ISO string, or None")


def _normalize_language_map(languages: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Normalize language mapping keys and values."""
    if not languages:
        return {}
    normalized: Dict[str, str] = {}
    for key, value in languages.items():
        if value:
            normalized[str(key)] = str(value).lower()
    return normalized


def _normalize_translations(translations: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Normalize translation mapping keys and values."""
    if not translations:
        return {}
    normalized: Dict[str, str] = {}
    for key, value in translations.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""

    turn_id: str
    user_utterance_text: str
    languages: Dict[str, str] = field(default_factory=dict)
    translations: Dict[str, str] = field(default_factory=dict)
    assistant_response: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.turn_id = str(self.turn_id)
        self.user_utterance_text = (self.user_utterance_text or "").strip()
        if not self.user_utterance_text:
            raise ValueError("user_utterance_text must be a non-empty string")
        self.languages = _normalize_language_map(self.languages)
        self.translations = _normalize_translations(self.translations)
        self.assistant_response = (self.assistant_response or "").strip() or None
        self.timestamp = _coerce_timestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this turn into a JSON-serializable dictionary."""
        return {
            "turn_id": self.turn_id,
            "user_utterance_text": self.user_utterance_text,
            "languages": dict(self.languages),
            "translations": dict(self.translations),
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConversationTurn":
        """Create a :class:`ConversationTurn` from a dictionary payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            turn_id=payload.get("turn_id", ""),
            user_utterance_text=payload.get("user_utterance_text", ""),
            languages=payload.get("languages"),
            translations=payload.get("translations"),
            assistant_response=payload.get("assistant_response"),
            timestamp=payload.get("timestamp"),
        )

