"""Lightweight persistence for language preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .detector import normalize_language_code


@dataclass(slots=True)
class StoredPreference:
    language_code: str
    confidence: float
    timestamp: str
    count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "language_code": self.language_code,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "count": self.count,
        }


class LanguagePreferenceManager:
    """Persist user language preferences on disk for conversational continuity."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._preferences: Dict[str, Dict[str, StoredPreference]] = {}
        if self._storage_path and self._storage_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            payload = {}
        for user_id, languages in payload.items():
            self._preferences[user_id] = {}
            for language_code, entry in languages.items():
                normalized = normalize_language_code(language_code)
                if not normalized:
                    continue
                self._preferences[user_id][normalized] = StoredPreference(
                    language_code=normalized,
                    confidence=float(entry.get("confidence", 0.0)),
                    timestamp=str(entry.get("timestamp", "")),
                    count=int(entry.get("count", 0)),
                )

    def _persist(self) -> None:
        if not self._storage_path:
            return
        payload: Dict[str, Dict[str, Dict[str, object]]] = {}
        for user_id, languages in self._preferences.items():
            payload[user_id] = {code: pref.to_dict() for code, pref in languages.items()}
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def update_preference(self, user_id: str, language_code: str, confidence: float) -> None:
        if not user_id:
            raise ValueError("user_id must be provided")
        normalized = normalize_language_code(language_code)
        if not normalized:
            raise ValueError("language_code must be provided")
        store = self._preferences.setdefault(user_id, {})
        existing = store.get(normalized)
        timestamp = datetime.now(timezone.utc).isoformat()
        if existing:
            existing.confidence = max(existing.confidence, confidence)
            existing.timestamp = timestamp
            existing.count += 1
        else:
            store[normalized] = StoredPreference(
                language_code=normalized,
                confidence=confidence,
                timestamp=timestamp,
                count=1,
            )
        self._persist()

    def get_preference(self, user_id: str) -> Optional[str]:
        languages = self._preferences.get(user_id)
        if not languages:
            return None
        best = max(languages.values(), key=lambda pref: (pref.count, pref.confidence))
        return best.language_code

