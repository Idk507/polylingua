"""Conversation state management primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .turn import ConversationTurn


@dataclass(slots=True)
class ConversationState:
    """Represents the mutable state of an ongoing conversation session."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    user_language_preference: Optional[str] = None
    internal_language: str = "en"
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = str(self.session_id)
        self.internal_language = (self.internal_language or "en").lower()
        if self.user_language_preference:
            self.user_language_preference = self.user_language_preference.lower()

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a new turn to the conversation history."""
        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn instance")
        self.turns.append(turn)

    def extend(self, turns: Iterable[ConversationTurn]) -> None:
        """Extend the conversation history with multiple turns."""
        for turn in turns:
            self.add_turn(turn)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    def recent_turns(self, limit: int = 5) -> List[ConversationTurn]:
        """Return the most recent ``limit`` turns (oldest to newest)."""
        if limit <= 0:
            return []
        return self.turns[-limit:]

    def to_dict(self) -> Dict[str, object]:
        """Serialize the conversation state for persistence."""
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "user_language_preference": self.user_language_preference,
            "internal_language": self.internal_language,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ConversationState":
        """Rehydrate a :class:`ConversationState` from a dictionary."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        turns_payload = payload.get("turns") or []
        turns: List[ConversationTurn] = []
        for turn_payload in turns_payload:
            turns.append(ConversationTurn.from_dict(turn_payload))
        return cls(
            session_id=payload.get("session_id", ""),
            turns=turns,
            user_language_preference=payload.get("user_language_preference"),
            internal_language=payload.get("internal_language", "en"),
            metadata=dict(payload.get("metadata") or {}),
        )

