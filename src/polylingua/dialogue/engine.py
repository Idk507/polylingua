"""Lightweight dialogue engine helpers."""

from __future__ import annotations

from typing import Iterable

from .state import ConversationState
from .turn import ConversationTurn


class DialogueEngine:
    """Provide simple conversation utilities such as summarization."""

    def __init__(self, max_summary_turns: int = 10) -> None:
        self.max_summary_turns = max(1, max_summary_turns)

    def summarize_conversation(self, state: ConversationState) -> str:
        """Return a plain-text summary of recent conversation turns."""
        turns = state.recent_turns(self.max_summary_turns)
        if not turns:
            return "No conversation history available."
        return "\n".join(self._format_turns_for_summary(turns))

    def generate_response(
        self,
        state: ConversationState,
        user_text: str,
        user_language: str,
    ) -> str:
        """Generate a naive assistant response.

        This placeholder implementation simply acknowledges the user's latest
        utterance. Replace with a real LLM integration when available.
        """
        del state  # not yet used in the placeholder response
        user_text = (user_text or "").strip()
        if not user_text:
            return "I did not catch that. Could you please repeat?"
        return f"Acknowledged ({user_language.lower()}): {user_text}"

    def _format_turns_for_summary(self, turns: Iterable[ConversationTurn]) -> Iterable[str]:
        for turn in turns:
            assistant_part = f" | assistant: {turn.assistant_response}" if turn.assistant_response else ""
            yield f"user[{turn.languages.get('user', 'unknown')}]: {turn.user_utterance_text}{assistant_part}"

