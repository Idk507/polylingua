from datetime import datetime

import pytest

from polylingua.dialogue.turn import ConversationTurn


def test_conversation_turn_round_trip() -> None:
    turn = ConversationTurn(
        turn_id="session-1",
        user_utterance_text="Hello",
        languages={"user": "en", "assistant": "es"},
        translations={"internal:en": "hello"},
        assistant_response="Hola",
        timestamp="2025-01-01T12:00:00Z",
    )
    payload = turn.to_dict()
    restored = ConversationTurn.from_dict(payload)
    assert restored.turn_id == "session-1"
    assert restored.user_utterance_text == "Hello"
    assert restored.languages["assistant"] == "es"
    assert restored.translations["internal:en"] == "hello"
    assert restored.assistant_response == "Hola"
    assert restored.timestamp.tzinfo is not None


def test_conversation_turn_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        ConversationTurn(turn_id="1", user_utterance_text="   ")


def test_conversation_turn_accepts_naive_datetime() -> None:
    naive = datetime(2025, 1, 1, 12, 0, 0)
    turn = ConversationTurn(turn_id="1", user_utterance_text="Hi", timestamp=naive)
    assert turn.timestamp.tzinfo is not None
    assert turn.timestamp.tzname() == "UTC"
