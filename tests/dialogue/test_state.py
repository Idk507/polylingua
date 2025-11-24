from polylingua.dialogue.state import ConversationState
from polylingua.dialogue.turn import ConversationTurn


def test_conversation_state_add_and_recent() -> None:
    state = ConversationState(session_id="abc")
    for idx in range(6):
        state.add_turn(ConversationTurn(turn_id=f"t{idx}", user_utterance_text=f"text {idx}"))
    recent = state.recent_turns(limit=3)
    assert len(recent) == 3
    assert [turn.turn_id for turn in recent] == ["t3", "t4", "t5"]


def test_conversation_state_to_from_dict() -> None:
    state = ConversationState(session_id="session", user_language_preference="ES")
    state.add_turn(ConversationTurn(turn_id="1", user_utterance_text="hola"))
    payload = state.to_dict()
    restored = ConversationState.from_dict(payload)
    assert restored.session_id == "session"
    assert restored.user_language_preference == "es"
    assert restored.turn_count == 1
    assert restored.turns[0].user_utterance_text == "hola"
