from agents.query_understanding import _format_chat_history


def test_format_chat_history_handles_missing_role_key():
    # Regression: a malformed entry missing "role" used to raise KeyError,
    # crashing the whole node instead of falling back gracefully.
    history = [{"content": "hello"}, {"role": "user", "content": "hi"}]
    result = _format_chat_history(history)
    assert "hi" in result


def test_format_chat_history_empty():
    assert _format_chat_history([]) == "No previous conversation."


def test_format_chat_history_normal_case():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    result = _format_chat_history(history)
    assert "User: hi" in result
    assert "Assistant: hello" in result
