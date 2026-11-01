from pymind.chat import build_chat_prompt


def test_build_chat_prompt_includes_recent_roles_and_current_message():
    history = [
        {"role": "user", "content": "வணக்கம்"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    prompt = build_chat_prompt("Explain Python", history)
    assert "System:" not in prompt
    assert prompt.startswith("User: வணக்கம்")
    assert "User: வணக்கம்" in prompt
    assert "Assistant: Hello" in prompt
    assert prompt.endswith("User: Explain Python\nAssistant:")


def test_build_chat_prompt_limits_history():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new"},
    ]
    prompt = build_chat_prompt("now", history, max_turns=1)
    assert "old" not in prompt
    assert "User: new" in prompt


def test_build_chat_prompt_rejects_empty_message():
    import pytest
    with pytest.raises(ValueError):
        build_chat_prompt("  ")
