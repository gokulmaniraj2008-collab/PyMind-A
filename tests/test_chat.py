from pymind.chat import LocalChatEngine, build_chat_prompt


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


def test_faq_retrieval_exact_and_paraphrased_question(tmp_path):
    corpus = tmp_path / "faq.txt"
    corpus.write_text(
        "User: What is Python?\n"
        "Assistant: Python is a general-purpose programming language.\n"
        "User: Hello. Introduce yourself briefly.\n"
        "Assistant: Hello! I am PyMind-A, a small experimental language model built in Python.\n",
        encoding="utf-8",
    )
    engine = LocalChatEngine(
        checkpoint_path=tmp_path / "not-needed-for-faq.pt",
        corpus_path=corpus,
    )
    assert engine.retrieve_answer("What is Python?") == (
        "Python is a general-purpose programming language."
    )
    assert engine.retrieve_answer("Tell me about Python") == (
        "Python is a general-purpose programming language."
    )
    assert "PyMind-A" in engine.retrieve_answer("Hello. Introduce yourself briefly.")
