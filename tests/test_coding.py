from pymind.coding import inspect_python_source


def test_inspector_summarizes_valid_python():
    report = inspect_python_source("import json\n\ndef greet(name):\n    return name\n\nclass Bot:\n    pass\n")
    assert report["syntax_valid"]
    assert report["imports"] == ["json"]
    assert report["functions"][0]["name"] == "greet"
    assert report["classes"][0]["name"] == "Bot"


def test_inspector_flags_dangerous_patterns_and_secret():
    source = "API_KEY = 'abcdefghijklmnop'\n\ndef run(code):\n    eval(code)\n"
    report = inspect_python_source(source)
    severities = [item["severity"] for item in report["findings"]]
    assert severities.count("high") >= 2


def test_inspector_reports_syntax_errors():
    report = inspect_python_source("def broken(:\n  pass\n", "broken.py")
    assert not report["syntax_valid"]
    assert report["syntax_error"]["line"] == 1


def test_inspector_flags_mutable_defaults_and_bare_except():
    report = inspect_python_source("def f(items=[]):\n    try:\n        pass\n    except:\n        pass\n")
    messages = " ".join(item["message"] for item in report["findings"])
    assert "mutable default" in messages
    assert "Bare except" in messages
