from pymind.data_tools import clean_text


def test_clean_text_normalizes_unicode_and_removes_duplicate_lines():
    cleaned, report = clean_text("Cafe\u0301\r\nதமிழ்\r\nதமிழ்\r\n\r\nHello\n")
    assert cleaned == "Café\nதமிழ்\n\nHello\n"
    assert report["duplicate_lines_removed"] == 1
    assert report["utf8_bytes"] == len(cleaned.encode("utf-8"))


def test_clean_text_empty_input():
    cleaned, report = clean_text("")
    assert cleaned == ""
    assert report["output_lines"] == 0
