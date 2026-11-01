"""Clean and inspect UTF-8 text datasets for PyMind-A."""

import argparse
import unicodedata
from pathlib import Path


def clean_text(text: str) -> tuple[str, dict]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = []
    seen = set()
    duplicates = 0
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line in seen:
            duplicates += 1
            continue
        seen.add(line)
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    result = "\n".join(lines) + ("\n" if lines else "")
    report = {
        "input_characters": len(text),
        "output_characters": len(result),
        "input_lines": len(normalized.splitlines()),
        "output_lines": len(result.splitlines()),
        "duplicate_lines_removed": duplicates,
        "utf8_bytes": len(result.encode("utf-8")),
    }
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input UTF-8 text file")
    parser.add_argument("--output", required=True, help="Cleaned output text file")
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    if source.resolve() == destination.resolve():
        parser.error("input and output must be different files")
    raw = source.read_text(encoding="utf-8")
    cleaned, report = clean_text(raw)
    if not cleaned.strip():
        parser.error("No text remains after cleaning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")
    print(f"Saved cleaned dataset: {destination}")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
