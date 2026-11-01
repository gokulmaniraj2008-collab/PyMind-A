"""Stable UTF-8 byte tokenizer (all Unicode text is supported)."""


class ByteTokenizer:
    """Encode text as UTF-8 bytes, mapping every byte to an ID from 0 to 255."""

    vocab_size = 256
    tokenizer_type = "utf8_bytes"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        values = bytes(int(token) for token in token_ids)
        return values.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": self.tokenizer_type, "vocab_size": self.vocab_size}
