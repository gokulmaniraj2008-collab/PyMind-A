"""Small, CPU-friendly configuration for independently initialized models."""

from dataclasses import asdict, dataclass

from pymind.model import GPTConfig


@dataclass(frozen=True)
class IndependentConfig:
    """Configuration for a small decoder-only model initialized from random weights."""

    vocab_size: int = 256
    block_size: int = 32
    n_embd: int = 32
    n_head: int = 4
    n_layer: int = 1
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

    def to_gpt_config(self) -> GPTConfig:
        """Convert to the existing, backwards-compatible TinyGPT configuration."""
        return GPTConfig(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)
