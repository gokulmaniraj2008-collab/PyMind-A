"""Conversation prompt utilities and local TinyGPT chat inference."""

from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


SYSTEM_PROMPT = (
    "You are PyMind, a helpful assistant. Reply clearly and briefly. "
    "Support Tamil and English. If you do not know something, say so."
)


def build_chat_prompt(message: str, history: list | None = None, max_turns: int = 4) -> str:
    """Build a bounded role-labelled prompt from recent conversation turns."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if max_turns < 0:
        raise ValueError("max_turns cannot be negative")
    lines = [f"System: {SYSTEM_PROMPT}"]
    valid_history = []
    for item in (history or []):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            valid_history.append((role.title(), content.strip()))
    for role, content in valid_history[-max_turns:] if max_turns else []:
        lines.append(f"{role}: {content}")
    lines.append(f"User: {message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


class LocalChatEngine:
    """Load and query a local PyMind-A UTF-8 byte-tokenizer checkpoint."""

    def __init__(self, checkpoint_path: str = "checkpoints/tiny_gpt.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = ByteTokenizer()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Train the model first using python -m pymind.train."
            )
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("tokenizer", {}).get("type") != ByteTokenizer.tokenizer_type:
            raise ValueError("Chat UI currently requires a UTF-8 byte-tokenizer checkpoint.")
        config = GPTConfig(**checkpoint["config"])
        self.model = TinyGPT(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def reply(
        self,
        message: str,
        history: list | None = None,
        temperature: float = 0.8,
        top_k: int = 40,
        max_new_tokens: int = 120,
    ) -> str:
        self.load()
        prompt = build_chat_prompt(message, history)
        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.05),
            top_k=top_k if top_k > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
        )[0].tolist()
        decoded = self.tokenizer.decode(output)
        answer = decoded[len(prompt):].strip()
        return answer or "I could not generate a readable response. Try a shorter message."
